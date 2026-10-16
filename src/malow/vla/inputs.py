from __future__ import annotations

from typing import Any

import torch
from qwen_vl_utils import process_vision_info

IGNORE_INDEX = -100
from .rope2d import get_rope_index_3


SPECIAL_ACTION_TOKENS = ("<think>", "</think>", "<tool_response>", "</tool_response>")
LEGACY_SPECIAL_ACTION_TOKEN_IDS = (151665, 151666, 151667, 151668)
VISION_TOKENS = ("<|image_pad|>",)
LEGACY_VISION_TOKEN_IDS = (151655, 248056)
LEGACY_ASSISTANT_TOKEN_IDS = (77091,)
LEGACY_IM_START_TOKEN_IDS = (151644, 248045)
LEGACY_IM_END_TOKEN_IDS = (151645, 248046)


def token_ids(tokenizer: Any, tokens: tuple[str, ...], fallback_ids: tuple[int, ...]) -> tuple[int, ...]:
    token_ids = list(fallback_ids)
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_token_id:
            token_ids.append(token_id)

    return tuple(dict.fromkeys(token_ids))


# Qwen3.5's chat template renders the last assistant turn as
#
#     <|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n{content}
#
# so it *injects* an empty reasoning block whenever the answer carries no
# `</think>` of its own. Those two tokens are action markers here, and the action
# head takes one hidden state per marker, so an 8-action LIBERO chunk renders as
# 10 markers and the MSE fails with "size of tensor a (10) must match ... b (8)".
#
# The tool-response answers are unaffected: they contain `</think>`, so the
# template's reasoning-extraction branch moves their own `<think></think>` pair
# into the reasoning slot and re-emits it, leaving the marker count unchanged.
# Qwen3-VL's template emits the answer verbatim and never injects anything.
INJECTED_EMPTY_REASONING_BLOCK = "<think>\n\n</think>\n\n"
ASSISTANT_TURN_PREFIX = "<|im_start|>assistant\n"


def strip_injected_reasoning_block(text: str, target_text: str) -> str:
    """Inference compatibility implementation; see SOURCE_NOTICE.md."""
    head, separator, assistant_turn = text.rpartition(ASSISTANT_TURN_PREFIX)
    if not separator:
        return text

    if not assistant_turn.startswith(INJECTED_EMPTY_REASONING_BLOCK + target_text):
        return text

    return head + separator + assistant_turn[len(INJECTED_EMPTY_REASONING_BLOCK) :]


def build_qwen_inputs(
    examples: list[dict[str, Any]],
    processor: Any,
    prompt_template: str,
    future_action_window_size: int,
    use_learned_action_token_embeddings: bool = False,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    messages = []
    target_texts = []
    for example in examples:
        images = example["image"]
        instruction = str(example.get("lang") or example.get("language") or "").lower()
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt_template.replace("{instruction}", instruction)})

        num_steps = future_action_window_size
        if use_learned_action_token_embeddings:
            target_text = "The expected actions are: " + "<think>" * (num_steps + 1)
        elif "</think>" in prompt_template and "</tool_response>" in prompt_template:
            if num_steps == 3:
                target_text = "The expected actions are: <think></think><tool_response></tool_response>"
            elif num_steps == 7:
                target_text = (
                    "The expected actions are: "
                    "<think><think></think></think><tool_response><tool_response></tool_response></tool_response>"
                )
            else:
                raise ValueError("Tool-response VLA prompt currently supports future_action_window_size 3 or 7.")
        else:
            target_text = "The expected actions are: " + "<think>" * (num_steps + 1)

        target_texts.append(target_text)
        messages.append(
            [
                {"role": "user", "content": content},
                {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
            ]
        )

    texts = [
        strip_injected_reasoning_block(
            processor.apply_chat_template(message, tokenize=False, add_generation_prompt=False), target_text
        )
        for message, target_text in zip(messages, target_texts, strict=True)
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")

    input_ids = inputs["input_ids"]
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    tokenizer = processor.tokenizer
    assistant_token_ids = set(token_ids(tokenizer, ("assistant",), LEGACY_ASSISTANT_TOKEN_IDS))
    im_start_token_ids = set(token_ids(tokenizer, ("<|im_start|>",), LEGACY_IM_START_TOKEN_IDS))
    im_end_token_ids = set(token_ids(tokenizer, ("<|im_end|>",), LEGACY_IM_END_TOKEN_IDS))
    for batch_idx in range(input_ids.shape[0]):
        ids = input_ids[batch_idx].tolist()
        pos = 0
        while pos < len(ids):
            if ids[pos] in assistant_token_ids and pos > 0 and ids[pos - 1] in im_start_token_ids:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < len(ids) and ids[ans_end] not in im_end_token_ids:
                    ans_end += 1
                if ans_end < len(ids):
                    labels[batch_idx, ans_start : ans_end + 2] = input_ids[batch_idx, ans_start : ans_end + 2]
                    pos = ans_end
            pos += 1
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = IGNORE_INDEX
    inputs["labels"] = labels

    merge_size = getattr(processor.image_processor, "merge_size", 2)
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")
    second_per_grid_ts = None
    if video_grid_thw is not None:
        second_per_grid_ts = [
            processor.video_processor.temporal_patch_size / processor.video_processor.fps
        ] * len(video_grid_thw)

    position_ids, _ = get_rope_index_3(
        spatial_merge_size=merge_size,
        input_ids=inputs["input_ids"],
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        attention_mask=inputs.get("attention_mask"),
    )
    inputs["position_ids"] = position_ids

    qwen_inputs = dict(inputs)
    if device is not None:
        return move_qwen_inputs_to_device(qwen_inputs, device)

    return qwen_inputs


def move_qwen_inputs_to_device(
    qwen_inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in qwen_inputs.items()}
