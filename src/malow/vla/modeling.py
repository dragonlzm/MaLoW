from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import torch
from torch import nn

from malow.vla.logging_utils import get_logger
from .inputs import (
    LEGACY_SPECIAL_ACTION_TOKEN_IDS,
    LEGACY_VISION_TOKEN_IDS,
    SPECIAL_ACTION_TOKENS,
    VISION_TOKENS,
    build_qwen_inputs,
    move_qwen_inputs_to_device,
    token_ids,
)
from .action_head import get_action_model


logger = get_logger(__name__)


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _get_hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", config)
    return int(getattr(text_config, "hidden_size"))




def _supported_predict_kwargs(action_model: nn.Module, **kwargs: Any) -> dict[str, Any]:
    """Drop `predict()` keyword arguments the action head does not accept.

    `steps` and `output_prior_action` are flow-specific; the MLP heads read the
    condition alone. `output_prior_action=True` asks for a quantity such a head
    cannot produce, so that case is an error rather than a silent fallback.
    """
    parameters = inspect.signature(action_model.predict).parameters
    if any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    supported = {name: value for name, value in kwargs.items() if name in parameters}
    if kwargs.get("output_prior_action") and "output_prior_action" not in supported:
        raise ValueError(
            f"{type(action_model).__name__}.predict does not support `output_prior_action`; "
            "it has no prior action head."
        )
    return supported


def _shift_mask_to_predicting_positions(token_mask: torch.Tensor) -> torch.Tensor:
    shifted_mask = torch.zeros_like(token_mask)
    shifted_mask[:, :-1] = token_mask[:, 1:]
    return shifted_mask


class VLAModelWrapper(nn.Module):
    def __init__(self, qwen_model: nn.Module, processor: Any, vla_config: Any) -> None:
        super().__init__()
        self.qwen_model = qwen_model
        self.processor = processor
        self.vla_config = vla_config
        self.action_model = get_action_model(config=vla_config)

        hidden_size = _get_hidden_size(qwen_model)
        value_q_dim = int(getattr(vla_config.framework.action_model, "value_q_dim", hidden_size))
        if value_q_dim == hidden_size:
            self.special_token_projection = nn.Identity()
        else:
            self.special_token_projection = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, value_q_dim),
            )

        layer_qformer_cfg = vla_config.framework.layer_qformer
        if int(getattr(layer_qformer_cfg, "num_query_tokens", -1)) != -1:
            raise ValueError("VLA v1 only supports `framework.layer_qformer.num_query_tokens: -1`.")

        dino_backbone = str(getattr(vla_config.framework.dino, "dino_backbone", "None"))
        if dino_backbone != "None":
            raise ValueError("VLA v1 only supports `framework.dino.dino_backbone: None`.")

        self.future_action_window_size = int(vla_config.framework.action_model.future_action_window_size)
        self.num_action_tokens = self.future_action_window_size + 1
        self.condition_on_predicting_special_token = bool(
            getattr(vla_config.framework, "condition_on_predicting_special_token", False)
        )
        self.use_learned_action_token_embeddings = bool(
            getattr(vla_config.framework, "use_learned_action_token_embeddings", False)
        )
        if self.use_learned_action_token_embeddings and self.condition_on_predicting_special_token:
            raise ValueError(
                "`use_learned_action_token_embeddings` is incompatible with "
                "`condition_on_predicting_special_token` in a causal decoder."
            )
        if self.use_learned_action_token_embeddings:
            self.special_action_embeddings = nn.Parameter(torch.empty(self.num_action_tokens, hidden_size))
            nn.init.normal_(self.special_action_embeddings, mean=0.0, std=hidden_size**-0.5)

    def _build_qwen_inputs(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        vla_data_cfg = self.vla_config.datasets.vla_data
        prompt_template = getattr(vla_data_cfg, "CoT_prompt", "{instruction}")
        return build_qwen_inputs(
            examples=examples,
            processor=self.processor,
            prompt_template=prompt_template,
            future_action_window_size=self.future_action_window_size,
            use_learned_action_token_embeddings=self.use_learned_action_token_embeddings,
            device=_model_device(self.qwen_model),
        )

    @staticmethod
    def _select_token_features(
        hidden_states: tuple[torch.Tensor, ...],
        token_mask: torch.Tensor,
        start_layer: int,
        end_layer: int,
    ) -> list[torch.Tensor]:
        selected_layers = []
        for layer_features in hidden_states[start_layer:end_layer]:
            per_sample = []
            expected_count = int(token_mask[0].sum().item())
            for batch_idx in range(layer_features.shape[0]):
                indices = token_mask[batch_idx].nonzero(as_tuple=True)[0]
                if indices.numel() != expected_count:
                    raise ValueError("VLA token feature selection requires equal token counts across the batch.")
                per_sample.append(layer_features[batch_idx, indices, :])
            selected_layers.append(torch.stack(per_sample, dim=0))

        return selected_layers

    @staticmethod
    def _validate_token_count(token_mask: torch.Tensor, expected_count: int, context: str) -> None:
        counts = token_mask.sum(dim=1)
        if not torch.all(counts == expected_count):
            raise ValueError(
                f"{context} requires exactly {expected_count} action marker tokens per sample. "
                f"Got counts: {counts.detach().cpu().tolist()}."
            )

    def _build_inputs_embeds_with_learned_action_tokens(
        self,
        qwen_inputs: dict[str, torch.Tensor],
        action_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_token_count(
            action_token_mask,
            self.num_action_tokens,
            "Learned VLA action-token embedding replacement",
        )
        input_embeddings = self.qwen_model.get_input_embeddings()
        inputs_embeds = input_embeddings(qwen_inputs["input_ids"]).clone()
        learned_embeddings = self.special_action_embeddings.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
        for batch_idx in range(inputs_embeds.shape[0]):
            indices = action_token_mask[batch_idx].nonzero(as_tuple=True)[0]
            inputs_embeds[batch_idx, indices, :] = learned_embeddings

        return inputs_embeds


    def _forward_qwen(self, model_inputs: dict[str, Any], **_: Any) -> Any:
        """Run the language model over the base branch.

        Extracted as its own method so `C2LVLAModelWrapper` can route the same call
        through the modulated model, which additionally needs the context-encoder inputs.
        Unknown keyword arguments are ignored here, preserving the `**_` this signature
        replaced.
        """
        return self.qwen_model(
            **model_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

    def _get_flow_condition(
        self,
        examples: list[dict[str, Any]],
        qwen_inputs: dict[str, torch.Tensor] | None = None,
        **forward_kwargs: Any,
    ) -> tuple[dict[str, torch.Tensor], Any, torch.Tensor]:
        if qwen_inputs is None:
            qwen_inputs = self._build_qwen_inputs(examples)
        else:
            qwen_inputs = move_qwen_inputs_to_device(qwen_inputs, _model_device(self.qwen_model))

        labels = qwen_inputs["labels"]
        action_token_ids = torch.tensor(
            token_ids(self.processor.tokenizer, SPECIAL_ACTION_TOKENS, LEGACY_SPECIAL_ACTION_TOKEN_IDS),
            device=labels.device,
        )
        special_mask = torch.isin(labels, action_token_ids)

        if self.use_learned_action_token_embeddings:
            model_inputs = {
                k: v for k, v in qwen_inputs.items() if k not in {"input_ids", "labels"}
            }
            model_inputs["inputs_embeds"] = self._build_inputs_embeds_with_learned_action_tokens(
                qwen_inputs, special_mask
            )
        else:
            model_inputs = {k: v for k, v in qwen_inputs.items() if k != "labels"}

        qwen_outputs = self._forward_qwen(model_inputs, **forward_kwargs)

        if self.condition_on_predicting_special_token:
            special_mask = _shift_mask_to_predicting_positions(special_mask)
        elif self.use_learned_action_token_embeddings:
            self._validate_token_count(
                special_mask,
                self.num_action_tokens,
                "Learned VLA action-token feature selection",
            )

        special_features = self._select_token_features(
            qwen_outputs.hidden_states,
            special_mask,
            int(self.vla_config.framework.special_token_start_layer),
            int(self.vla_config.framework.special_token_end_layer),
        )

        input_ids = qwen_inputs["input_ids"]
        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for token_id in token_ids(self.processor.tokenizer, VISION_TOKENS, LEGACY_VISION_TOKEN_IDS):
            vision_mask |= input_ids == token_id
        _ = self._select_token_features(
            qwen_outputs.hidden_states,
            vision_mask,
            int(self.vla_config.framework.vision_token_start_layer),
            int(self.vla_config.framework.vision_token_end_layer),
        )

        flow_condition = self.special_token_projection(special_features[0])
        action_dtype = next(self.action_model.parameters()).dtype
        flow_condition = flow_condition.to(dtype=action_dtype)

        return qwen_inputs, qwen_outputs, flow_condition

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: list[list[Any]],
        instructions: list[str],
        num_ddim_steps: int | None = None,
        output_prior_action: bool = False,
        **forward_kwargs: Any,
    ) -> dict[str, np.ndarray]:
        examples = [
            {"image": images, "lang": instruction}
            for images, instruction in zip(batch_images, instructions, strict=True)
        ]
        _, _, flow_condition = self._get_flow_condition(examples, **forward_kwargs)
        steps = num_ddim_steps
        if steps is None:
            steps = int(getattr(self.vla_config.framework.action_model, "num_ddim_steps", 10))

        predict_kwargs = _supported_predict_kwargs(
            self.action_model,
            steps=steps,
            output_prior_action=output_prior_action,
        )
        normalized_actions = self.action_model.predict(condition=flow_condition, **predict_kwargs)
        return {"normalized_actions": normalized_actions.float().cpu().numpy()}
