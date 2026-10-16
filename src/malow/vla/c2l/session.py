"""Inference compatibility implementation; see SOURCE_NOTICE.md."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from PIL import Image

from . import blocks
from .encoder_inputs import build_context_encoder_inputs, build_context_encoder_text


def _to_pil(frame: Any) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame
    return Image.fromarray(np.asarray(frame))


def clone_interactive_state(state: Any) -> Any:
    """Deep-copy a `C2LInteractiveState` so a provisional block can be folded and dropped.

    `copy.deepcopy` rather than reaching into the cache: the linear-attention cache layout
    (`DynamicCache` and its per-layer conv/recurrent tensors) differs across Transformers
    versions, and deepcopy clones tensors on any device without depending on that.

    Copied **inside** `torch.inference_mode()` because that is where the originals were
    made: `interactive_update` is decorated with it, so the cache holds inference tensors,
    and the clone is only ever written to by another `interactive_update`. Copying them out
    here would create ordinary tensors from inference ones and leave the copy's in-place
    updates depending on how torch happens to treat that boundary.
    """
    with torch.inference_mode():
        return copy.deepcopy(state)


class C2LVLASession:
    """One episode's streaming C2L state.

    Call :meth:`step` once per env step, in order, with that step's observation. It
    returns the LoRA to run that step's action prediction under, or ``None`` at step 0.
    When ``interleave_action_history`` is on, call :meth:`record_action` with that step's
    predicted action before the next :meth:`step`.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: Any = None,
        interleave_timestamp: bool = True,
        interleave_text_prompt: bool = False,
        interleave_action_history: bool = False,
        action_dim: int | None = None,
        step_seconds: float = blocks.SIMPLER_BRIDGE_STEP_SECONDS,
        prompt_template: str = "{instruction}",
        encoder_text: str | None = None,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device
        self.interleave_timestamp = bool(interleave_timestamp)
        self.interleave_text_prompt = bool(interleave_text_prompt)
        self.interleave_action_history = bool(interleave_action_history)
        self.prompt_template = str(prompt_template)
        self.step_seconds = float(step_seconds)
        if not self.step_seconds > 0.0:
            raise ValueError(f"C2L-VLA session needs a positive step_seconds, got {step_seconds!r}.")
        if self.interleave_action_history and not action_dim:
            raise ValueError(
                "C2L-VLA action interleaving needs the action width; pass `action_dim`."
            )
        self.action_dim = int(action_dim) if action_dim else None
        self.encoder_text = encoder_text or build_context_encoder_text(processor)
        self.reset()

    def reset(self, instruction: str | None = None) -> None:
        """Start a new episode.

        Must be called at every episode boundary: a worker runs many tasks back to back
        through one long-lived model, and without this, episode n+1 starts out remembering
        episode n.
        """
        self.state = self.model.new_interactive_state()
        self.committed_blocks = 0
        self.next_step = 0
        self._frames: dict[int, Image.Image] = {}
        self._actions: dict[int, np.ndarray] = {}
        self.folded_blocks: list[str] = []
        self.instruction: str | None = None
        if instruction is not None:
            self.set_instruction(instruction)

    def set_instruction(self, instruction: str) -> None:
        """Set the episode instruction used by the optional text-prompt stream."""
        normalized = str(instruction or "").strip().lower()
        if self.interleave_text_prompt and not normalized:
            raise ValueError("C2L-VLA text-prompt interleaving requires a non-empty instruction.")
        if (
            self.instruction is not None
            and normalized != self.instruction
            and self.committed_blocks > 0
        ):
            raise ValueError(
                "C2L-VLA instruction changed after context blocks were already folded; "
                "reset the episode before changing the instruction."
            )
        self.instruction = normalized

    def _formatted_text_prompt(self) -> str:
        if self.instruction is None:
            raise ValueError(
                "C2L-VLA text-prompt interleaving has no episode instruction. Call "
                "reset(instruction) or set_instruction() before folding context."
            )
        return self.prompt_template.replace("{instruction}", self.instruction)

    # --- action history -------------------------------------------------------------

    def record_action(self, action: Any, step_index: int | None = None) -> None:
        """Record the action taken at env step ``step_index``.

        Frame ``i+1`` is paired with the action of step ``i`` when its block is folded,
        so it has to be here before that block is folded.

        ``step_index`` defaults to the step :meth:`step` was last called for, which is
        the self-recording path: the policy files its own prediction immediately after
        making it. The action-feedback path passes the index explicitly instead, because
        it records the *executed* action of the previous step at the top of the current
        one -- see `C2LVLAInteractivePolicy`. That is still early enough: the context at
        step ``t`` reads no action newer than ``a_{t-2}``.
        """
        if not self.interleave_action_history:
            return
        if step_index is None:
            if self.next_step == 0:
                raise RuntimeError("C2L-VLA session cannot record an action before its first step().")
            step_index = self.next_step - 1
        else:
            step_index = int(step_index)
            if step_index < 0:
                raise ValueError(f"C2L-VLA action step index must be non-negative, got {step_index}.")
        values = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_dim is not None and values.shape[0] != self.action_dim:
            raise ValueError(
                f"C2L-VLA session expects {self.action_dim}-dim actions, got {values.shape[0]}."
            )
        self._actions[step_index] = values

    def _block_actions(self, action_steps: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """``([1, frames, dim], [1, frames])`` for one block's action slots."""
        values = np.zeros((len(action_steps), self.action_dim), dtype=np.float32)
        is_empty = np.zeros((len(action_steps),), dtype=bool)
        for position, action_step in enumerate(action_steps):
            if action_step == blocks.EMPTY_ACTION_STEP:
                is_empty[position] = True
                continue
            if action_step not in self._actions:
                raise RuntimeError(
                    f"C2L-VLA session is missing its own prediction for env step {action_step}. "
                    "Actions are pruned to the three most recent, so `record_action()` must be "
                    "called once per `step()`, in order."
                )
            values[position] = self._actions[action_step]
        return values[None, ...], is_empty[None, ...]

    # --- folding --------------------------------------------------------------------

    def _fold(self, state: Any, frame_steps: list[int], action_steps: list[int], label: str) -> None:
        missing = [step for step in frame_steps if step not in self._frames]
        if missing:
            raise RuntimeError(
                f"C2L-VLA session is missing observations for env steps {missing}. Frames are "
                "pruned to the two most recent, so `step()` must be called once per env step, "
                "in order."
            )
        block_actions = block_action_is_empty = None
        if self.interleave_action_history:
            block_actions, block_action_is_empty = self._block_actions(action_steps)

        encoder_inputs = build_context_encoder_inputs(
            self.processor,
            [[self._frames[step] for step in frame_steps]],
            step_seconds=self.step_seconds,
            block_labels=[[label]] if self.interleave_timestamp else None,
            block_actions=[block_actions] if block_actions is not None else None,
            block_action_is_empty=(
                [block_action_is_empty] if block_action_is_empty is not None else None
            ),
            text_prompts=[self._formatted_text_prompt()] if self.interleave_text_prompt else None,
            encoder_text=self.encoder_text,
            device=self.device,
        )
        self.model.interactive_update(encoder_inputs, state)

    def _prune(self, step_index: int) -> None:
        for step in [step for step in self._frames if step < step_index - 1]:
            del self._frames[step]
        # The newest committed block folded at an even step t reads `a_{t-3}`.
        for step in [step for step in self._actions if step < step_index - 3]:
            del self._actions[step]

    def step(self, frame: Any) -> Any | None:
        """Observe env step ``t``'s frame and return the LoRA for that step's decision.

        The observation itself goes to the *base* model, not into the context: the context
        at step ``t`` is ``[0, t-1]``. It becomes context material from step ``t+1`` on.
        """
        step_index = self.next_step
        self._frames[step_index] = _to_pil(frame)
        self.next_step += 1

        target_blocks = blocks.committed_block_count(step_index)
        while self.committed_blocks < target_blocks:
            block_index = self.committed_blocks
            label = blocks.committed_block_label(block_index, self.step_seconds)
            self._fold(
                self.state,
                blocks.committed_block_frame_steps(block_index),
                blocks.committed_block_action_steps(block_index),
                label,
            )
            self.folded_blocks.append(label)
            self.committed_blocks += 1

        self._prune(step_index)

        if step_index == 0:
            return None

        if blocks.has_provisional_block(step_index):
            working_state = clone_interactive_state(self.state)
            self._fold(
                working_state,
                blocks.provisional_block_frame_steps(step_index),
                blocks.provisional_block_action_steps(step_index),
                blocks.provisional_block_label(step_index, self.step_seconds),
            )
        else:
            working_state = self.state

        lora_dict, _layernorms = self.model.interactive_generate_weights(working_state)
        return lora_dict

    def block_labels_at(self, step_index: int) -> list[str]:
        """Block stream this session would present at ``step_index``; for verification."""
        return blocks.context_block_labels(step_index, self.step_seconds)
