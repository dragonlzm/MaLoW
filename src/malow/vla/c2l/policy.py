"""Interactive C2L-VLA policy: one streaming session behind a `predict_action` call.

The SimplerEnv rollout is closed loop -- one model call per env step -- so this object
receives observations one at a time and keeps the context between them. Everything about
the context lives here; the model itself stays stateless.

`reset` is not optional bookkeeping. `maniskill2_evaluator` calls the interface's
`reset(task_description)` once per episode and then steps; a worker runs many episodes
back to back through one long-lived server process, so without a reset that reaches this
object, episode n+1 begins remembering episode n.

The action history folded into that context is this policy's own prediction by default.
`action_feedback=True` switches it to the action the environment actually executed --
the client-side ensembler's blend, re-normalized and returned with the following
request. See `_accept_executed_action` for the protocol and `session.py` for why the
schedule leaves room for it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from malow.vla.logging_utils import get_logger
from .blocks import SIMPLER_BRIDGE_STEP_SECONDS
from .session import C2LVLASession


logger = get_logger(__name__)


class C2LVLAInteractivePolicy:
    """Adapts a `C2LVLAModelWrapper` to the websocket policy server's contract."""

    def __init__(
        self,
        model: Any,
        processor: Any,
        device: Any = None,
        step_seconds: float = SIMPLER_BRIDGE_STEP_SECONDS,
        action_feedback: bool = False,
    ) -> None:
        self.model = model
        self.processor = processor
        self.device = device or next(model.parameters()).device
        modulated_model = model.qwen_model
        interleave_text_prompt = bool(
            getattr(modulated_model, "interleave_text_prompt", False)
        )
        interleave_action_history = bool(
            getattr(modulated_model, "interleave_action_history", False)
        )
        action_dim = None
        if interleave_action_history:
            encoder = getattr(modulated_model.hypernet, "action_history_encoder", None)
            if encoder is None:
                raise ValueError(
                    "This checkpoint records interleave_action_history=True but carries no "
                    "action encoder; it cannot be evaluated."
                )
            action_dim = int(encoder.action_dim)
        self.session = C2LVLASession(
            model=modulated_model,
            processor=processor,
            device=self.device,
            interleave_timestamp=bool(getattr(modulated_model, "interleave_timestamp", True)),
            interleave_text_prompt=interleave_text_prompt,
            interleave_action_history=interleave_action_history,
            action_dim=action_dim,
            step_seconds=step_seconds,
            prompt_template=getattr(
                model.vla_config.datasets.vla_data, "CoT_prompt", "{instruction}"
            ),
        )
        self.action_feedback = bool(action_feedback)
        if self.action_feedback and not interleave_action_history:
            raise ValueError(
                "action_feedback needs an action-history checkpoint, but this one records "
                "interleave_action_history=False, so there is no action slot to feed. Drop "
                "--c2l_action_feedback / VLA_C2L_ACTION_FEEDBACK."
            )
        self.episodes = 0
        logger.info_rank0(
            f"C2L-VLA interactive policy: step_seconds={self.session.step_seconds}, "
            f"interleave_timestamp={self.session.interleave_timestamp}, "
            f"interleave_text_prompt={interleave_text_prompt}, "
            f"interleave_action_history={interleave_action_history} (action_dim={action_dim}), "
            f"action_feedback={self.action_feedback}"
        )

    def reset(self, instruction: str | None = None, **_: Any) -> dict[str, Any]:
        self.session.reset(instruction)
        self.episodes += 1
        logger.info_rank0(f"C2L-VLA session reset (episode {self.episodes}): {instruction!r}")
        return {"reset": True, "episode": self.episodes}

    def predict_action(
        self,
        batch_images: list[list[Any]],
        instructions: list[str],
        num_ddim_steps: int | None = None,
        output_prior_action: bool = False,
        executed_action: Any = None,
        executed_step: int | None = None,
        **_: Any,
    ) -> dict[str, np.ndarray]:
        if len(batch_images) != 1:
            raise ValueError(
                "Interactive C2L-VLA holds one episode's context, so it serves one "
                f"observation per call; got a batch of {len(batch_images)}."
            )
        frames = batch_images[0]
        if len(frames) != 1:
            raise ValueError(
                "The C2L-VLA base branch takes the current observation only; got "
                f"{len(frames)} images for this step."
            )

        if self.session.interleave_text_prompt:
            if len(instructions) != 1:
                raise ValueError(
                    "Interactive C2L-VLA text-prompt conditioning expects one instruction, "
                    f"got {len(instructions)}."
                )
            self.session.set_instruction(instructions[0])

        self._accept_executed_action(executed_action, executed_step)

        generated_loras = self.session.step(frames[0])
        step_index = self.session.next_step - 1

        if generated_loras is None:
            # Env step 0: no history exists yet, so this one step runs as the stage-1
            # policy rather than under a LoRA generated from nothing.
            forward_kwargs = {"unmodulated": True}
        else:
            forward_kwargs = {"generated_loras": generated_loras}

        outputs = self.model.predict_action(
            batch_images=batch_images,
            instructions=instructions,
            num_ddim_steps=num_ddim_steps,
            output_prior_action=output_prior_action,
            **forward_kwargs,
        )
        # The context at step t+1 pairs frame t with the action of step t, so it has to be
        # recorded. Two ways to get it, and the default is the first:
        #
        #   self-recording (default) -- take `normalized_actions[0, 0]`, this step's own
        #     predicted action, right here. It is *not* what the environment executes: the
        #     client-side ensembler blends this chunk with earlier ones first, and that
        #     blend never comes back over the wire.
        #   action feedback (opt-in) -- do nothing here. The client re-normalizes the
        #     action it actually executed and sends it with the *next* request, where
        #     `_accept_executed_action` files it under the same step index. Recording our
        #     own prediction as well would overwrite the executed one at step t with the
        #     unensembled one, so this branch is exclusive.
        if self.session.interleave_action_history and not self.action_feedback:
            self.session.record_action(np.asarray(outputs["normalized_actions"])[0, 0])

        outputs["c2l_step"] = np.asarray([step_index], dtype=np.int64)
        outputs["c2l_blocks"] = np.asarray([self.session.committed_blocks], dtype=np.int64)
        return outputs

    def _accept_executed_action(self, executed_action: Any, executed_step: int | None) -> None:
        """File the action the environment actually executed at the previous env step.

        Called before `session.step()` for the current step, which is early enough by two
        steps: the context at step ``t`` reads no action newer than ``a_{t-2}``, and this
        supplies ``a_{t-1}``.

        Every disagreement between client and server is fatal here rather than silent. A
        missing, stale, or duplicated feedback message would shift the whole action
        history by a step and produce a plausible-looking number that means nothing, and
        `predict_action`'s `**_` would otherwise swallow the evidence.
        """
        expected_step = self.session.next_step - 1

        if not self.action_feedback:
            if executed_action is not None:
                raise RuntimeError(
                    "Client sent an executed action but this server was started without "
                    "--c2l_action_feedback, so the session would fold its own predictions "
                    "instead. Start the server with the flag (or VLA_C2L_ACTION_FEEDBACK=1), "
                    "or drop the flag on the client."
                )
            return

        if expected_step < 0:
            # First observation of the episode: nothing has been executed yet.
            if executed_action is not None:
                raise RuntimeError(
                    f"Client sent an executed action for step {executed_step} at the first "
                    "observation of an episode, where no action has been taken yet. The "
                    "per-episode reset most likely did not reach the client's feedback state."
                )
            return

        if executed_action is None:
            raise RuntimeError(
                f"Action feedback is on, but the request for env step {self.session.next_step} "
                f"carried no executed action for step {expected_step}. The client must send "
                "`executed_action`/`executed_step` on every request after the first of an "
                "episode."
            )
        if executed_step is None:
            raise RuntimeError(
                "Action feedback needs an explicit `executed_step` alongside "
                "`executed_action`; without it a dropped message would silently misalign "
                "the action history."
            )
        if int(executed_step) != expected_step:
            raise RuntimeError(
                f"Action-feedback step mismatch: client sent step {int(executed_step)}, server "
                f"expected {expected_step}. The stream is out of sync -- a request or a reset "
                "was dropped -- and the folded action history would be shifted."
            )

        self.session.record_action(executed_action, step_index=expected_step)
