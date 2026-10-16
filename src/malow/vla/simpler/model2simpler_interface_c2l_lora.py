"""Inference compatibility implementation; see SOURCE_NOTICE.md."""

from typing import Dict, Optional

import numpy as np

from malow.vla.simpler.model2simpler_interface_lora import M1InferenceLora


# Position of the gripper flag in the 7-dim action, matching `unnormalize_actions` and the
GRIPPER_DIM = 6


class M1InferenceC2LLora(M1InferenceLora):
    """`M1InferenceLora` whose episode reset propagates to the stateful policy server,
    and which can optionally feed the executed action back for the next step's LoRA."""

    def __init__(self, *args, action_feedback: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_feedback = bool(action_feedback)
        # `_pending_*` carry one step of executed action from `_on_action_executed` to the
        # next `_extra_infer_payload`. `_env_step` is the index that action belongs to.
        self._pending_executed_action: Optional[np.ndarray] = None
        self._pending_executed_step: Optional[int] = None
        self._env_step = 0

        if self.action_feedback:
            metadata = self.client.get_server_metadata() or {}
            if not metadata.get("action_feedback", False):
                # The server ignores unknown payload keys, so without this check the run
                # would look healthy while quietly scoring the default behaviour.
                raise RuntimeError(
                    "Action feedback was requested, but the policy server did not advertise "
                    "it (metadata: %r). Start the server with --c2l_action_feedback, or set "
                    "VLA_C2L_ACTION_FEEDBACK=1 for both processes." % (metadata,)
                )
            print("*** C2L action feedback ON: folding the executed (ensembled) action ***")

    def reset(self, task_description: str) -> None:
        super().reset(task_description)
        # Clear before the server reset, so an exception below cannot leave a stale action
        # from the previous episode queued for the next one's first request.
        self._pending_executed_action = None
        self._pending_executed_step = None
        self._env_step = 0
        try:
            self.client.reset(task_description)
        except Exception as exc:  # noqa: BLE001
            # A silently un-reset server leaks one episode's context into the next and
            # quietly degrades every later episode, which is far worse than stopping here.
            raise RuntimeError(
                "Failed to reset the interactive C2L-VLA policy server; the streaming "
                "context would carry over into the next episode."
            ) from exc

    def _extra_infer_payload(self) -> Dict:
        """Attach the previous step's executed action, if feedback is on and one is queued.

        Returns `{}` on the first request of an episode (nothing executed yet) and
        whenever feedback is off, which keeps the default wire payload byte-identical.
        """
        if not self.action_feedback or self._pending_executed_action is None:
            return {}
        return {
            "executed_action": self._pending_executed_action,
            "executed_step": self._pending_executed_step,
        }

    def _on_action_executed(self, executed_raw_action: np.ndarray, raw_action: Dict, action: Dict) -> None:
        """Inference compatibility implementation; see SOURCE_NOTICE.md."""
        if not self.action_feedback:
            return

        normalized = self.normalize_actions(
            np.asarray(executed_raw_action, dtype=np.float32).reshape(-1),
            self.action_norm_stats,
        ).astype(np.float32)
        normalized[GRIPPER_DIM] = float(np.asarray(raw_action["open_gripper"]).reshape(-1)[0] > 0.5)

        self._pending_executed_action = normalized
        self._pending_executed_step = self._env_step
        self._env_step += 1
