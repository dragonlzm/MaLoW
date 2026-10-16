"""Inference compatibility implementation; see SOURCE_NOTICE.md."""

from __future__ import annotations


# SimplerEnv's WidowX bridge setup runs its control loop at 5 Hz, which is also the
# rate BridgeData V2 was recorded at. Used as the evaluation-side default only; the
SIMPLER_BRIDGE_STEP_SECONDS = 0.2

# Qwen3.5 vision tower `temporal_patch_size`; also the C2L `frame_per_lora`.
FRAMES_PER_BLOCK = 2

# Action slot for "the step before the episode began", which has no action. Kept as a
# sentinel index rather than a separate mask so the frame and action streams stay the
# same shape and index the same way.
EMPTY_ACTION_STEP = -1


def _check_step_seconds(step_seconds: float) -> float:
    value = float(step_seconds)
    if not value > 0.0:
        raise ValueError(f"C2L-VLA step_seconds must be positive, got {step_seconds!r}.")
    return value


def context_frame_steps(step: int) -> list[int]:
    """Env-step indices feeding the context encoder for an observation at ``step``.

    ``[0, step-1]``, padded by repeating the last frame when the count is odd so the
    stream divides evenly into two-frame blocks. Empty at ``step == 0``.
    """
    if step < 0:
        raise ValueError(f"C2L-VLA context requires a non-negative step, got {step}.")
    if step == 0:
        return []

    frames = list(range(step))
    if len(frames) % FRAMES_PER_BLOCK:
        frames.extend([frames[-1]] * (FRAMES_PER_BLOCK - len(frames) % FRAMES_PER_BLOCK))
    return frames


def context_action_steps(step: int) -> list[int]:
    """Action indices paired with :func:`context_frame_steps`, one per context frame.

    Frame ``i`` is paired with ``a_{i-1}`` -- the action executed at the previous step,
    i.e. the one that produced this frame. Frame 0 pairs with
    :data:`EMPTY_ACTION_STEP`, which no episode has. Because the pairing is a pure
    function of the frame index, the padded frames automatically repeat their action.
    """
    return [frame_step - 1 for frame_step in context_frame_steps(step)]


def context_frame_times(step: int, step_seconds: float) -> list[float]:
    """Absolute timestamps, in seconds, of :func:`context_frame_steps`."""
    step_seconds = _check_step_seconds(step_seconds)
    return [index * step_seconds for index in context_frame_steps(step)]


def context_terminal_time(step: int, step_seconds: float) -> float:
    """End of the context window: "now", i.e. the current observation's timestamp.

    Only the final block of an odd-length history reads this; every other block's end
    is the start of the block after it.
    """
    return step * _check_step_seconds(step_seconds)


def block_count(step: int) -> int:
    """Number of temporal blocks in the context for an observation at ``step``."""
    return len(context_frame_steps(step)) // FRAMES_PER_BLOCK


def committed_block_count(step: int) -> int:
    """Blocks whose two frames are both real, so their content and label are final."""
    return step // FRAMES_PER_BLOCK


def has_provisional_block(step: int) -> bool:
    """Whether the newest block is the padded ``[x_{t-1}, x_{t-1}]`` one."""
    return bool(step % FRAMES_PER_BLOCK)


def committed_block_frame_steps(block_index: int) -> list[int]:
    """Env-step indices of committed block ``block_index``."""
    if block_index < 0:
        raise ValueError(f"C2L-VLA block index must be non-negative, got {block_index}.")
    start = block_index * FRAMES_PER_BLOCK
    return list(range(start, start + FRAMES_PER_BLOCK))


def committed_block_action_steps(block_index: int) -> list[int]:
    """Action indices of committed block ``block_index``: one per frame, ``a_{i-1}``."""
    return [frame_step - 1 for frame_step in committed_block_frame_steps(block_index)]


def provisional_block_frame_steps(step: int) -> list[int]:
    """Env-step indices of the padded final block at an odd ``step``."""
    if not has_provisional_block(step):
        raise ValueError(f"Step {step} has no provisional block; its history is even-length.")
    return [step - 1] * FRAMES_PER_BLOCK


def provisional_block_action_steps(step: int) -> list[int]:
    """Action indices of the padded final block at an odd ``step``."""
    return [frame_step - 1 for frame_step in provisional_block_frame_steps(step)]


def block_timestamp_label(start_seconds: float, end_seconds: float) -> str:
    """Format one block's label exactly as the offline C2L collator does."""
    return f"{start_seconds:g}s-{end_seconds:g}s"


def committed_block_label(block_index: int, step_seconds: float) -> str:
    """Label of committed block ``block_index`` -- independent of the current step."""
    step_seconds = _check_step_seconds(step_seconds)
    start = block_index * FRAMES_PER_BLOCK * step_seconds
    return block_timestamp_label(start, start + FRAMES_PER_BLOCK * step_seconds)


def provisional_block_label(step: int, step_seconds: float) -> str:
    """Label of the padded final block at an odd ``step``."""
    if not has_provisional_block(step):
        raise ValueError(f"Step {step} has no provisional block; its history is even-length.")
    step_seconds = _check_step_seconds(step_seconds)
    return block_timestamp_label((step - 1) * step_seconds, step * step_seconds)


def context_block_labels(step: int, step_seconds: float) -> list[str]:
    """Every block label for the context at ``step``, oldest first.

    Equivalent to running the offline collator's
    ``_timestamp_labels_from_frame_times(context_frame_times(step, dt), 2,
    context_terminal_time(step, dt))``; `test_c2l_vla_blocks.py` asserts that equality
    so the streaming path cannot drift from the offline one.
    """
    step_seconds = _check_step_seconds(step_seconds)
    labels = [
        committed_block_label(index, step_seconds) for index in range(committed_block_count(step))
    ]
    if has_provisional_block(step):
        labels.append(provisional_block_label(step, step_seconds))
    return labels


def context_block_action_steps(step: int) -> list[list[int]]:
    """Per-block action indices for the context at ``step``, oldest first.

    One inner list of :data:`FRAMES_PER_BLOCK` entries per block, aligned with
    :func:`context_block_labels`. Entries equal to :data:`EMPTY_ACTION_STEP` mark the
    pre-episode slot.
    """
    action_steps = context_action_steps(step)
    return [
        action_steps[start : start + FRAMES_PER_BLOCK]
        for start in range(0, len(action_steps), FRAMES_PER_BLOCK)
    ]
