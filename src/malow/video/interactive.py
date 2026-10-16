# Interactive inference adapted from Hippocampus master f4b15d9. See NOTICE.
"""Interactive (streaming) Video Context-to-LoRA evaluation.

Offline evaluation answers one question from one whole-video pass. The interactive
protocol instead interleaves context and questions,

    [0s, t1] Q1/A1  (t1, t2] Q2/A2  (t2, t3] Q3/A3 ...

so at question ``k`` the model must have consumed everything up to ``t_k`` while having
paid only for the *new* footage. Both branches of the C2L temporal module support that
exactly:

* the Gated DeltaNet is a linear-attention recurrence, so carrying its
  ``(conv_state, recurrent_state)`` across segments reproduces the single-pass last token;
* the skip connection is a mean over block queries, so a running ``(sum, count)`` reproduces
  the single-pass mean.

The remaining piece is frame selection. Offline sampling is uniform over the current
prefix, which reshuffles previously chosen frames every time the prefix grows. Interactive
mode instead samples a **fixed grid**: frame ``i`` always sits at ``origin + i / sample_fps``
and segment boundaries land on whole ``frame_per_lora`` blocks, so the block stream seen
after ``k`` segments is exactly the block stream a single pass over ``[origin, t_k - tail]``
would produce. ``verify_session_against_single_pass`` checks that claim at runtime.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable

from malow.video.sampling import (
    add_c2l_timestamp_inputs,
    load_video_frames_at_times,
    processor_predecoded_video_inputs,
    video_duration_seconds,
)


@dataclass(frozen=True)
class InteractiveConfig:
    """Fixed-grid encoder sampling policy for one interactive run."""

    frame_per_lora: int = 2
    sample_fps: float = 1.0
    tail_seconds: float = 2.0
    interleave_timestamp: bool = False
    max_encoder_frames: int | None = None
    # Largest context segment handed to the encoder in one forward. A catch-up longer than
    # this is split into consecutive sub-segments folded into the same streaming state,
    # which is exactly equivalent (see `advance_session`) but keeps every vision forward
    # inside the size the offline path is known to run at.
    max_segment_blocks: int | None = None
    rollout_step_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.frame_per_lora < 1:
            raise ValueError("Interactive C2L requires frame_per_lora >= 1.")
        if not math.isfinite(self.sample_fps) or self.sample_fps <= 0:
            raise ValueError("Interactive C2L requires sample_fps > 0.")
        if not math.isfinite(self.tail_seconds) or self.tail_seconds < 0:
            raise ValueError("Interactive C2L requires tail_seconds >= 0.")
        if self.max_segment_blocks is not None and self.max_segment_blocks < 1:
            raise ValueError("Interactive C2L requires max_segment_blocks >= 1 when set.")

        if self.rollout_step_seconds is not None and (
            not math.isfinite(self.rollout_step_seconds) or self.rollout_step_seconds <= 0
        ):
            raise ValueError("Rollout step must be finite and positive.")

    @property
    def frame_period(self) -> float:
        return 1.0 / float(self.sample_fps)

    @property
    def block_period(self) -> float:
        return self.frame_per_lora * self.frame_period

    def blocks_before(self, span_seconds: float) -> int:
        """Whole temporal blocks fully contained in ``span_seconds`` of context."""
        if span_seconds <= 0:
            return 0
        blocks = int(math.floor(span_seconds / self.block_period + 1e-9))
        if self.max_encoder_frames is not None:
            blocks = min(blocks, self.max_encoder_frames // self.frame_per_lora)
        return max(0, blocks)


def interactive_config_from_args(model: Any, args: Any) -> InteractiveConfig:
    """Read the sampling policy off the loaded checkpoint and the evaluation arguments."""
    max_frames = getattr(args, "interactive_max_encoder_frames", None)
    frame_per_lora = int(getattr(model, "frame_per_lora", 2) or 2)
    # Limit each forward, never the total history: explicit override, checkpoint micro
    # budget, then 766 frames. Round down so the requested frame bound is not exceeded.
    segment_frames = getattr(args, "interactive_max_segment_frames", None)
    if not segment_frames:
        segment_frames = getattr(model, "micro_c2l_forward_frame", None)
    if not segment_frames:
        segment_frames = 766
    max_segment_blocks = None
    if segment_frames:
        max_segment_blocks = int(segment_frames) // frame_per_lora
    return InteractiveConfig(
        frame_per_lora=frame_per_lora,
        sample_fps=float(getattr(args, "interactive_sample_fps", 1.0) or 1.0),
        tail_seconds=(
            float(getattr(args, "base_tail_seconds", 2.0))
            if getattr(args, "base_with_last_2seconds", False)
            else 0.0
        ),
        interleave_timestamp=bool(getattr(model, "interleave_timestamp", False)),
        max_encoder_frames=int(max_frames) if max_frames else None,
        max_segment_blocks=max_segment_blocks,
        rollout_step_seconds=(getattr(args, "rollout_step_seconds", 2.0) if getattr(args, "eval_mode", "interactive") == "fixed-gap" else None),
    )


@dataclass
class InteractiveSession:
    """One video's streaming C2L state plus bookkeeping for the report."""

    state: Any
    config: InteractiveConfig
    origin: float = 0.0
    consumed_blocks: int = 0
    segments: int = 0
    turns: int = 0
    fallback_turns: int = 0
    last_segment_start: int | None = None
    verifications: list[dict[str, float]] = field(default_factory=list)
    skipped_verifications: int = 0
    next_tick: int = 1
    last_question_time: float | None = None

    def encoder_end(self, question_time: float) -> float:
        """Last second of context the encoder may see for a question at ``question_time``."""
        return max(self.origin, float(question_time) - self.config.tail_seconds)

    def target_blocks(self, question_time: float) -> int:
        return self.config.blocks_before(self.encoder_end(question_time) - self.origin)

    def has_context(self) -> bool:
        return self.consumed_blocks > 0


def new_session(model: Any, config: InteractiveConfig, origin: float = 0.0) -> InteractiveSession:
    return InteractiveSession(state=model.new_interactive_state(), config=config, origin=float(origin))


def block_frame_times(config: InteractiveConfig, origin: float, first_block: int, last_block: int) -> list[float]:
    """Absolute grid timestamps for blocks ``[first_block, last_block)``."""
    return [
        origin + index * config.frame_period
        for index in range(first_block * config.frame_per_lora, last_block * config.frame_per_lora)
    ]


_REPORTED_OVERRUNS: set[str] = set()

# OVO derives a question's time from the chunk's own duration using its own probe, which can
# disagree with ours in the last decimal. Clamp either way -- it costs at most one block --
# but only say so when the overrun is large enough to mean a broken annotation.
OVERRUN_REPORT_SECONDS = 1.0


def clamp_question_time(video_path: str, question_time: float) -> float:
    """Cap a question's timestamp at the end of the video it is asked about.

    Offline evaluation clamps its clip window to the reader (StreamingBench's
    ``adjust_video_time_range_for_reader``), so a malformed annotation timestamp is
    harmless there. Interactive mode builds a *grid* out to ``question_time`` instead, and
    the decoders pad out-of-range indices with copies of the last frame, so an annotation
    like ``"00:12450"`` on a 20-minute video silently becomes a 12448-frame segment of
    duplicated stills. Clamp here, once, for every benchmark.

    Benchmarks whose video is already the pre-truncated chunk for this question (OVO) take
    ``question_time == duration`` and are unaffected.
    """
    duration = video_duration_seconds(video_path)
    if duration is None or float(question_time) <= duration:
        return float(question_time)
    if question_time > duration + OVERRUN_REPORT_SECONDS and video_path not in _REPORTED_OVERRUNS:
        _REPORTED_OVERRUNS.add(video_path)
        print(
            f"WARNING: question timestamp {float(question_time):.1f}s is past the end of "
            f"{video_path} ({duration:.1f}s); clamping. This usually means a malformed "
            "annotation timestamp.",
            flush=True,
        )
    return float(duration)


def build_segment_encoder_inputs(
    processor: Any,
    encoder_text: str | list[str],
    video_path: str,
    frame_times: list[float],
    config: InteractiveConfig,
    args: Any,
    annotate_temporal_patch_size: Callable[[Any, Any], Any],
    device: Any = None,
) -> Any:
    """Processor inputs for one context segment, timestamped on the absolute video clock."""
    if not frame_times:
        raise ValueError("Interactive C2L segment has no frames.")
    args._resolution_video_path = video_path
    # The decoders clamp out-of-range indices to the last frame, so a grid that runs past
    # the end of the video comes back the right *length* and passes the count check below
    # while being mostly duplicated stills. Refuse it loudly instead: with
    # `clamp_question_time` in front of every segment this can no longer be reached, so if
    # it fires, the grid itself is wrong.
    duration = video_duration_seconds(video_path)
    if duration is not None and frame_times[-1] > duration + config.frame_period:
        raise ValueError(
            f"Interactive C2L segment runs to {frame_times[-1]:.1f}s but {video_path} is only "
            f"{duration:.1f}s long; the decoder would pad {len(frame_times)} timestamps with "
            "copies of the final frame."
        )
    frames = load_video_frames_at_times(video_path, frame_times)
    if len(frames) != len(frame_times):
        raise RuntimeError(
            f"Decoded {len(frames)} frames for {len(frame_times)} requested timestamps in {video_path}."
        )

    encoder_inputs = processor_predecoded_video_inputs(
        processor,
        encoder_text,
        [frames],
        args,
        metadata_fps=config.sample_fps,
        timestamps=[frame_times],
    )
    annotate_temporal_patch_size(processor, encoder_inputs)
    if config.interleave_timestamp:
        # Absolute times, so a block's label is the same whether it arrived in segment 1 or 7.
        terminal_time = frame_times[-1] + config.frame_period
        add_c2l_timestamp_inputs(
            encoder_inputs,
            processor.tokenizer,
            config.frame_per_lora,
            [frame_times],
            [terminal_time],
        )
    if device is not None:
        encoder_inputs = encoder_inputs.to(device)
    return encoder_inputs


def _advance_to_time(
    model: Any,
    processor: Any,
    session: InteractiveSession,
    encoder_text: str | list[str],
    video_path: str,
    question_time: float,
    args: Any,
    annotate_temporal_patch_size: Callable[[Any, Any], Any],
    device: Any = None,
) -> int:
    """Feed the footage between the last question and this one. Returns new block count.

    A catch-up longer than ``config.max_segment_blocks`` is split into consecutive
    sub-segments folded into the same streaming state. That is exactly equivalent to one
    call -- the Gated DeltaNet carry and the skip connection's running ``(sum, count)`` are
    both associative, which is the property `verify_session_against_single_pass` asserts --
    but it bounds every vision-encoder forward, which is otherwise the one part of the
    interactive path with no size limit at all.
    """
    target = session.target_blocks(clamp_question_time(video_path, question_time))
    if target <= session.consumed_blocks:
        # Two questions inside the same block, or the tail window already covers the gap.
        return 0

    first_block = session.consumed_blocks
    step = session.config.max_segment_blocks or (target - first_block)
    while session.consumed_blocks < target:
        stop = min(session.consumed_blocks + step, target)
        frame_times = block_frame_times(
            session.config, session.origin, session.consumed_blocks, stop
        )
        encoder_inputs = build_segment_encoder_inputs(
            processor,
            encoder_text,
            video_path,
            frame_times,
            session.config,
            args,
            annotate_temporal_patch_size,
            device=device,
        )
        model.interactive_update(encoder_inputs, session.state)
        # The verification control re-encodes from here to ask "what if the carried state
        # were dropped?", so it has to be the start of the *last* forward, not of the
        # whole catch-up.
        session.last_segment_start = session.consumed_blocks
        session.consumed_blocks = stop
        session.segments += 1
    return target - first_block


def advance_session(model, processor, session, encoder_text, video_path, question_time,
                    args, annotate_temporal_patch_size, device=None):
    """Advance on anchored ticks, then catch up at the exact question time."""
    if not math.isfinite(question_time) or question_time < session.origin:
        raise ValueError("Question time must be finite and at or after the session origin.")
    question_time = max(session.origin, clamp_question_time(video_path, question_time))
    if session.last_question_time is not None and question_time < session.last_question_time - 1e-9:
        raise ValueError("An interactive session cannot move backwards in time.")
    prior = session.consumed_blocks
    step = session.config.rollout_step_seconds
    if step is not None:
        while session.origin + session.next_tick * step <= question_time + 1e-9:
            tick = min(question_time, session.origin + session.next_tick * step)
            _advance_to_time(model, processor, session, encoder_text, video_path, tick,
                             args, annotate_temporal_patch_size, device)
            session.next_tick += 1
    _advance_to_time(model, processor, session, encoder_text, video_path, question_time,
                     args, annotate_temporal_patch_size, device)
    session.last_question_time = question_time
    return session.consumed_blocks - prior


def session_generated_loras(model: Any, session: InteractiveSession) -> Any:
    lora_dict, _layernorms = model.interactive_generate_weights(session.state)
    return lora_dict


# A single-pass check re-encodes the whole prefix at once, which is exactly the cost
# interactive mode exists to avoid. Skip the check past this depth rather than risk an OOM
# in the middle of an evaluation.
MAX_VERIFY_BLOCKS = 128

# The streamed LoRA and the single-pass LoRA are the same computation, so on GPU they should
# differ only by bf16 kernel noise. Anything above this is a real disagreement.
VERIFY_TOLERANCE = 5e-2


def verify_session_against_single_pass(
    model: Any,
    processor: Any,
    session: InteractiveSession,
    encoder_text: str | list[str],
    video_path: str,
    args: Any,
    annotate_temporal_patch_size: Callable[[Any, Any], Any],
    device: Any = None,
    max_blocks: int = MAX_VERIFY_BLOCKS,
) -> dict[str, float] | None:
    """Compare the streamed LoRA against one pass over the identical block stream.

    This isolates the streaming claim from frame selection: both sides see exactly the same
    blocks, so any difference is the temporal module's state carry. The carry is algebraic,
    so in fp32 the two agree to round-off (~1e-10); under the bf16 CUDA kernels the residual
    is kernel noise, not a different computation.
    """
    import torch

    if not session.has_context():
        raise ValueError("Cannot verify an interactive session that has consumed no context.")
    max_blocks = min(max_blocks, session.config.max_segment_blocks or max_blocks)
    if session.consumed_blocks > max_blocks:
        session.skipped_verifications += 1
        return None

    def single_pass(first_block: int):
        frame_times = block_frame_times(
            session.config, session.origin, first_block, session.consumed_blocks
        )
        encoder_inputs = build_segment_encoder_inputs(
            processor,
            encoder_text,
            video_path,
            frame_times,
            session.config,
            args,
            annotate_temporal_patch_size,
            device=device,
        )
        return model.generate_weights(encoder_inputs)[0]

    def worst_difference(reference, candidate) -> tuple[float, float]:
        max_abs = max_rel = 0.0
        for module_name, module_loras in reference.items():
            for matrix_key in ("A", "B"):
                expected = module_loras[matrix_key].float()
                actual = candidate[module_name][matrix_key].float()
                if actual.shape != expected.shape or not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
                    raise RuntimeError("Interactive verification found nonfinite or incompatible LoRA tensors.")
                diff = (actual - expected).abs().amax()
                scale = expected.abs().amax().clamp(min=1e-12)
                max_abs = max(max_abs, float(diff))
                max_rel = max(max_rel, float(diff / scale))
        return max_abs, max_rel

    with torch.inference_mode():
        offline_loras = single_pass(0)
        streamed_loras = session_generated_loras(model, session)
        max_abs, max_rel = worst_difference(offline_loras, streamed_loras)

        # Control: what the answer would look like if the carried state were dropped, i.e.
        # if only the newest segment counted. Without this the comparison above cannot
        # distinguish "the state is carried correctly" from "history barely matters here",
        # and a kernel that ignored `initial_state` would pass unnoticed.
        dropped_rel = None
        if session.last_segment_start:
            _dropped_abs, dropped_rel = worst_difference(
                offline_loras, single_pass(session.last_segment_start)
            )

    result = {
        "blocks": float(session.consumed_blocks),
        "segments": float(session.segments),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "dropped_state_rel_diff": dropped_rel,
        "ok": max_rel <= VERIFY_TOLERANCE,
    }
    session.verifications.append(result)
    if getattr(args, "_verification_log", None):
        import json
        from pathlib import Path
        with Path(args._verification_log).open("a") as stream:
            stream.write(json.dumps({"video_path": video_path, **result}) + "\n")
    if not result["ok"]:
        raise RuntimeError(
            f"Interactive LoRA verification failed for {video_path}: relative error "
            f"{max_rel:.3e} exceeds {VERIFY_TOLERANCE}. Scores are invalid."
        )
    return result


def summarize_verifications(sessions: list[InteractiveSession]) -> dict[str, Any] | None:
    records = [record for session in sessions for record in session.verifications]
    skipped = sum(session.skipped_verifications for session in sessions)
    if not records:
        return {"checks": 0, "skipped_too_long": skipped}

    controls = [record["dropped_state_rel_diff"] for record in records if record["dropped_state_rel_diff"]]
    failures = [record for record in records if not record["ok"]]
    summary = {
        "checks": len(records),
        "failures": len(failures),
        "skipped_too_long": skipped,
        "max_blocks_checked": max(record["blocks"] for record in records),
        "max_abs_diff": max(record["max_abs_diff"] for record in records),
        "max_rel_diff": max(record["max_rel_diff"] for record in records),
        "mean_rel_diff": sum(record["max_rel_diff"] for record in records) / len(records),
        "tolerance": VERIFY_TOLERANCE,
    }
    if controls:
        # How wrong the answer would be if the carried state were dropped. This must be much
        # larger than `max_rel_diff`, otherwise the check proved nothing on this hardware.
        summary["dropped_state_rel_diff_min"] = min(controls)
        summary["separation"] = min(controls) / max(summary["max_rel_diff"], 1e-12)
    return summary


def verification_verdict(summary: dict[str, Any] | None) -> str | None:
    """A one-line human verdict, or None when there is nothing to report."""
    if not summary:
        return None
    if not summary.get("checks"):
        return "INCONCLUSIVE: no eligible context was verified (empty or exceeding the reference-pass frame bound)."
    if summary["failures"]:
        return (
            f"FAILED: {summary['failures']}/{summary['checks']} interactive sessions disagree with a "
            f"single pass over the same blocks (max rel {summary['max_rel_diff']:.2e} > "
            f"{summary['tolerance']:.0e}). The streaming state carry is not reproducing the offline "
            "computation on this hardware; treat these scores as invalid."
        )
    separation = summary.get("separation")
    if separation is None:
        # Every checked session was a single segment, so nothing was ever carried across a
        # gap. Agreement here says nothing about the state carry.
        return (
            f"INCONCLUSIVE: {summary['checks']} sessions matched to {summary['max_rel_diff']:.2e}, but "
            "none of them carried state across a segment boundary, so the carry was never exercised."
        )
    if separation < 10:
        return (
            f"INCONCLUSIVE: streamed vs single pass is {summary['max_rel_diff']:.2e}, but dropping the "
            f"carried state only changes the result by {summary['dropped_state_rel_diff_min']:.2e} "
            f"({separation:.1f}x). The check cannot tell a working state carry from a dropped one here."
        )
    return (
        f"OK: {summary['checks']} sessions match a single pass to {summary['max_rel_diff']:.2e} relative, "
        f"while dropping the carried state would change them by "
        f"{summary['dropped_state_rel_diff_min']:.2e} ({separation:.0f}x larger)."
    )


def enforce_verification(summary: dict[str, Any] | None, verdict: str | None) -> None:
    """Abort on numerical disagreement; invalid scores must not be published."""
    if not summary or not summary.get("failures"):
        return
    raise RuntimeError(verdict or "Interactive C2L verification failed.")


def describe_streaming_path(model: Any) -> str:
    """Which temporal-streaming path this build takes, for the run log.

    Purely informational, so it never raises: a log line must not be able to fail a run.
    """
    try:
        import torch  # noqa: PLC0415

        from malow.video.ctx_to_lora.modeling.temporal_aggregator import (  # noqa: PLC0415
            multitoken_continuation_supported,
        )

        aggregator = model.hypernet.temporal_aggregator
        parameter = next(aggregator.parameters(), None)
        device_type = parameter.device.type if parameter is not None else "cpu"
        dtype = parameter.dtype if parameter is not None else torch.float32
        supported = multitoken_continuation_supported(device_type, dtype)
    except Exception as error:  # noqa: BLE001
        return f"temporal streaming path could not be determined ({error})"
    if supported:
        return (
            f"temporal streaming on {device_type}/{dtype}: multi-token cache continuation is "
            "supported; each segment is one forward."
        )
    return (
        f"temporal streaming on {device_type}/{dtype}: this Transformers build ignores cached "
        "linear-attention state on multi-token forwards (<= 5.6.0 gates it on seq_len == 1), "
        "so warm segments are streamed one block at a time."
    )


def finish_session(session):
    """Release GPU tensors; retain only the small verification report."""
    session.state = None
    return session
