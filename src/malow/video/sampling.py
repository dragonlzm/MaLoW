# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
"""Shared video-frame sampling helpers for Video C2L evaluation."""

from __future__ import annotations

import math
import os
from typing import Any


def c2l_encoder_frame_count(args: Any, default: int = 32) -> int:
    num_frames = getattr(args, "num_frames", None)
    if num_frames is not None:
        return max(1, int(num_frames))
    max_frames = getattr(args, "max_frames", None)
    if max_frames is not None:
        return max(1, int(max_frames))
    return max(1, int(default))


def _uniform_indices(start_index: int, end_index: int, sample_count: int) -> list[int]:
    if sample_count <= 1:
        return [start_index]
    return [
        start_index + int(index * (end_index - start_index) / (sample_count - 1))
        for index in range(sample_count)
    ]


def _load_video_clip_frames_decord(
    video_path: str,
    num_frames: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> list[Any]:
    return _load_video_clip_frames_decord_with_timestamps(
        video_path, num_frames, clip_start_seconds, clip_end_seconds
    )[0]


def _load_video_clip_frames_decord_with_timestamps(
    video_path: str,
    num_frames: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> tuple[list[Any], list[float], float]:
    try:
        from decord import VideoReader, cpu
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Decord clip loading requires `decord` and `Pillow`.") from exc

    reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(reader)
    if total_frames < 1:
        raise ValueError(f"No frames found in {video_path}.")

    fps = float(reader.get_avg_fps() or 1.0)
    start_frame = min(max(0, int(float(clip_start_seconds) * fps)), total_frames - 1)
    if clip_end_seconds is None:
        end_frame = total_frames - 1
        terminal_time = total_frames / fps
    else:
        end_frame = min(max(start_frame, int(float(clip_end_seconds) * fps)), total_frames - 1)
        terminal_time = max(float(clip_start_seconds), float(clip_end_seconds))

    sample_count = min(num_frames, end_frame - start_frame + 1)
    sample_indices = _uniform_indices(start_frame, end_frame, sample_count)
    frame_batch = reader.get_batch(sample_indices).asnumpy()
    frames = [Image.fromarray(frame) for frame in frame_batch]
    if not frames:
        raise ValueError(
            f"No frames sampled from {video_path} between {clip_start_seconds} and {clip_end_seconds} seconds."
        )
    return frames, [float(index) / fps for index in sample_indices], terminal_time


def _load_video_clip_frames_pyav(
    video_path: str,
    num_frames: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> list[Any]:
    return _load_video_clip_frames_pyav_with_timestamps(
        video_path, num_frames, clip_start_seconds, clip_end_seconds
    )[0]


def _load_video_clip_frames_pyav_with_timestamps(
    video_path: str,
    num_frames: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> tuple[list[Any], list[float], float]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("Video C2L clip loading requires PyAV (`av`).") from exc

    container = av.open(video_path, "r")
    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"No video stream found in {video_path}.")

        fps = float(stream.average_rate or 1.0)
        start_frame = max(0, int(float(clip_start_seconds) * fps))
        if clip_end_seconds is None:
            end_frame = int(stream.frames) - 1 if getattr(stream, "frames", 0) else start_frame
            terminal_time = (int(stream.frames) / fps) if getattr(stream, "frames", 0) else float(clip_start_seconds)
        else:
            end_frame = max(start_frame, int(float(clip_end_seconds) * fps))
            terminal_time = max(float(clip_start_seconds), float(clip_end_seconds))
        if getattr(stream, "frames", 0):
            start_frame = min(start_frame, int(stream.frames) - 1)
            end_frame = min(end_frame, int(stream.frames) - 1)

        sample_count = min(num_frames, end_frame - start_frame + 1)
        sample_indices = set(_uniform_indices(start_frame, end_frame, sample_count))
        frames = []
        decoded_indices = []
        container.seek(0)
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index > end_frame:
                break
            if frame_index in sample_indices:
                frames.append(frame.to_image())
                decoded_indices.append(frame_index)
        if not frames:
            raise ValueError(
                f"No frames sampled from {video_path} between {clip_start_seconds} and {clip_end_seconds} seconds."
            )
        return frames, [float(index) / fps for index in decoded_indices], terminal_time
    finally:
        container.close()


def load_video_clip_frames(
    video_path: str,
    num_frames: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> list[Any]:
    if num_frames < 1:
        raise ValueError("Video C2L clip loading requires num_frames >= 1.")
    try:
        return _load_video_clip_frames_decord(video_path, num_frames, clip_start_seconds, clip_end_seconds)
    except Exception as decord_error:
        try:
            return _load_video_clip_frames_pyav(video_path, num_frames, clip_start_seconds, clip_end_seconds)
        except Exception as pyav_error:
            raise RuntimeError(
                "Failed to load Video C2L clip with Decord and PyAV: "
                f"{video_path}. Decord error: {decord_error!r}. PyAV error: {pyav_error!r}."
            ) from pyav_error


def load_video_clip_frames_with_timestamps(
    video_path: str,
    num_frames: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> tuple[list[Any], list[float], float]:
    if num_frames < 1:
        raise ValueError("Video C2L clip loading requires num_frames >= 1.")
    try:
        return _load_video_clip_frames_decord_with_timestamps(
            video_path, num_frames, clip_start_seconds, clip_end_seconds
        )
    except Exception as decord_error:
        try:
            return _load_video_clip_frames_pyav_with_timestamps(
                video_path, num_frames, clip_start_seconds, clip_end_seconds
            )
        except Exception as pyav_error:
            raise RuntimeError(
                "Failed to load Video C2L clip with Decord and PyAV: "
                f"{video_path}. Decord error: {decord_error!r}. PyAV error: {pyav_error!r}."
            ) from pyav_error


def _video_duration_seconds(video_path: str) -> float | None:
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(video_path, ctx=cpu(0))
        fps = float(reader.get_avg_fps() or 0.0)
        if fps > 0:
            return len(reader) / fps
    except Exception:
        pass

    try:
        import av

        container = av.open(video_path, "r")
        try:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                return None
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            fps = float(stream.average_rate or 0.0)
            if fps > 0 and getattr(stream, "frames", 0):
                return float(stream.frames) / fps
        finally:
            container.close()
    except Exception:
        return None

    return None


def _resolve_clip_bounds(
    video_path: str,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> tuple[float, float]:
    start = max(0.0, float(clip_start_seconds))
    if clip_end_seconds is not None:
        end = max(start, float(clip_end_seconds))
    else:
        duration = _video_duration_seconds(video_path)
        end = max(start, duration if duration is not None else start)
    return start, end


def _frame_count_at_fps(duration_seconds: float, fps: float, max_frames: int) -> int:
    if max_frames < 1:
        raise ValueError("Video C2L FPS sampling requires max_frames >= 1.")
    if duration_seconds <= 0:
        return 1
    return max(1, min(int(max_frames), int(math.ceil(duration_seconds * fps))))


def split_encoder_base_frames(
    frames: list[Any],
    encoder_frame_count: int,
    base_frame_count: int,
) -> tuple[list[Any], list[Any]]:
    if base_frame_count < 1:
        return frames, []
    if not frames:
        raise ValueError("C2L base-input video split received no decoded frames.")
    if len(frames) > encoder_frame_count:
        return frames[:encoder_frame_count], frames[encoder_frame_count : encoder_frame_count + base_frame_count]

    base_start = max(0, len(frames) - base_frame_count)
    base_frames = frames[base_start:] or frames
    encoder_frames = frames[:base_start] or frames
    return encoder_frames, base_frames


def _ensure_even_frames(frames: list[Any]) -> list[Any]:
    if len(frames) % 2 == 0 or not frames:
        return frames
    return frames + [frames[-1]]


def load_split_c2l_video_frames(
    video_path: str,
    encoder_frame_count: int,
    base_frame_count: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> tuple[list[Any], list[Any]]:
    encoder_frames, base_frames, _encoder_times, _base_times, _terminal_time = (
        load_split_c2l_video_frames_with_timestamps(
            video_path,
            encoder_frame_count,
            base_frame_count,
            clip_start_seconds,
            clip_end_seconds,
        )
    )
    return encoder_frames, base_frames


def load_split_c2l_video_frames_with_timestamps(
    video_path: str,
    encoder_frame_count: int,
    base_frame_count: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
) -> tuple[list[Any], list[Any], list[float], list[float], float]:
    frames, timestamps, terminal_time = load_video_clip_frames_with_timestamps(
        video_path,
        encoder_frame_count + max(0, int(base_frame_count)),
        clip_start_seconds=clip_start_seconds,
        clip_end_seconds=clip_end_seconds,
    )
    encoder_frames, base_frames = split_encoder_base_frames(frames, encoder_frame_count, base_frame_count)
    encoder_times, base_times = split_encoder_base_frames(timestamps, encoder_frame_count, base_frame_count)
    return _ensure_even_frames(encoder_frames), _ensure_even_frames(base_frames), encoder_times, base_times, terminal_time


def load_base_last_2seconds_c2l_video_frames(
    video_path: str,
    encoder_frame_count: int,
    base_frame_count: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
    tail_seconds: float = 2.0,
    sample_fps: float = 1.0,
) -> tuple[list[Any], list[Any]]:
    encoder_frames, base_frames, _encoder_times, _base_times, _terminal_time = (
        load_base_last_2seconds_c2l_video_frames_with_timestamps(
            video_path,
            encoder_frame_count,
            base_frame_count,
            clip_start_seconds,
            clip_end_seconds,
            tail_seconds,
            sample_fps,
        )
    )
    return encoder_frames, base_frames


def load_base_last_2seconds_c2l_video_frames_with_timestamps(
    video_path: str,
    encoder_frame_count: int,
    base_frame_count: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
    tail_seconds: float = 2.0,
    sample_fps: float = 1.0,
) -> tuple[list[Any], list[Any], list[float], list[float], float]:
    if not math.isfinite(tail_seconds) or tail_seconds <= 0:
        raise ValueError("Video C2L base tail seconds must be finite and > 0.")
    if base_frame_count < 1:
        raise ValueError("Video C2L base-last-2seconds sampling requires base_frame_count >= 1.")
    if sample_fps <= 0:
        raise ValueError("Video C2L base-last-2seconds sampling requires sample_fps > 0.")

    start, end = _resolve_clip_bounds(video_path, clip_start_seconds, clip_end_seconds)
    duration = max(0.0, end - start)
    if duration <= tail_seconds:
        shared_frames, shared_times, _terminal_time = load_video_clip_frames_with_timestamps(
            video_path,
            base_frame_count,
            clip_start_seconds=start,
            clip_end_seconds=end,
        )
        shared_frames = _ensure_even_frames(shared_frames)
        return shared_frames, shared_frames, shared_times, shared_times, end

    tail_start = max(start, end - tail_seconds)
    prefix_duration = max(0.0, tail_start - start)
    encoder_sample_count = _frame_count_at_fps(prefix_duration, sample_fps, encoder_frame_count)
    encoder_frames, encoder_times, encoder_terminal_time = load_video_clip_frames_with_timestamps(
        video_path,
        encoder_sample_count,
        clip_start_seconds=start,
        clip_end_seconds=tail_start,
    )
    base_frames, base_times, _base_terminal_time = load_video_clip_frames_with_timestamps(
        video_path,
        base_frame_count,
        clip_start_seconds=tail_start,
        clip_end_seconds=end,
    )
    return _ensure_even_frames(encoder_frames), _ensure_even_frames(base_frames), encoder_times, base_times, encoder_terminal_time


def _video_metadata(videos: list[list[Any]], fps: float) -> list[dict[str, Any]]:
    metadata = []
    for video in videos:
        frame_count = len(video)
        metadata.append(
            {
                "fps": fps,
                "duration": frame_count / max(fps, 1e-6),
                "total_num_frames": frame_count,
                "frames_indices": list(range(frame_count)),
            }
        )
    return metadata


def processor_predecoded_video_inputs(
    processor,
    texts: str | list[str],
    videos: list[list[Any]],
    args: Any,
    *,
    metadata_fps: float | None = None,
    timestamps: list[list[float]] | None = None,
    branch: str = "encoder",
):
    fps = float(metadata_fps if metadata_fps is not None else (getattr(args, "video_fps", None) or 2.0))
    original_videos = videos
    cap = getattr(args, "base_video_max_pixels" if branch == "base" else "encoder_video_max_pixels", 0) or 0
    videos = resize_c2l_video_frames(videos, cap, getattr(args, "video_min_pixels", 1024))
    metadata = _video_metadata(videos, fps)
    base_kwargs: dict[str, Any] = {
        "text": texts,
        "images": None,
        "videos": videos,
        "padding": True,
        "padding_side": "left",
        "return_tensors": "pt",
        "video_metadata": metadata,
        "fps": fps,
        "return_metadata": True,
        "do_sample_frames": False,
    }
    longest = getattr(args, "encoder_video_longest_edge", 0) if branch == "encoder" else 0
    video_processor = getattr(processor, "video_processor", None)
    current_size = getattr(video_processor, "size", None)
    if longest:
        shortest = (current_size.get("shortest_edge") if isinstance(current_size, dict)
                    else getattr(current_size, "shortest_edge", None)) or 4096
        base_kwargs["size"] = {"shortest_edge": int(shortest), "longest_edge": int(longest)}
    inputs = processor(**base_kwargs)
    if getattr(args, "_resolution_log", None):
        record_video_resolution(args, processor, original_videos, videos, inputs, branch,
                                cap, base_kwargs.get("size", current_size), timestamps)
    # `video_metadata` controls processor-side sampling but is not a Qwen model
    # input. Leaving the returned copy in the batch makes Transformers 5.6
    # reject it as an unused kwarg during generation.
    inputs.pop("video_metadata", None)
    return inputs


def _local_frame_period(frame_times: list[float]) -> float:
    for previous, current in zip(reversed(frame_times[:-1]), reversed(frame_times[1:]), strict=True):
        period = current - previous
        if period > 0:
            return period
    return 1.0


def _timestamp_labels_from_frame_times(
    frame_times: list[float], frame_per_lora: int, terminal_time: float | None
) -> list[str]:
    if frame_per_lora < 1:
        raise ValueError("C2L timestamp interleaving requires frame_per_lora >= 1.")
    if not frame_times:
        raise ValueError("C2L timestamp interleaving requires at least one decoded frame time.")

    labels = []
    for start_index in range(0, len(frame_times), frame_per_lora):
        end_index = min(start_index + frame_per_lora, len(frame_times))
        start = frame_times[start_index]
        end = frame_times[end_index] if end_index < len(frame_times) else terminal_time
        if end is None:
            end = frame_times[-1] + _local_frame_period(frame_times)
        labels.append(f"{start:g}s-{end:g}s")
    return labels


def _processor_decoded_timestamps(encoder_inputs: dict[str, Any], sample_idx: int) -> list[float] | None:
    video_metadata = encoder_inputs.get("video_metadata")
    if not isinstance(video_metadata, list) or sample_idx >= len(video_metadata):
        return None
    metadata = video_metadata[sample_idx]
    if metadata is None:
        return None
    frame_indices = getattr(metadata, "frames_indices", None)
    fps = getattr(metadata, "fps", None)
    if isinstance(metadata, dict):
        frame_indices = metadata.get("frames_indices", frame_indices)
        fps = metadata.get("fps", fps)
    if not isinstance(frame_indices, (list, tuple)) or not frame_indices or not fps:
        return None
    fps = float(fps)
    if fps <= 0:
        return None
    return [float(frame_index) / fps for frame_index in frame_indices]


def _encoder_input_device(encoder_inputs: dict[str, Any]):
    for value in encoder_inputs.values():
        device = getattr(value, "device", None)
        if device is not None:
            return device
    return None


def add_c2l_timestamp_inputs(
    encoder_inputs: dict[str, Any],
    tokenizer,
    frame_per_lora: int,
    frame_timestamps: list[list[float] | None] | None = None,
    terminal_times: list[float | None] | None = None,
) -> None:
    """Attach timestamp text tensors expected by timestamp-interleaved C2L checkpoints."""
    if frame_timestamps is not None:
        sample_count = len(frame_timestamps)
    else:
        video_grid_thw = encoder_inputs.get("video_grid_thw")
        sample_count = int(video_grid_thw.shape[0]) if video_grid_thw is not None else 0
    if sample_count < 1:
        raise ValueError("C2L timestamp interleaving requires at least one encoder video context.")

    timestamp_labels = []
    timestamp_block_owners = []
    for sample_idx in range(sample_count):
        timestamps = frame_timestamps[sample_idx] if frame_timestamps is not None else None
        if timestamps is None:
            timestamps = _processor_decoded_timestamps(encoder_inputs, sample_idx)
        if timestamps is None:
            raise ValueError(
                f"C2L timestamp interleaving is missing decoded frame times for sample {sample_idx}."
            )
        if frame_per_lora == 1 and len(timestamps) % 2:
            timestamps = list(timestamps) + [timestamps[-1]]
        terminal_time = terminal_times[sample_idx] if terminal_times is not None else None
        labels = _timestamp_labels_from_frame_times(
            [float(timestamp) for timestamp in timestamps], frame_per_lora, terminal_time
        )
        timestamp_labels.extend(labels)
        timestamp_block_owners.extend([sample_idx] * len(labels))

    tokenized = tokenizer(timestamp_labels, add_special_tokens=False)
    timestamp_features = [
        {"input_ids": input_ids, "attention_mask": attention_mask}
        for input_ids, attention_mask in zip(
            tokenized["input_ids"],
            tokenized.get("attention_mask", [[1] * len(input_ids) for input_ids in tokenized["input_ids"]]),
            strict=True,
        )
    ]
    padded = tokenizer.pad(timestamp_features, padding=True, return_tensors="pt")
    import torch

    device = _encoder_input_device(encoder_inputs)
    encoder_inputs["timestamp_input_ids"] = padded["input_ids"].to(device) if device is not None else padded["input_ids"]
    encoder_inputs["timestamp_attention_mask"] = (
        padded["attention_mask"].to(device) if device is not None else padded["attention_mask"]
    )
    owners = torch.tensor(timestamp_block_owners, dtype=torch.long)
    encoder_inputs["timestamp_block_owners"] = owners.to(device) if device is not None else owners
    # Processor metadata is useful for deriving timestamps but is not accepted
    # by Qwen generation as a model input under Transformers 5.6.
    encoder_inputs.pop("video_metadata", None)


# Interactive inference adapted from Hippocampus master f4b15d9. See NOTICE.
from functools import lru_cache

@lru_cache(maxsize=1)
def _interactive_decord_reader(video_path: str):
    """Keep at most one indexed reader per worker, reusing it between rollout ticks."""
    from decord import VideoReader, cpu
    return VideoReader(video_path, ctx=cpu(0))


def _decode_frames_at_indices_decord(video_path: str, frame_indices: list[int]) -> list[Any]:
    from PIL import Image

    reader = _interactive_decord_reader(video_path)
    total_frames = len(reader)
    if total_frames < 1:
        raise ValueError(f"No frames found in {video_path}.")
    clamped = [min(max(0, index), total_frames - 1) for index in frame_indices]
    return [Image.fromarray(frame) for frame in reader.get_batch(clamped).asnumpy()]

def _decode_frames_at_indices_pyav(video_path: str, frame_indices: list[int]) -> list[Any]:
    """Seek to the segment's preceding keyframe, avoiding a decode of the entire prefix."""
    import av

    container = av.open(video_path, "r")
    try:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise ValueError(f"No video stream found in {video_path}.")
        fps = float(stream.average_rate or 1.0)
        start_pts = stream.start_time or 0
        wanted = sorted(set(frame_indices))
        seek_pts = start_pts + int(wanted[0] / fps / float(stream.time_base))
        container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
        decoded = {}
        cursor = 0
        last_image = None
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError("Interactive PyAV decoding requires frame timestamps.")
            position = round(float((frame.pts - start_pts) * stream.time_base) * fps)
            last_image = frame.to_image()
            while cursor < len(wanted) and wanted[cursor] <= position:
                decoded[wanted[cursor]] = last_image
                cursor += 1
            if cursor == len(wanted):
                break
        if last_image is None:
            raise ValueError(f"No frames decoded from {video_path} after seeking.")
        for index in wanted[cursor:]:
            decoded[index] = last_image
        return [decoded[index] for index in frame_indices]
    finally:
        container.close()


def load_video_frames_at_times(video_path: str, frame_times: list[float]) -> list[Any]:
    """Decode the frames nearest to a fixed list of absolute timestamps, in order.

    Interactive Video C2L samples the encoder on a *fixed* grid rather than uniformly over
    the current prefix, so that the frames chosen for ``[0, t_k]`` are always the frames
    chosen for ``[0, t_{k-1}]`` plus the new ones. Uniform prefix sampling does not have
    that property and would make the streamed block sequence disagree with the offline one.
    """
    if not frame_times:
        return []

    reader_choice = str(os.environ.get("FORCE_QWENVL_VIDEO_READER", "")).strip().lower()
    errors: list[str] = []
    for backend in ("decord", "pyav") if reader_choice != "pyav" else ("pyav", "decord"):
        try:
            if backend == "decord":
                reader = _interactive_decord_reader(video_path)
                fps = float(reader.get_avg_fps() or 1.0)
                del reader
                indices = [int(round(float(time) * fps)) for time in frame_times]
                return _decode_frames_at_indices_decord(video_path, indices)

            import av  # noqa: PLC0415

            container = av.open(video_path, "r")
            try:
                stream = next((item for item in container.streams if item.type == "video"), None)
                if stream is None:
                    raise ValueError(f"No video stream found in {video_path}.")
                fps = float(stream.average_rate or 1.0)
            finally:
                container.close()
            indices = [int(round(float(time) * fps)) for time in frame_times]
            return _decode_frames_at_indices_pyav(video_path, indices)
        except ImportError as exc:
            errors.append(f"{backend}: {exc}")
        except Exception as exc:  # noqa: BLE001 - fall through to the other backend
            errors.append(f"{backend}: {exc}")
    raise RuntimeError(f"Could not decode {video_path} at fixed timestamps ({'; '.join(errors)}).")

@lru_cache(maxsize=8192)
def video_duration_seconds(video_path: str) -> float | None:
    """Wall-clock length of ``video_path`` in seconds, or None if it cannot be probed.

    Cached: interactive evaluation asks for the same video once per question, and the
    answer is a property of the file.
    """
    return _video_duration_seconds(video_path)

def load_c2l_base_tail_frames(
    video_path: str,
    base_frame_count: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
    tail_seconds: float = 2.0,
) -> tuple[list[Any], list[float]]:
    """Only the base-model tail window of ``load_base_last_2seconds_c2l_video_frames``.

    Interactive evaluation already streams the encoder context, so re-decoding the whole
    prefix just to slice off its last two seconds would make every turn cost O(prefix).
    This reproduces the offline loader's base frames in both of its cases: when the clip is
    longer than ``tail_seconds`` it is the same tail call, and when it is shorter
    ``tail_start`` collapses to ``start``, which is the offline degenerate branch.
    """
    if base_frame_count < 1:
        raise ValueError("Video C2L base tail sampling requires base_frame_count >= 1.")

    start, end = _resolve_clip_bounds(video_path, clip_start_seconds, clip_end_seconds)
    tail_start = max(start, end - max(0.0, float(tail_seconds)))
    base_frames, base_times, _terminal = load_video_clip_frames_with_timestamps(
        video_path,
        base_frame_count,
        clip_start_seconds=tail_start,
        clip_end_seconds=end,
    )
    return _ensure_even_frames_with_timestamps(base_frames, base_times)

def _ensure_even_frames_with_timestamps(
    frames: list[Any], timestamps: list[float], multiple: int = 2
) -> tuple[list[Any], list[float]]:
    """Pad frames and their absolute source timestamps together.

    Frames and timestamps must stay aligned on short clips.
    """
    if multiple != 2:
        raise ValueError("Only two-frame temporal patches are supported.")
    padded_frames = _ensure_even_frames(frames)
    pad = len(padded_frames) - len(frames)
    if pad <= 0:
        return padded_frames, timestamps
    if not timestamps:
        raise ValueError("Cannot timestamp padded C2L video frames without a source frame timestamp.")
    return padded_frames, timestamps + [timestamps[-1]] * pad


C2L_QWEN_MIN_FRAME_SIDE = 28

def _preprocess_c2l_frame(frame, max_pixels: int, min_pixels: int, min_side: int):
    """One frame, resized exactly the way training resizes it.

    Mirrors `MMPluginMixin._preprocess_image` plus `Qwen2VLPlugin`'s 28-pixel side
    floor, including PIL's default BICUBIC resampling and the `int()` truncation of
    the scaled edges, so an evaluation frame and a training frame of the same source
    size come out byte-identical.
    """
    from PIL import Image

    if not isinstance(frame, Image.Image):
        # Already a tensor/array: no source geometry to rescale from.
        return frame

    width, height = frame.size
    if width * height > max_pixels:
        resize_factor = math.sqrt(max_pixels / (width * height))
        frame = frame.resize((int(width * resize_factor), int(height * resize_factor)))

    width, height = frame.size
    if width * height < min_pixels:
        resize_factor = math.sqrt(min_pixels / (width * height))
        frame = frame.resize((int(width * resize_factor), int(height * resize_factor)))

    if frame.mode != "RGB":
        frame = frame.convert("RGB")

    if min_side > 0 and min(frame.size) < min_side:
        frame = frame.resize((max(frame.size[0], min_side), max(frame.size[1], min_side)))

    return frame

def resize_c2l_video_frames(
    videos: list[list[Any]],
    max_pixels: int | None,
    min_pixels: int | None,
    *,
    min_side: int = C2L_QWEN_MIN_FRAME_SIDE,
) -> list[list[Any]]:
    """Apply the training-time per-frame pixel budget to pre-decoded frames.

    Training routes C2L encoder and base frames through the mm_plugin, which caps every
    frame at `video_max_pixels` and floors it at `video_min_pixels` (`_regularize_videos`
    resizes pre-decoded PIL frames too, via its nested-images branch). Evaluation decodes
    its own frames and calls the processor directly, so without this the checkpoint is
    fed frames at source resolution -- a 766-frame OVO context arrived at ~5x the area
    per frame that training ever used, which is both a train/eval mismatch and the reason
    the encoder context blew past a memory budget training fit inside.

    A missing or non-positive budget leaves the frames untouched.
    """
    if not max_pixels or max_pixels < 1:
        return videos

    min_pixels = int(min_pixels or 0)
    max_pixels = int(max_pixels)
    if min_pixels > max_pixels:
        raise ValueError(
            f"C2L frame budget needs video_min_pixels <= video_max_pixels, got {min_pixels} > {max_pixels}."
        )

    # Identity is preserved when nothing needs resizing, so callers that assert the
    # frames pass through untouched keep holding the same objects.
    changed = False
    resized_videos = []
    for video in videos:
        resized = [_preprocess_c2l_frame(frame, max_pixels, min_pixels, min_side) for frame in video]
        if any(new is not old for new, old in zip(resized, video, strict=True)):
            changed = True
            resized_videos.append(resized)
        else:
            resized_videos.append(video)

    return resized_videos if changed else videos


def record_video_resolution(args, processor, source, resized, inputs, branch, cap, size, timestamps):
    """Log actual processor geometry, without retaining decoded frames or tensors."""
    import json
    from pathlib import Path

    vp = getattr(processor, "video_processor", None)
    patch = getattr(vp, "patch_size", None)
    grid = inputs.get("video_grid_thw")
    grid = grid.tolist() if hasattr(grid, "tolist") else grid
    if size is not None and not isinstance(size, dict):
        size = {k: getattr(size, k, None) for k in ("shortest_edge", "longest_edge")}
    def dimensions(frame):
        from PIL import Image
        if isinstance(frame, Image.Image):
            return [frame.height, frame.width]
        shape = getattr(frame, "shape", ())
        return list(shape[:2]) if len(shape) == 3 and shape[-1] in (1, 3, 4) else list(shape[-2:])
    record = {
        "branch": branch, "eval_mode": args.eval_mode, "per_frame_pre_cap": cap,
        "video_path": getattr(args, "_resolution_video_path", None),
        "processor_size": size, "patch_size": patch,
        "temporal_patch_size": getattr(vp, "temporal_patch_size", None),
        "merge_size": getattr(vp, "merge_size", None), "video_grid_thw": grid,
        "frame_counts": [len(v) for v in source],
        "source_dimensions_hw": [dimensions(v[0]) for v in source],
        "preprocessed_dimensions_hw": [dimensions(v[0]) for v in resized],
        "processed_dimensions_hw": [[g[1] * patch, g[2] * patch] for g in grid] if grid is not None and patch else None,
        "timestamps": timestamps,
    }
    path = Path(args._resolution_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(record) + "\n")
