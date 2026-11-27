import copy
import glob
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import torch
from datasets import load_dataset

from ...data import SFTDataCollatorWith4DAttentionMask
from ...data.data_utils import merge_dataset
from ...data.parser import get_dataset_list
from ...extras import logging
from ...extras.constants import IGNORE_INDEX, IMAGE_PLACEHOLDER, VIDEO_PLACEHOLDER
from .vjepa2 import prepare_vjepa2_encoder_inputs


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...data import Template
    from ...hparams import DataArguments, FinetuningArguments, ModelArguments


def apply_answer_instruction(content: str, answer_instruction: str | None) -> str:
    content = content.replace(IMAGE_PLACEHOLDER, "").replace(VIDEO_PLACEHOLDER, "").strip()
    if not answer_instruction:
        return content

    return f"{content.rstrip()}\n\n{answer_instruction.strip()}"


@dataclass
class C2LProfileSource:
    name: str
    data_files: list[str]
    answer_instruction: str
    video_root: str | None = None
    source_video_root: str | None = None


@dataclass
class C2LVideoMergeCandidate:
    video: Any
    video_root: str | None = None
    source_video_root: str | None = None
    path_key: str | None = None


_C2L_ANSWER_INSTRUCTION_COLUMN = "__c2l_answer_instruction"
_C2L_VIDEO_ROOT_COLUMN = "__c2l_video_root"
_C2L_SOURCE_VIDEO_ROOT_COLUMN = "__c2l_source_video_root"
_C2L_SFT_TARGET_FIELD = "c2l_sft_target"
_C2L_SFT_TARGET_GENERATED = "generated_answer"
_C2L_SFT_TARGET_EXPECTED_OR_GENERATED = "expected_or_generated"
_C2L_SFT_TARGET_VALUES = {
    _C2L_SFT_TARGET_GENERATED,
    _C2L_SFT_TARGET_EXPECTED_OR_GENERATED,
}
_C2L_SPARSE_TEACHER_COLUMNS = ("generated_token_ids", "logprobs_indices", "logprobs_vals")
_C2L_SFT_PROFILE_COLUMNS = (
    "question",
    "turns",
    "images",
    "videos",
    "original_record",
    "expected_answer",
    "generated_answer",
    "reference_answer",
    "expected_answer_text",
    "has_reference_answer",
    _C2L_SFT_TARGET_FIELD,
)


class _RepeatIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Any) -> None:
        self.dataset = dataset

    def __iter__(self):
        while True:
            yielded = False
            for row in self.dataset:
                yielded = True
                yield row
            if not yielded:
                return

    def set_epoch(self, epoch: int) -> None:
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


def _resolve_media_paths(images: list[str] | None, media_dir: str | None) -> list[str]:
    resolved = []
    for image in images or []:
        if media_dir is None:
            resolved.append(image)
            continue

        if os.path.isabs(image):
            if os.path.isfile(image):
                resolved.append(image)
                continue

            # Profile rows may contain absolute paths from the profiling machine.
            # Remap stale paths to the current media_dir using the VQAv2/COCO suffix.
            parts = image.split(os.sep)
            suffix = None
            for anchor in ("coco", "train2014", "val2014", "test2015"):
                if anchor in parts:
                    suffix = os.path.join(*parts[parts.index(anchor) :])
                    break

            if suffix is not None:
                media_path = os.path.join(media_dir, suffix)
                if os.path.isfile(media_path):
                    resolved.append(media_path)
                    continue

            resolved.append(image)
            continue

        media_path = os.path.join(media_dir, image)
        resolved.append(media_path if os.path.isfile(media_path) else image)

    return resolved


def _is_path_within_root(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([os.path.abspath(path), os.path.abspath(root)]) == os.path.abspath(root)
    except ValueError:
        return False


def _dedupe_paths(paths: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for path in paths:
        normalized = os.path.normpath(path)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def _candidate_video_root_paths(path: str, video_root: str, source_video_root: str | None) -> list[str]:
    path = os.path.expanduser(path)
    video_root = os.path.expanduser(video_root)
    if source_video_root:
        source_video_root = os.path.expanduser(source_video_root)

    if not os.path.isabs(path):
        return _dedupe_paths([os.path.join(video_root, path), path])

    candidates = []
    if source_video_root and _is_path_within_root(path, source_video_root):
        candidates.append(os.path.join(video_root, os.path.relpath(path, source_video_root)))

    parts = path.split(os.sep)
    root_names = []
    if source_video_root:
        root_names.append(os.path.basename(os.path.normpath(source_video_root)))
    root_names.append(os.path.basename(os.path.normpath(video_root)))
    for root_name in root_names:
        if root_name and root_name in parts:
            suffix_parts = parts[parts.index(root_name) + 1 :]
            if suffix_parts:
                candidates.append(os.path.join(video_root, *suffix_parts))

    candidates.append(path)
    return _dedupe_paths(candidates)


def _resolve_media_path(
    path: str,
    media_dir: str | None,
    video_root: str | None = None,
    source_video_root: str | None = None,
) -> str:
    if video_root:
        candidate_paths = _candidate_video_root_paths(path, video_root, source_video_root)
        for candidate_path in candidate_paths:
            if os.path.exists(candidate_path):
                return candidate_path

        logger.warning_rank0_once(f"C2L video does not exist and will be skipped: {candidate_paths[0]}")
        return candidate_paths[0]

    if media_dir is None:
        if not os.path.exists(path):
            logger.warning_rank0_once(f"C2L video does not exist and will be skipped: {path}")
        return path
    if os.path.isabs(path):
        if not os.path.exists(path):
            logger.warning_rank0_once(f"C2L video does not exist and will be skipped: {path}")
        return path

    media_path = os.path.join(media_dir, path)
    if os.path.exists(media_path):
        return media_path

    if os.path.exists(path):
        logger.warning_rank0_once(f"C2L media path does not exist, using original path instead: {media_path}")
        return path

    logger.warning_rank0_once(f"C2L video does not exist and will be skipped: {media_path}")
    return media_path


def _uniform_indices(start_index: int, end_index: int, sample_count: int) -> list[int]:
    if sample_count <= 1:
        return [start_index]
    return [
        start_index + int(index * (end_index - start_index) / (sample_count - 1))
        for index in range(sample_count)
    ]


def _load_video_clip_frames(
    video_path: str,
    clip_end_seconds: float,
    num_frames: int,
    clip_start_seconds: float = 0.0,
) -> list[Any]:
    return _load_video_clip_frames_with_timestamps(
        video_path, clip_end_seconds, num_frames, clip_start_seconds
    )[0]


def _load_video_clip_frames_with_timestamps(
    video_path: str,
    clip_end_seconds: float,
    num_frames: int,
    clip_start_seconds: float = 0.0,
) -> tuple[list[Any], list[float], float]:
    if num_frames < 1:
        raise ValueError("TimeLens video C2L clip loading requires num_frames >= 1.")
    try:
        return _load_video_clip_frames_decord_with_timestamps(
            video_path, clip_end_seconds, num_frames, clip_start_seconds
        )
    except Exception as decord_error:
        try:
            return _load_video_clip_frames_pyav_with_timestamps(
                video_path, clip_end_seconds, num_frames, clip_start_seconds
            )
        except Exception as pyav_error:
            raise RuntimeError(
                "Failed to load TimeLens video clip with Decord and PyAV: "
                f"{video_path}. Decord error: {decord_error!r}. PyAV error: {pyav_error!r}."
            ) from pyav_error


def _video_duration_seconds(video_path: str) -> float | None:
    try:
        from decord import VideoReader, cpu

        video_reader = VideoReader(video_path, ctx=cpu(0))
        fps = float(video_reader.get_avg_fps() or 0.0)
        if fps > 0 and len(video_reader) > 0:
            return float(len(video_reader)) / fps
    except Exception:
        pass

    try:
        import av

        container = av.open(video_path, "r")
        try:
            video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
            if video_stream is None:
                return None
            if video_stream.duration is not None and video_stream.time_base is not None:
                return float(video_stream.duration * video_stream.time_base)
            fps = float(video_stream.average_rate or 0.0)
            if fps > 0 and getattr(video_stream, "frames", 0):
                return float(video_stream.frames) / fps
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
        return start, max(start, float(clip_end_seconds))

    duration = _video_duration_seconds(video_path)
    return start, max(start, duration if duration is not None else start)


def _load_video_clip_frames_decord(
    video_path: str,
    clip_end_seconds: float,
    num_frames: int,
    clip_start_seconds: float = 0.0,
) -> list[Any]:
    return _load_video_clip_frames_decord_with_timestamps(
        video_path, clip_end_seconds, num_frames, clip_start_seconds
    )[0]


def _load_video_clip_frames_decord_with_timestamps(
    video_path: str,
    clip_end_seconds: float,
    num_frames: int,
    clip_start_seconds: float = 0.0,
) -> tuple[list[Any], list[float], float]:
    try:
        from decord import VideoReader, cpu
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Decord clip loading requires `decord` and `Pillow`.") from exc

    video_reader = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(video_reader)
    if total_frames < 1:
        raise ValueError(f"No frames found in {video_path}.")

    fps = float(video_reader.get_avg_fps() or 1.0)
    start_frame = min(max(0, int(float(clip_start_seconds) * fps)), total_frames - 1)
    end_frame = min(max(start_frame, int(float(clip_end_seconds) * fps)), total_frames - 1)
    sample_count = min(num_frames, end_frame - start_frame + 1)
    sample_indices = _uniform_indices(start_frame, end_frame, sample_count)
    frame_batch = video_reader.get_batch(sample_indices).asnumpy()
    frames = [Image.fromarray(frame) for frame in frame_batch]
    if not frames:
        raise ValueError(
            f"No frames sampled from {video_path} between {clip_start_seconds} and {clip_end_seconds} seconds."
        )

    return frames, [float(sample_index) / fps for sample_index in sample_indices], float(clip_end_seconds)


def _load_video_clip_frames_pyav(
    video_path: str,
    clip_end_seconds: float,
    num_frames: int,
    clip_start_seconds: float = 0.0,
) -> list[Any]:
    return _load_video_clip_frames_pyav_with_timestamps(
        video_path, clip_end_seconds, num_frames, clip_start_seconds
    )[0]


def _load_video_clip_frames_pyav_with_timestamps(
    video_path: str,
    clip_end_seconds: float,
    num_frames: int,
    clip_start_seconds: float = 0.0,
) -> tuple[list[Any], list[float], float]:
    try:
        import av
    except ImportError as exc:
        raise RuntimeError("TimeLens video C2L clip loading requires PyAV (`av`).") from exc

    container = av.open(video_path, "r")
    try:
        video_stream = next((stream for stream in container.streams if stream.type == "video"), None)
        if video_stream is None:
            raise ValueError(f"No video stream found in {video_path}.")
        fps = float(video_stream.average_rate or 1.0)
        start_frame = max(0, int(float(clip_start_seconds) * fps))
        end_frame = max(start_frame, int(float(clip_end_seconds) * fps))
        if getattr(video_stream, "frames", 0):
            start_frame = min(start_frame, int(video_stream.frames) - 1)
            end_frame = min(end_frame, int(video_stream.frames) - 1)
        sample_count = min(num_frames, end_frame - start_frame + 1)
        sample_indices = _uniform_indices(start_frame, end_frame, sample_count)
        selected_indices = set(sample_indices)
        frames = []
        decoded_indices = []
        container.seek(0)
        for frame_index, frame in enumerate(container.decode(video_stream)):
            if frame_index > end_frame:
                break
            if frame_index in selected_indices:
                frames.append(frame.to_image())
                decoded_indices.append(frame_index)
        if not frames:
            raise ValueError(
                f"No frames sampled from {video_path} between {clip_start_seconds} and {clip_end_seconds} seconds."
            )
        return frames, [float(sample_index) / fps for sample_index in decoded_indices], float(clip_end_seconds)
    finally:
        container.close()


def _split_encoder_base_frames(
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


def _ensure_even_frames_with_timestamps(
    frames: list[Any], timestamps: list[float]
) -> tuple[list[Any], list[float]]:
    if len(frames) % 2 == 0 or not frames:
        return frames, timestamps

    # The padding frame duplicates the final decoded frame, so its real timestamp is
    # also the final decoded timestamp. Keeping both lists aligned is required when
    # independently padded video segments are concatenated by video_merge_extend.
    return frames + [frames[-1]], timestamps + [timestamps[-1]]


def _rebase_video_timestamps(
    timestamps: list[float],
    source_start: float,
    destination_start: float,
    terminal_time: float | None,
) -> tuple[list[float], float | None]:
    offset = float(destination_start) - float(source_start)
    rebased = [float(timestamp) + offset for timestamp in timestamps]
    rebased_terminal = None if terminal_time is None else float(terminal_time) + offset
    return rebased, rebased_terminal


def _frame_count_at_fps(duration_seconds: float, fps: float, max_frames: int) -> int:
    if max_frames < 1:
        raise ValueError("Video C2L FPS sampling requires max_frames >= 1.")
    if duration_seconds <= 0:
        return 1
    return max(1, min(int(max_frames), int(math.ceil(duration_seconds * fps))))


def _load_base_last_2seconds_c2l_video_frames(
    video_path: str,
    encoder_frame_count: int,
    base_frame_count: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
    tail_seconds: float = 2.0,
    sample_fps: float = 1.0,
) -> tuple[list[Any], list[Any]]:
    encoder_frames, base_frames, _encoder_timestamps, _base_timestamps, _terminal_time = (
        _load_base_last_2seconds_c2l_video_frames_with_timestamps(
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


def _load_base_last_2seconds_c2l_video_frames_with_timestamps(
    video_path: str,
    encoder_frame_count: int,
    base_frame_count: int,
    clip_start_seconds: float = 0.0,
    clip_end_seconds: float | None = None,
    tail_seconds: float = 2.0,
    sample_fps: float = 1.0,
) -> tuple[list[Any], list[Any], list[float], list[float], float]:
    if not math.isfinite(tail_seconds) or tail_seconds <= 0:
        raise ValueError("C2L base tail seconds must be finite and > 0.")
    if base_frame_count < 1:
        raise ValueError("C2L `base_with_last_2seconds` requires base_input_frame_number >= 1.")
    if sample_fps <= 0:
        raise ValueError("C2L `base_with_last_2seconds` requires sample FPS > 0.")

    start, end = _resolve_clip_bounds(video_path, clip_start_seconds, clip_end_seconds)
    duration = max(0.0, end - start)
    if duration <= tail_seconds:
        shared_frames, shared_timestamps, _ = _load_video_clip_frames_with_timestamps(
            video_path,
            end,
            base_frame_count,
            start,
        )
        shared_frames, shared_timestamps = _ensure_even_frames_with_timestamps(shared_frames, shared_timestamps)
        return shared_frames, shared_frames, shared_timestamps, shared_timestamps, end

    tail_start = max(start, end - tail_seconds)
    prefix_duration = max(0.0, tail_start - start)
    encoder_sample_count = _frame_count_at_fps(prefix_duration, sample_fps, encoder_frame_count)
    encoder_frames, encoder_timestamps, encoder_terminal_time = _load_video_clip_frames_with_timestamps(
        video_path,
        tail_start,
        encoder_sample_count,
        start,
    )
    base_frames, base_timestamps, _ = _load_video_clip_frames_with_timestamps(
        video_path,
        end,
        base_frame_count,
        tail_start,
    )
    encoder_frames, encoder_timestamps = _ensure_even_frames_with_timestamps(encoder_frames, encoder_timestamps)
    base_frames, base_timestamps = _ensure_even_frames_with_timestamps(base_frames, base_timestamps)
    return encoder_frames, base_frames, encoder_timestamps, base_timestamps, encoder_terminal_time


def _resolve_video_spec(
    video: dict[str, Any],
    media_dir: str | None,
    base_input_frame_number: int = 0,
    video_root: str | None = None,
    source_video_root: str | None = None,
    base_with_last_2seconds: bool = False,
    base_last_2seconds_max_frames: int = 128,
    base_tail_seconds: float = 2.0,
) -> tuple[Any, Any | None]:
    encoder_video, base_video, _encoder_timestamps, _encoder_end, _base_timestamps = (
        _resolve_video_spec_with_timestamps(
            video,
            media_dir,
            base_input_frame_number,
            video_root,
            source_video_root,
            base_with_last_2seconds,
            base_last_2seconds_max_frames,
            base_tail_seconds=base_tail_seconds,
        )
    )
    return encoder_video, base_video


def _resolve_video_spec_with_timestamps(
    video: dict[str, Any],
    media_dir: str | None,
    base_input_frame_number: int = 0,
    video_root: str | None = None,
    source_video_root: str | None = None,
    base_with_last_2seconds: bool = False,
    base_last_2seconds_max_frames: int = 128,
    base_tail_seconds: float = 2.0,
) -> tuple[Any, Any | None, list[float] | None, float | None, list[float] | None]:
    path = video.get("path") or video.get("video_path")
    if not isinstance(path, str):
        if base_input_frame_number > 0:
            raise ValueError("C2L `base_input_video` requires video specs with a string `path` or `video_path`.")
        return video, None, None, None, None

    resolved_path = _resolve_media_path(path, media_dir, video_root, source_video_root)
    if video.get("clip_end_seconds") is None and not base_with_last_2seconds:
        if base_input_frame_number > 0:
            raise ValueError("C2L `base_input_video` requires TimeLens video specs with `clip_end_seconds`.")
        return resolved_path, None, None, None, None

    num_frames = int(video.get("num_frames") or video.get("answer_video_num_frames") or 32)
    clip_start_seconds = float(video.get("clip_start_seconds") or 0.0)
    if base_with_last_2seconds:
        encoder_frames, base_frames, encoder_timestamps, base_timestamps, encoder_terminal_time = (
            _load_base_last_2seconds_c2l_video_frames_with_timestamps(
                resolved_path,
                base_last_2seconds_max_frames,
                base_input_frame_number,
                clip_start_seconds,
                float(video["clip_end_seconds"]) if video.get("clip_end_seconds") is not None else None,
                tail_seconds=base_tail_seconds,
            )
        )
        return encoder_frames, base_frames, encoder_timestamps, encoder_terminal_time, base_timestamps

    total_frames = num_frames + max(0, int(base_input_frame_number))
    frames, timestamps, terminal_time = _load_video_clip_frames_with_timestamps(
        resolved_path,
        float(video["clip_end_seconds"]),
        total_frames,
        clip_start_seconds,
    )
    encoder_frames, base_frames = _split_encoder_base_frames(frames, num_frames, base_input_frame_number)
    encoder_timestamps, base_timestamps = _split_encoder_base_frames(
        timestamps, num_frames, base_input_frame_number
    )
    return encoder_frames, base_frames, encoder_timestamps, terminal_time, base_timestamps


def _resolve_video_paths(
    videos: list[Any] | None,
    media_dir: str | None,
    base_input_frame_number: int = 0,
    video_root: str | None = None,
    source_video_root: str | None = None,
    base_with_last_2seconds: bool = False,
    base_last_2seconds_max_frames: int = 128,
    include_timestamps: bool = False,
    base_tail_seconds: float = 2.0,
) -> tuple[list[Any], list[Any]] | tuple[list[Any], list[Any], list[list[float] | None], list[float | None], list[list[float] | None]]:
    resolved = []
    base_resolved = []
    encoder_timestamps = []
    encoder_end_seconds = []
    base_timestamps = []
    for video in videos or []:
        if isinstance(video, dict):
            encoder_video, base_video, video_timestamps, video_end_seconds, video_base_timestamps = (
                _resolve_video_spec_with_timestamps(
                video,
                media_dir,
                base_input_frame_number,
                video_root,
                source_video_root,
                base_with_last_2seconds,
                base_last_2seconds_max_frames,
                base_tail_seconds=base_tail_seconds,
            )
            )
            resolved.append(encoder_video)
            encoder_timestamps.append(video_timestamps)
            encoder_end_seconds.append(video_end_seconds)
            if base_video is not None:
                base_resolved.append(base_video)
                base_timestamps.append(video_base_timestamps)
        elif isinstance(video, list):
            frames = (
                [
                    _resolve_media_path(frame, media_dir, video_root, source_video_root)
                    if isinstance(frame, str)
                    else frame
                    for frame in video
                ]
            )
            if base_input_frame_number > 0:
                encoder_frames, base_frames = _split_encoder_base_frames(
                    frames, max(0, len(frames) - base_input_frame_number), base_input_frame_number
                )
                resolved.append(encoder_frames)
                base_resolved.append(base_frames)
                encoder_timestamps.append(None)
                encoder_end_seconds.append(None)
                base_timestamps.append(None)
            else:
                resolved.append(frames)
                encoder_timestamps.append(None)
                encoder_end_seconds.append(None)
        elif isinstance(video, str):
            if base_input_frame_number > 0:
                raise ValueError(
                    "C2L `base_input_video` requires pre-decoded frames or `clip_end_seconds` video specs."
                )
            resolved.append(_resolve_media_path(video, media_dir, video_root, source_video_root))
            encoder_timestamps.append(None)
            encoder_end_seconds.append(None)
        else:
            if base_input_frame_number > 0:
                raise TypeError(
                    "C2L `base_input_video` requires video inputs to be dict specs, frame lists, or path strings."
                )
            resolved.append(video)
            encoder_timestamps.append(None)
            encoder_end_seconds.append(None)

    if include_timestamps:
        return resolved, base_resolved, encoder_timestamps, encoder_end_seconds, base_timestamps
    return resolved, base_resolved


def _frame_sequence(video: Any) -> list[Any]:
    return list(video) if isinstance(video, list) else [video]


def _normalize_video_frame_shapes(frames: list[Any]) -> list[Any]:
    try:
        from PIL import Image
    except ImportError:
        return frames

    target_size = None
    normalized = []
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC", 3)
    for frame in frames:
        if not isinstance(frame, Image.Image):
            normalized.append(frame)
            continue

        if target_size is None:
            target_size = frame.size

        if frame.mode != "RGB":
            frame = frame.convert("RGB")
        if frame.size != target_size:
            target_width, target_height = target_size
            scale = min(target_width / frame.size[0], target_height / frame.size[1])
            resized_size = (
                max(1, int(round(frame.size[0] * scale))),
                max(1, int(round(frame.size[1] * scale))),
            )
            resized = frame.resize(resized_size, resampling)
            canvas = Image.new("RGB", target_size)
            paste_xy = ((target_width - resized_size[0]) // 2, (target_height - resized_size[1]) // 2)
            canvas.paste(resized, paste_xy)
            frame = canvas

        normalized.append(frame)

    return normalized


def _resolve_merge_candidate_frames(
    candidate: C2LVideoMergeCandidate,
    media_dir: str | None,
    base_input_frame_number: int,
    base_last_2seconds_max_frames: int,
    base_tail_seconds: float = 2.0,
) -> tuple[list[Any], list[Any], list[float] | None, list[float] | None, float | None]:
    if isinstance(candidate.video, dict):
        encoder_video, base_video, encoder_timestamps, encoder_end_seconds, base_timestamps = (
            _resolve_video_spec_with_timestamps(
            copy.deepcopy(candidate.video),
            media_dir,
            base_input_frame_number,
            video_root=candidate.video_root,
            source_video_root=candidate.source_video_root,
            base_with_last_2seconds=True,
            base_last_2seconds_max_frames=base_last_2seconds_max_frames,
            base_tail_seconds=base_tail_seconds,
        )
        )
        if base_video is None:
            raise ValueError("C2L `video_merge_extend` requires extra video specs with base-video frames.")

        return (
            _frame_sequence(encoder_video),
            _frame_sequence(base_video),
            encoder_timestamps,
            base_timestamps,
            encoder_end_seconds,
        )

    if isinstance(candidate.video, list):
        frames = [
            _resolve_media_path(frame, media_dir, candidate.video_root, candidate.source_video_root)
            if isinstance(frame, str)
            else frame
            for frame in candidate.video
        ]
        encoder_frames, base_frames = _split_encoder_base_frames(
            frames,
            max(0, len(frames) - base_input_frame_number),
            base_input_frame_number,
        )
        return _frame_sequence(encoder_frames), _frame_sequence(base_frames), None, None, None

    raise TypeError("C2L `video_merge_extend` extra videos must be dict specs or frame lists.")


def _video_spec_clip_bounds(
    video: Any,
    media_dir: str | None,
    video_root: str | None,
    source_video_root: str | None,
) -> tuple[float, float] | None:
    if not isinstance(video, dict):
        return None

    path = video.get("path") or video.get("video_path")
    if not isinstance(path, str):
        return None

    resolved_path = _resolve_media_path(path, media_dir, video_root, source_video_root)
    return _resolve_clip_bounds(
        resolved_path,
        float(video.get("clip_start_seconds") or 0.0),
        float(video["clip_end_seconds"]) if video.get("clip_end_seconds") is not None else None,
    )


def _video_path_key(
    video: Any,
    media_dir: str | None,
    video_root: str | None,
    source_video_root: str | None,
) -> str | None:
    path = None
    if isinstance(video, dict):
        path = video.get("path") or video.get("video_path")
    elif isinstance(video, str):
        path = video

    if not isinstance(path, str):
        return None

    return os.path.normpath(os.path.abspath(_resolve_media_path(path, media_dir, video_root, source_video_root)))


def _stable_video_merge_digest(row: dict[str, Any], seed: int) -> bytes:
    key = {
        "videos": _get_row_videos(row),
        "question": row.get("question"),
        "turns": row.get("turns"),
    }
    payload = json.dumps(key, sort_keys=True, default=str, ensure_ascii=True)
    return hashlib.sha256(f"{seed}:{payload}".encode("utf-8")).digest()


def _should_apply_video_merge_extend(row: dict[str, Any], percentage: float, seed: int) -> bool:
    if percentage <= 0.0:
        return False
    if percentage >= 1.0:
        return True

    digest = _stable_video_merge_digest(row, seed)
    probability = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    return probability < percentage


def _select_video_merge_candidate(
    row: dict[str, Any],
    candidates: list[C2LVideoMergeCandidate],
    seed: int,
    media_dir: str | None,
    video_root: str | None,
    source_video_root: str | None,
) -> C2LVideoMergeCandidate | None:
    if not candidates:
        return None

    current_path_keys = {
        path_key
        for path_key in (
            _video_path_key(video, media_dir, video_root, source_video_root)
            for video in _get_row_videos(row, media_dir)
        )
        if path_key is not None
    }
    digest = _stable_video_merge_digest(row, seed)
    start_index = int.from_bytes(digest[8:16], byteorder="big") % len(candidates)
    for offset in range(len(candidates)):
        candidate = candidates[(start_index + offset) % len(candidates)]
        if candidate.path_key is None or candidate.path_key not in current_path_keys:
            return candidate

    return None


def _maybe_apply_video_merge_extend(
    row: dict[str, Any],
    encoder_videos: list[Any],
    base_videos: list[Any],
    media_dir: str | None,
    video_root: str | None,
    source_video_root: str | None,
    base_input_frame_number: int,
    base_last_2seconds_max_frames: int,
    video_merge_candidates: list[C2LVideoMergeCandidate] | None,
    video_merge_seed: int,
    video_merge_extend_perc: float,
    encoder_video_timestamps: list[list[float] | None] | None = None,
    encoder_video_end_seconds: list[float | None] | None = None,
    base_video_timestamps: list[list[float] | None] | None = None,
    base_tail_seconds: float = 2.0,
) -> tuple[list[Any], list[Any], list[list[float] | None] | None, list[float | None] | None]:
    if not video_merge_candidates or not encoder_videos or not base_videos:
        return encoder_videos, base_videos, encoder_video_timestamps, encoder_video_end_seconds
    if not _should_apply_video_merge_extend(row, video_merge_extend_perc, video_merge_seed):
        return encoder_videos, base_videos, encoder_video_timestamps, encoder_video_end_seconds

    candidate = _select_video_merge_candidate(
        row,
        video_merge_candidates,
        video_merge_seed,
        media_dir,
        video_root,
        source_video_root,
    )
    if candidate is None:
        return encoder_videos, base_videos, encoder_video_timestamps, encoder_video_end_seconds

    extra_encoder_frames, extra_base_frames, extra_encoder_timestamps, _extra_base_timestamps, extra_end_seconds = (
        _resolve_merge_candidate_frames(
            candidate,
            media_dir,
            base_input_frame_number,
            base_last_2seconds_max_frames,
            base_tail_seconds=base_tail_seconds,
        )
    )
    merged_encoder_frames: list[Any] = []
    merged_timestamps: list[float] | None = []
    for encoder_video in encoder_videos:
        merged_encoder_frames.extend(_frame_sequence(encoder_video))
    for timestamps in encoder_video_timestamps or []:
        if timestamps is None:
            merged_timestamps = None
            break
        merged_timestamps.extend(timestamps)
    for base_video in base_videos:
        merged_encoder_frames.extend(_frame_sequence(base_video))
    if merged_timestamps is not None:
        for timestamps in base_video_timestamps or []:
            if timestamps is None:
                merged_timestamps = None
                break
            merged_timestamps.extend(timestamps)
    merged_encoder_frames.extend(extra_encoder_frames)
    if merged_timestamps is not None:
        if extra_encoder_timestamps is None:
            merged_timestamps = None
        else:
            current_videos = _get_row_videos(row, media_dir)
            current_bounds = (
                _video_spec_clip_bounds(current_videos[0], media_dir, video_root, source_video_root)
                if len(current_videos) == 1
                else None
            )
            extra_bounds = _video_spec_clip_bounds(
                candidate.video,
                media_dir,
                candidate.video_root,
                candidate.source_video_root,
            )
            if current_bounds is None or extra_bounds is None:
                merged_timestamps = None
                extra_end_seconds = None
            else:
                current_start, current_end = current_bounds
                extra_start, _extra_end = extra_bounds
                merged_timestamps, _ = _rebase_video_timestamps(
                    merged_timestamps,
                    source_start=current_start,
                    destination_start=0.0,
                    terminal_time=None,
                )
                shifted_extra_timestamps, extra_end_seconds = _rebase_video_timestamps(
                    extra_encoder_timestamps,
                    source_start=extra_start,
                    destination_start=current_end - current_start,
                    terminal_time=extra_end_seconds,
                )
                merged_timestamps.extend(shifted_extra_timestamps)
    merged_encoder_frames = _normalize_video_frame_shapes(merged_encoder_frames)
    extra_base_frames = _normalize_video_frame_shapes(extra_base_frames)

    if merged_timestamps is None:
        merged_encoder_frames = _ensure_even_frames(merged_encoder_frames)
    else:
        merged_encoder_frames, merged_timestamps = _ensure_even_frames_with_timestamps(
            merged_encoder_frames, merged_timestamps
        )
    return (
        [merged_encoder_frames],
        [_ensure_even_frames(extra_base_frames)],
        [merged_timestamps] if merged_timestamps is not None else [None],
        [extra_end_seconds],
    )


def _resolved_video_exists(
    video: Any,
    media_dir: str | None,
    video_root: str | None,
    source_video_root: str | None,
) -> bool:
    if isinstance(video, dict):
        path = video.get("path") or video.get("video_path")
        if not isinstance(path, str):
            return True
        return os.path.exists(_resolve_media_path(path, media_dir, video_root, source_video_root))

    if isinstance(video, str):
        return os.path.exists(_resolve_media_path(video, media_dir, video_root, source_video_root))

    if isinstance(video, list):
        paths = [frame for frame in video if isinstance(frame, str)]
        return all(os.path.exists(_resolve_media_path(path, media_dir, video_root, source_video_root)) for path in paths)

    return True


def _row_has_existing_videos(
    row: dict[str, Any],
    media_dir: str | None,
    video_root: str | None,
    source_video_root: str | None,
) -> bool:
    videos = _get_row_videos(row, media_dir)
    if not videos:
        return True

    return all(_resolved_video_exists(video, media_dir, video_root, source_video_root) for video in videos)


_PROFILE_EXTENSIONS = (".json", ".jsonl")


def _has_glob_pattern(path: str) -> bool:
    return any(char in path for char in "*?[]")


def _is_profile_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith(_PROFILE_EXTENSIONS)


def _get_profile_files(local_path: str) -> list[str]:
    local_path = os.path.expanduser(local_path)

    if _has_glob_pattern(local_path):
        return sorted(path for path in glob.glob(local_path) if _is_profile_file(path))

    if _is_profile_file(local_path):
        return [local_path]

    if os.path.isdir(local_path):
        return [
            os.path.join(local_path, file_name)
            for file_name in sorted(os.listdir(local_path))
            if _is_profile_file(os.path.join(local_path, file_name))
        ]

    return []


def _looks_like_profile_path(dataset_name: str) -> bool:
    return (
        os.path.isabs(dataset_name)
        or _has_glob_pattern(dataset_name)
        or os.sep in dataset_name
        or (os.altsep is not None and os.altsep in dataset_name)
        or dataset_name.endswith(_PROFILE_EXTENSIONS)
    )


def _profile_path_error(local_path: str) -> ValueError:
    local_path = os.path.expanduser(local_path)
    if _has_glob_pattern(local_path):
        return ValueError(f"C2L profile glob matched no JSON/JSONL files: {local_path}.")

    if os.path.isdir(local_path):
        entries = sorted(os.listdir(local_path))
        entry_preview = ", ".join(entries[:8]) if entries else "none"
        return ValueError(
            f"C2L profile directory contains no JSON/JSONL files: {local_path}. "
            f"Found entries: {entry_preview}. Point `dataset` at the directory containing profile shards, "
            "or use a glob such as `path/to/profile/*.jsonl`."
        )

    if os.path.exists(local_path):
        return ValueError(f"C2L profile path is not a JSON/JSONL file or directory: {local_path}.")

    return ValueError(
        f"C2L profile path not found: {local_path}. "
        "Check that the offline profile archive was extracted and that `dataset` points to the shard directory."
    )


def _normalize_video_root(video_root: str | None) -> str | None:
    if video_root is None:
        return None

    video_root = str(video_root).strip()
    return video_root or None


def _infer_source_video_root(local_path: str) -> str | None:
    normalized = os.path.normpath(local_path)
    marker = f"{os.sep}c2l_video_profile{os.sep}"
    if marker in normalized:
        return normalized.split(marker, 1)[0]

    suffix = f"{os.sep}c2l_video_profile"
    if normalized.endswith(suffix):
        return normalized[: -len(suffix)]

    return None


def _get_video_roots(dataset_names: list[str], data_args: "DataArguments") -> list[str | None]:
    video_roots = getattr(data_args, "video_roots", None)

    if isinstance(video_roots, str):
        video_roots = [item.strip() for item in video_roots.split(",")]

    if video_roots is None:
        return [None] * len(dataset_names)

    if len(video_roots) != len(dataset_names):
        raise ValueError("The length of C2L dataset and video roots should be identical.")

    return [_normalize_video_root(video_root) for video_root in video_roots]


def _resolve_profile_sources(dataset_names: list[str], data_args: "DataArguments") -> list[C2LProfileSource]:
    profile_sources: list[C2LProfileSource] = []
    unresolved_names: list[str] = []
    video_roots = _get_video_roots(dataset_names, data_args)
    unresolved_video_roots: list[str | None] = []
    for dataset_name, video_root in zip(dataset_names, video_roots):
        local_name = os.path.expanduser(dataset_name)
        if not os.path.isabs(local_name):
            local_name = os.path.join(data_args.dataset_dir, local_name)

        data_files = _get_profile_files(local_name)
        if data_files:
            profile_sources.append(
                C2LProfileSource(
                    dataset_name,
                    data_files,
                    "",
                    video_root,
                    _infer_source_video_root(local_name),
                )
            )
        elif _looks_like_profile_path(dataset_name):
            raise _profile_path_error(local_name)
        else:
            unresolved_names.append(dataset_name)
            unresolved_video_roots.append(video_root)

    if unresolved_names:
        for dataset_attr, video_root in zip(
            get_dataset_list(unresolved_names, data_args.dataset_dir), unresolved_video_roots
        ):
            if dataset_attr.load_from != "file":
                raise ValueError(f"C2L only supports local JSON/JSONL files, got {dataset_attr.load_from}.")

            local_path = dataset_attr.dataset_name
            if not os.path.isabs(local_path):
                local_path = os.path.join(data_args.dataset_dir, local_path)

            data_files = _get_profile_files(local_path)
            if data_files:
                profile_sources.append(
                    C2LProfileSource(
                        str(dataset_attr),
                        data_files,
                        dataset_attr.c2l_answer_instruction or "",
                        video_root,
                        _infer_source_video_root(local_path),
                    )
                )
            else:
                raise ValueError(f"C2L profile file or directory not found: {local_path}.")

    if not profile_sources:
        raise ValueError("No C2L profile JSON/JSONL files were found.")

    return profile_sources


def _get_row_images(row: dict[str, Any], media_dir: str | None = None) -> list[str]:
    original_record = row.get("original_record") or {}
    original_images = original_record.get("images") or []
    if media_dir is not None and original_images:
        return original_images

    return row.get("images") or original_images


def _get_row_videos(row: dict[str, Any], media_dir: str | None = None) -> list[Any]:
    original_record = row.get("original_record") or {}
    original_videos = original_record.get("videos") or []
    if media_dir is not None and original_videos:
        return original_videos

    return row.get("videos") or original_videos


def _iter_profile_records(file_path: str) -> Iterator[dict[str, Any]]:
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row
            return

        data = json.load(f)

    if isinstance(data, list):
        for row in data:
            if isinstance(row, dict):
                yield row
        return
    if isinstance(data, dict):
        rows = data.get("data")
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict):
                    yield row
            return
        yield data


def _collect_video_merge_candidates(
    profile_sources: list[C2LProfileSource],
    data_args: "DataArguments",
) -> list[C2LVideoMergeCandidate]:
    candidates: list[C2LVideoMergeCandidate] = []
    for profile_source in profile_sources:
        for file_path in profile_source.data_files:
            for row in _iter_profile_records(file_path):
                videos = _get_row_videos(row, data_args.media_dir)
                if not videos:
                    continue

                video = videos[0]
                if not _resolved_video_exists(
                    video,
                    data_args.media_dir,
                    profile_source.video_root,
                    profile_source.source_video_root,
                ):
                    continue

                candidates.append(
                    C2LVideoMergeCandidate(
                        copy.deepcopy(video),
                        profile_source.video_root,
                        profile_source.source_video_root,
                        _video_path_key(
                            video,
                            data_args.media_dir,
                            profile_source.video_root,
                            profile_source.source_video_root,
                        ),
                    )
                )

    return candidates


def _require_list_field(record: dict[str, Any], field_name: str, owner: str) -> list[Any]:
    value = record.get(field_name)
    if not isinstance(value, list):
        raise ValueError(f"C2L profile {owner} must contain list `{field_name}`.")
    return value


def _normalize_distribution_turn(
    record: dict[str, Any],
    answer_instruction: str | None,
    owner: str,
) -> dict[str, Any]:
    generated_token_ids = [
        int(token_id) for token_id in _require_list_field(record, "generated_token_ids", owner)
    ]
    logprobs_indices = [
        [int(token_id) for token_id in token_indices]
        for token_indices in _require_list_field(record, "logprobs_indices", owner)
    ]
    logprobs_vals = [
        [float(logprob) for logprob in token_logprobs]
        for token_logprobs in _require_list_field(record, "logprobs_vals", owner)
    ]
    return {
        "question": apply_answer_instruction(str(record.get("question", "")), answer_instruction or ""),
        "generated_answer": str(record.get("generated_answer") or ""),
        "generated_token_ids": generated_token_ids,
        "logprobs_indices": logprobs_indices,
        "logprobs_vals": logprobs_vals,
    }


def _normalize_sft_target(value: Any, owner: str) -> str:
    if value is None:
        # Keep the projected Hugging Face schema identical across legacy and
        # CoT profile sources. This is also the pre-CoT answer-selection policy:
        # prefer a nonempty expected answer and otherwise use generated_answer.
        return _C2L_SFT_TARGET_EXPECTED_OR_GENERATED
    if not isinstance(value, str) or value not in _C2L_SFT_TARGET_VALUES:
        allowed = ", ".join(sorted(_C2L_SFT_TARGET_VALUES))
        raise ValueError(
            f"C2L `{_C2L_SFT_TARGET_FIELD}` in {owner} must be one of: {allowed}. Got {value!r}."
        )
    return value


def _select_sft_answer(
    record: dict[str, Any],
    loss_type: str,
    owner: str,
    inherited_sft_target: str | None = None,
) -> str:
    if loss_type not in ("sft_generated", "sft_expected_or_generated"):
        raise ValueError(f"Unsupported C2L SFT loss type: {loss_type}.")
    sft_target = _normalize_sft_target(
        record.get(_C2L_SFT_TARGET_FIELD, inherited_sft_target), owner
    )
    if loss_type == "sft_generated" or sft_target == _C2L_SFT_TARGET_GENERATED:
        answer = record.get("generated_answer")
    elif loss_type == "sft_expected_or_generated":
        expected = record.get("expected_answer")
        answer = expected if isinstance(expected, str) and expected.strip() else record.get("generated_answer")
    if isinstance(answer, str):
        # Teacher CoT/answers may use video tags as timestamp markup, but the
        # multimodal plugin interprets every opening tag as a media input.
        answer = answer.replace(VIDEO_PLACEHOLDER, "").replace("</video>", "")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"C2L {loss_type} requires a nonempty textual answer in {owner}.")
    return answer.strip()


def _normalize_sft_turn(
    record: dict[str, Any],
    answer_instruction: str | None,
    loss_type: str,
    owner: str,
    inherited_sft_target: str | None = None,
) -> dict[str, Any]:
    sft_target = _normalize_sft_target(
        record.get(_C2L_SFT_TARGET_FIELD, inherited_sft_target), owner
    )
    turn = {
        "question": apply_answer_instruction(str(record.get("question", "")), answer_instruction or ""),
        "expected_answer": record.get("expected_answer"),
        "generated_answer": record.get("generated_answer"),
        "reference_answer": record.get("reference_answer"),
        "expected_answer_text": record.get("expected_answer_text"),
        "has_reference_answer": record.get("has_reference_answer"),
    }
    if sft_target is not None:
        turn[_C2L_SFT_TARGET_FIELD] = sft_target
    turn["selected_answer"] = _select_sft_answer(turn, loss_type, owner)
    return turn


def _get_profile_turns(
    row: dict[str, Any], answer_instruction: str | None, loss_type: str
) -> list[dict[str, Any]]:
    raw_turns = row.get("turns")
    if raw_turns is None:
        if loss_type == "kl":
            return [_normalize_distribution_turn(row, answer_instruction, "row")]
        return [_normalize_sft_turn(row, answer_instruction, loss_type, "row")]

    if not isinstance(raw_turns, list) or not raw_turns:
        raise ValueError("C2L multi-turn profile rows must contain a non-empty list `turns`.")

    inherited_sft_target = None
    if loss_type != "kl":
        inherited_sft_target = _normalize_sft_target(row.get(_C2L_SFT_TARGET_FIELD), "row")

    turns: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(raw_turns):
        if not isinstance(turn, dict):
            raise TypeError(f"C2L profile turns[{turn_index}] must be a JSON object.")
        if loss_type == "kl":
            turns.append(_normalize_distribution_turn(turn, answer_instruction, f"turns[{turn_index}]"))
        else:
            turns.append(
                _normalize_sft_turn(
                    turn,
                    answer_instruction,
                    loss_type,
                    f"turns[{turn_index}]",
                    inherited_sft_target=inherited_sft_target,
                )
            )

    return turns


def _canonicalize_streaming_profile_record(row: dict[str, Any], loss_type: str) -> dict[str, Any]:
    if loss_type != "kl":
        return _project_sft_profile_record(row, loss_type)

    canonical_row = dict(row)
    raw_turns = row.get("turns")
    if raw_turns is not None:
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError("C2L multi-turn profile rows must contain a non-empty list `turns`.")

        canonical_turns = []
        for turn_index, turn in enumerate(raw_turns):
            if not isinstance(turn, dict):
                raise TypeError(f"C2L profile turns[{turn_index}] must be a JSON object.")
            canonical_turns.append(_normalize_distribution_turn(turn, None, f"turns[{turn_index}]"))

        canonical_row["turns"] = canonical_turns

    original_record = row.get("original_record")
    if isinstance(original_record, dict):
        canonical_original_record = {
            field_name: original_record[field_name]
            for field_name in ("images", "videos")
            if field_name in original_record
        }
        if canonical_original_record:
            canonical_row["original_record"] = canonical_original_record
        else:
            canonical_row.pop("original_record", None)

    return canonical_row


def _project_sft_profile_record(row: dict[str, Any], loss_type: str) -> dict[str, Any]:
    """Build the streaming SFT schema without carrying raw teacher payloads forward."""
    canonical_row = {
        field_name: copy.deepcopy(row[field_name])
        for field_name in _C2L_SFT_PROFILE_COLUMNS
        if field_name in row
    }
    canonical_row[_C2L_SFT_TARGET_FIELD] = _normalize_sft_target(
        row.get(_C2L_SFT_TARGET_FIELD), "row"
    )
    raw_turns = row.get("turns")
    if raw_turns is not None:
        if not isinstance(raw_turns, list) or not raw_turns:
            raise ValueError("C2L multi-turn profile rows must contain a non-empty list `turns`.")

        inherited_sft_target = _normalize_sft_target(row.get(_C2L_SFT_TARGET_FIELD), "row")
        canonical_turns = []
        for turn_index, turn in enumerate(raw_turns):
            if not isinstance(turn, dict):
                raise TypeError(f"C2L profile turns[{turn_index}] must be a JSON object.")
            canonical_turns.append(
                _normalize_sft_turn(
                    turn,
                    None,
                    loss_type,
                    f"turns[{turn_index}]",
                    inherited_sft_target=inherited_sft_target,
                )
            )
        canonical_row["turns"] = canonical_turns
    else:
        canonical_row.update(_normalize_sft_turn(row, None, loss_type, "row"))

    original_record = row.get("original_record")
    if isinstance(original_record, dict):
        canonical_original_record = {
            field_name: copy.deepcopy(original_record[field_name])
            for field_name in ("images", "videos")
            if field_name in original_record
        }
        if canonical_original_record:
            canonical_row["original_record"] = canonical_original_record
        else:
            canonical_row.pop("original_record", None)

    return canonical_row


def _sft_map_remove_columns(dataset: "Dataset | IterableDataset") -> list[str]:
    """Return every known raw column plus fixed sparse fields for SFT map replacement."""
    column_names = getattr(dataset, "column_names", None)
    if column_names is not None:
        return list(column_names)

    # IterableDataset maps may have unknown features after an earlier map. The
    # sparse teacher names remain fixed, so remove them and every SFT staging
    # field explicitly before merging the callback output.
    return list(
        dict.fromkeys(
            (
                *_C2L_SFT_PROFILE_COLUMNS,
                *_C2L_SPARSE_TEACHER_COLUMNS,
                _C2L_ANSWER_INSTRUCTION_COLUMN,
                _C2L_VIDEO_ROOT_COLUMN,
                _C2L_SOURCE_VIDEO_ROOT_COLUMN,
            )
        )
    )


def _map_c2l_profile_dataset(
    dataset: "Dataset | IterableDataset", function: Any, loss_type: str
) -> "Dataset | IterableDataset":
    if loss_type == "kl":
        return dataset.map(function)

    return dataset.map(function, remove_columns=_sft_map_remove_columns(dataset))


def _validate_turn_alignment(turns: list[dict[str, Any]], strict_alignment: bool) -> None:
    if not strict_alignment:
        return

    for turn_index, turn in enumerate(turns):
        token_count = len(turn["generated_token_ids"])
        topk_id_count = len(turn["logprobs_indices"])
        topk_val_count = len(turn["logprobs_vals"])
        if token_count != topk_id_count or token_count != topk_val_count:
            raise ValueError(
                "C2L target length mismatch in profile turn "
                f"{turn_index}: generated_token_ids={token_count}, "
                f"logprobs_indices={topk_id_count}, logprobs_vals={topk_val_count}."
            )


def _answer_labels(
    answer_token_ids: list[int],
    loss_type: str,
    tokenizer: "PreTrainedTokenizer",
) -> list[int]:
    """Supervise the answer and its EOS, but not the template's trailing newline.

    `format_assistant` renders `{{content}}<|im_end|>\\n`, so the SFT target ids end with an
    EOS token followed by a newline. Generation stops at EOS, so that newline is never
    produced at inference; supervising it only adds a free token to the loss denominator.
    It stays in `input_ids` so multi-turn context remains template-correct. The KL path is
    untouched because it uses the profiled generation ids verbatim.
    """
    if loss_type == "kl":
        return answer_token_ids

    labels = list(answer_token_ids)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and len(labels) >= 2 and labels[-2] == eos_token_id:
        labels[-1] = IGNORE_INDEX

    return labels


def _base_user_content(question: str, include_video: bool) -> str:
    if not include_video:
        return question
    return f"{VIDEO_PLACEHOLDER}\n{question}" if question else VIDEO_PLACEHOLDER


def _process_base_messages(
    base_messages: list[dict[str, str]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    base_videos: list[Any],
) -> tuple[list[dict[str, str]], list[int], list[int]]:
    if not base_videos:
        return base_messages, [], []

    base_messages = template.mm_plugin.process_messages(base_messages, [], base_videos, [], processor)
    base_prefix_ids, base_prefix_labels = template.mm_plugin.process_token_ids(
        [], [], [], base_videos, [], tokenizer, processor
    )
    if base_prefix_labels is None or len(base_prefix_labels) != len(base_prefix_ids):
        base_prefix_labels = [IGNORE_INDEX] * len(base_prefix_ids)

    return base_messages, base_prefix_ids, base_prefix_labels


def _encode_base_turns(
    turns: list[dict[str, Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    row_has_turns: bool,
    loss_type: str,
    base_videos: list[Any] | None = None,
) -> tuple[list[int], list[int], list[list[int]], list[list[float]]]:
    base_input_ids: list[int] = []
    base_labels: list[int] = []
    logprobs_indices: list[list[int]] = []
    logprobs_vals: list[list[float]] = []
    base_videos = base_videos or []

    if not row_has_turns:
        turn = turns[0]
        base_messages = [
            {"role": "user", "content": _base_user_content(turn["question"], bool(base_videos))},
            {"role": "assistant", "content": turn.get("selected_answer", "")},
        ]
        base_messages, base_prefix_ids, base_prefix_labels = _process_base_messages(
            base_messages, template, tokenizer, processor, base_videos
        )
        base_source_ids, target_ids = template.encode_oneturn(tokenizer, base_messages)
        answer_token_ids = turn["generated_token_ids"] if loss_type == "kl" else target_ids
        base_input_ids = base_prefix_ids + base_source_ids + answer_token_ids
        base_labels = (
            base_prefix_labels
            + [IGNORE_INDEX] * len(base_source_ids)
            + _answer_labels(answer_token_ids, loss_type, tokenizer)
        )
        if loss_type == "kl":
            logprobs_indices.extend(turn["logprobs_indices"])
            logprobs_vals.extend(turn["logprobs_vals"])
        return base_input_ids, base_labels, logprobs_indices, logprobs_vals

    base_messages = []
    for turn_index, turn in enumerate(turns):
        base_messages.append(
            {
                "role": "user",
                "content": _base_user_content(turn["question"], bool(base_videos) and turn_index == 0),
            }
        )
        base_messages.append({"role": "assistant", "content": turn.get("selected_answer", turn["generated_answer"])})

    base_messages, base_prefix_ids, base_prefix_labels = _process_base_messages(
        base_messages, template, tokenizer, processor, base_videos
    )
    encoded_turns = list(template.encode_multiturn(tokenizer, base_messages))
    if len(encoded_turns) != len(turns):
        raise ValueError(f"Encoded {len(encoded_turns)} turns for {len(turns)} C2L profile turns.")

    base_input_ids.extend(base_prefix_ids)
    base_labels.extend(base_prefix_labels)
    for (source_ids, target_ids), turn in zip(encoded_turns, turns, strict=True):
        answer_token_ids = turn["generated_token_ids"] if loss_type == "kl" else target_ids
        base_input_ids.extend(source_ids)
        base_input_ids.extend(answer_token_ids)
        base_labels.extend([IGNORE_INDEX] * len(source_ids))
        base_labels.extend(_answer_labels(answer_token_ids, loss_type, tokenizer))
        if loss_type == "kl":
            logprobs_indices.extend(turn["logprobs_indices"])
            logprobs_vals.extend(turn["logprobs_vals"])

    return base_input_ids, base_labels, logprobs_indices, logprobs_vals


def preprocess_profile_record(
    row: dict[str, Any],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
    answer_instruction: str | None = None,
    video_root: str | None = None,
    source_video_root: str | None = None,
    video_merge_candidates: list[C2LVideoMergeCandidate] | None = None,
    video_merge_seed: int = 0,
) -> dict[str, Any]:
    row_has_turns = "turns" in row
    loss_type = getattr(finetuning_args, "c2l_loss_type", "kl")
    turns = _get_profile_turns(row, answer_instruction, loss_type)
    if loss_type == "kl":
        _validate_turn_alignment(turns, finetuning_args.c2l_strict_target_alignment)

    base_input_video = bool(getattr(finetuning_args, "base_input_video", False))
    base_input_frame_number = int(getattr(finetuning_args, "base_input_frame_number", 4))
    base_with_last_2seconds = bool(getattr(finetuning_args, "base_with_last_2seconds", False))
    base_last_2seconds_max_frames = int(getattr(finetuning_args, "base_last_2seconds_max_frames", 128))
    base_tail_seconds = float(getattr(finetuning_args, "base_tail_seconds", 2.0))
    video_merge_extend = bool(getattr(finetuning_args, "video_merge_extend", False))
    video_merge_extend_perc = float(getattr(finetuning_args, "video_merge_extend_perc", 0.0))
    images = _resolve_media_paths(_get_row_images(row, data_args.media_dir), data_args.media_dir)
    videos, base_videos, encoder_video_timestamps, encoder_video_end_seconds, base_video_timestamps = _resolve_video_paths(
        _get_row_videos(row, data_args.media_dir),
        data_args.media_dir,
        base_input_frame_number if base_input_video else 0,
        video_root=video_root,
        source_video_root=source_video_root,
        base_with_last_2seconds=base_with_last_2seconds,
        base_last_2seconds_max_frames=base_last_2seconds_max_frames,
        include_timestamps=True,
        base_tail_seconds=base_tail_seconds,
    )
    if video_merge_extend and base_input_video and base_with_last_2seconds:
        videos, base_videos, encoder_video_timestamps, encoder_video_end_seconds = _maybe_apply_video_merge_extend(
            row,
            videos,
            base_videos,
            data_args.media_dir,
            video_root,
            source_video_root,
            base_input_frame_number,
            base_last_2seconds_max_frames,
            video_merge_candidates,
            video_merge_seed,
            video_merge_extend_perc,
            encoder_video_timestamps,
            encoder_video_end_seconds,
            base_video_timestamps,
            base_tail_seconds=base_tail_seconds,
        )
    if not images and not videos:
        raise ValueError("C2L multimodal context requires at least one image or video.")
    if base_input_video and not base_videos:
        raise ValueError("C2L `base_input_video` requires profile rows with video context.")

    base_input_ids, base_labels, logprobs_indices, logprobs_vals = _encode_base_turns(
        turns,
        template,
        tokenizer,
        processor,
        row_has_turns,
        loss_type,
        base_videos if base_input_video else [],
    )
    encoder_question = "\n\n".join(turn["question"] for turn in turns if turn["question"])

    media_placeholder = VIDEO_PLACEHOLDER if videos else IMAGE_PLACEHOLDER
    encoder_content = media_placeholder
    if getattr(finetuning_args, "c2l_encoder_context", "image_only") == "image_text":
        encoder_content = f"{media_placeholder}\n{encoder_question}" if encoder_question else media_placeholder

    encoder_messages = [
        {"role": "user", "content": encoder_content},
        {"role": "assistant", "content": ""},
    ]
    encoder_messages = template.mm_plugin.process_messages(
        encoder_messages, images, videos, [], processor
    )
    encoder_prefix_ids, _ = template.mm_plugin.process_token_ids(
        [], [], images, videos, [], tokenizer, processor
    )
    encoder_source_ids, _ = template.encode_oneturn(tokenizer, encoder_messages)
    # tokenizer.decode(encoder_source_ids): '<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|image_pad|><|vision_end|><|im_end|>\n<|im_start|>assistant\n'
    # encoder_source_ids: [248045, 846, 198, 248053, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248056, 248054, 248046, 198, 248045, 74455, 198]
    # encoder_prefix_ids: []
    encoder_input_ids = encoder_prefix_ids + encoder_source_ids

    target_token_count = sum(len(turn["generated_token_ids"]) for turn in turns) if loss_type == "kl" else 0
    if loss_type == "kl" and finetuning_args.c2l_strict_target_alignment and target_token_count != len(logprobs_vals):
        raise ValueError(
            "C2L target length mismatch: "
            f"generated_token_ids={target_token_count}, logprobs_vals={len(logprobs_vals)}."
        )

    result = {
        "input_ids": base_input_ids,
        "attention_mask": [1] * len(base_input_ids),
        "labels": base_labels,
        "encoder_input_ids": encoder_input_ids,
        "encoder_attention_mask": [1] * len(encoder_input_ids),
        "encoder_labels": [IGNORE_INDEX] * len(encoder_input_ids),
        "images": images,
        "videos": videos,
        "encoder_video_timestamps": encoder_video_timestamps,
        "encoder_video_end_seconds": encoder_video_end_seconds,
        "base_images": [],
        "base_videos": base_videos if base_input_video else [],
    }
    if loss_type == "kl":
        result["logprobs_indices"] = logprobs_indices
        result["logprobs_vals"] = logprobs_vals
    return result


def get_c2l_dataset(
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: "ProcessorMixin",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
    dataset_names: list[str] | None,
    seed: int,
    repeat_streaming: bool = False,
) -> "Dataset | IterableDataset | None":
    if dataset_names is None:
        return None

    profile_sources = _resolve_profile_sources(dataset_names, data_args)
    loss_type = getattr(finetuning_args, "c2l_loss_type", "kl")
    video_merge_candidates: list[C2LVideoMergeCandidate] | None = None
    if bool(getattr(finetuning_args, "video_merge_extend", False)):
        video_merge_candidates = _collect_video_merge_candidates(profile_sources, data_args)
        if video_merge_candidates:
            logger.info_rank0(
                "Loaded %d C2L video-merge candidates for video_merge_extend.",
                len(video_merge_candidates),
            )
        else:
            logger.warning_rank0(
                "C2L `video_merge_extend` is enabled, but no extra video candidates were found."
            )

    datasets = []
    for profile_source in profile_sources:
        dataset = load_dataset(
            "json",
            data_files=profile_source.data_files,
            split="train",
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            streaming=data_args.streaming,
        )
        dataset = dataset.filter(
            lambda row, profile_source=profile_source: _row_has_existing_videos(
                row,
                data_args.media_dir,
                profile_source.video_root,
                profile_source.source_video_root,
            )
        )
        if data_args.streaming:
            dataset = _map_c2l_profile_dataset(
                dataset,
                lambda row, profile_source=profile_source, loss_type=loss_type: {
                    **_canonicalize_streaming_profile_record(row, loss_type),
                    _C2L_ANSWER_INSTRUCTION_COLUMN: profile_source.answer_instruction,
                    _C2L_VIDEO_ROOT_COLUMN: profile_source.video_root or "",
                    _C2L_SOURCE_VIDEO_ROOT_COLUMN: profile_source.source_video_root or "",
                },
                loss_type,
            )
            datasets.append(dataset)
            continue

        dataset = _map_c2l_profile_dataset(
            dataset,
            lambda row, answer_instruction=profile_source.answer_instruction: preprocess_profile_record(
                row,
                template,
                tokenizer,
                processor,
                data_args,
                finetuning_args,
                answer_instruction=answer_instruction,
                video_root=profile_source.video_root,
                source_video_root=profile_source.source_video_root,
                video_merge_candidates=video_merge_candidates,
                video_merge_seed=seed,
            ),
            loss_type,
        )
        datasets.append(dataset)

    dataset = merge_dataset(datasets, data_args, seed=seed)
    if not data_args.streaming:
        return dataset

    dataset = _map_c2l_profile_dataset(
        dataset,
        lambda row: preprocess_profile_record(
            row,
            template,
            tokenizer,
            processor,
            data_args,
            finetuning_args,
            answer_instruction=row.get(_C2L_ANSWER_INSTRUCTION_COLUMN) or None,
            video_root=row.get(_C2L_VIDEO_ROOT_COLUMN) or None,
            source_video_root=row.get(_C2L_SOURCE_VIDEO_ROOT_COLUMN) or None,
            video_merge_candidates=video_merge_candidates,
            video_merge_seed=seed,
        ),
        loss_type,
    )
    if repeat_streaming:
        repeat = getattr(dataset, "repeat", None)
        logger.info_rank0("Repeating finite C2L streaming dataset for max_steps-based training.")
        dataset = repeat(None) if callable(repeat) else _RepeatIterableDataset(dataset)

    return dataset


def _local_frame_period(frame_times: list[float]) -> float:
    for previous, current in zip(reversed(frame_times[:-1]), reversed(frame_times[1:]), strict=True):
        period = current - previous
        if period > 0:
            return period
    return 1.0


def _format_timestamp_seconds(timestamp: float) -> str:
    return f"{float(timestamp):.1f}".rstrip("0").rstrip(".")


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
        labels.append(f"{_format_timestamp_seconds(start)}s-{_format_timestamp_seconds(end)}s")
    return labels


def _unwrap_timestamp_scalar(value: Any, field_name: str, sample_idx: int) -> float | None:
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                "C2L timestamp interleaving expected one "
                f"{field_name} for sample {sample_idx}, got {value!r}."
            )
        value = value[0]

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "C2L timestamp interleaving expected a numeric "
            f"{field_name} for sample {sample_idx}, got {value!r}."
        ) from exc


class C2LDataCollator:
    def __init__(
        self,
        tokenizer: "PreTrainedTokenizer",
        encoder_collator: SFTDataCollatorWith4DAttentionMask,
        interleave_timestamp: bool = False,
        frame_per_lora: int = 1,
        loss_type: str = "kl",
        vjepa2_video_processor=None,
        vjepa2_model_config=None,
        vjepa2_window_size: int = 16,
        vjepa2_window_stride: int = 2,
        vjepa2_spatial_pool_size: int = 2,
        vjepa2_input_dtype: torch.dtype | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.encoder_collator = encoder_collator
        self.interleave_timestamp = interleave_timestamp
        self.frame_per_lora = frame_per_lora
        self.loss_type = loss_type
        self.vjepa2_video_processor = vjepa2_video_processor
        self.vjepa2_model_config = vjepa2_model_config
        self.vjepa2_window_size = vjepa2_window_size
        self.vjepa2_window_stride = vjepa2_window_stride
        self.vjepa2_spatial_pool_size = vjepa2_spatial_pool_size
        self.vjepa2_input_dtype = vjepa2_input_dtype

    def _collate_encoder_inputs(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if self.vjepa2_video_processor is not None:
            videos = []
            for sample_idx, feature in enumerate(features):
                sample_videos = feature.get("videos", [])
                if feature.get("images"):
                    raise ValueError("V-JEPA2 C2L context encoding supports video inputs only.")
                if len(sample_videos) != 1:
                    raise ValueError(
                        "V-JEPA2 C2L context encoding requires exactly one video per sample: "
                        f"sample={sample_idx}, videos={len(sample_videos)}."
                    )
                videos.append(sample_videos[0])
            return prepare_vjepa2_encoder_inputs(
                videos,
                self.vjepa2_video_processor,
                model_config=self.vjepa2_model_config,
                window_size=self.vjepa2_window_size,
                stride=self.vjepa2_window_stride,
                spatial_pool_size=self.vjepa2_spatial_pool_size,
                dtype=self.vjepa2_input_dtype,
            )

        encoder_features = [
            {
                "input_ids": feature["encoder_input_ids"],
                "attention_mask": feature["encoder_attention_mask"],
                "labels": feature["encoder_labels"],
                "images": feature["images"],
                "videos": feature["videos"],
                "audios": None,
            }
            for feature in features
        ]
        encoder_inputs = self.encoder_collator(encoder_features)
        encoder_inputs.pop("labels", None)
        self._annotate_video_temporal_patch_size(encoder_inputs)
        return encoder_inputs

    def _annotate_video_temporal_patch_size(self, encoder_inputs: dict[str, Any]) -> None:
        video_grid_thw = encoder_inputs.get("video_grid_thw")
        if video_grid_thw is None or int(video_grid_thw.numel()) == 0:
            return

        processor = getattr(self.encoder_collator, "processor", None)
        video_processor = getattr(processor, "video_processor", None) or getattr(processor, "image_processor", None)
        temporal_patch_size = int(getattr(video_processor, "temporal_patch_size", 0) or 0)
        if temporal_patch_size < 1:
            return

        encoder_inputs["video_temporal_patch_size"] = torch.full(
            (video_grid_thw.shape[0],),
            temporal_patch_size,
            dtype=torch.long,
            device=video_grid_thw.device,
        )

    @staticmethod
    def _processor_decoded_timestamps(
        encoder_inputs: dict[str, Any], sample_idx: int
    ) -> list[float] | None:
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

    def _add_timestamp_inputs(
        self, encoder_inputs: dict[str, Any], features: list[dict[str, Any]]
    ) -> None:
        timestamp_labels = []
        timestamp_block_owners = []
        for sample_idx, feature in enumerate(features):
            videos = feature.get("videos", [])
            if len(videos) != 1:
                raise ValueError(
                    "C2L timestamp interleaving requires exactly one video context for "
                    f"sample {sample_idx}, got {len(videos)}."
                )

            sample_timestamps = feature.get("encoder_video_timestamps")
            sample_end_seconds = feature.get("encoder_video_end_seconds")
            timestamps = (
                sample_timestamps[0]
                if isinstance(sample_timestamps, list) and len(sample_timestamps) == 1
                else None
            )
            terminal_time = _unwrap_timestamp_scalar(
                sample_end_seconds, "encoder video end time", sample_idx
            )
            if timestamps is None:
                timestamps = self._processor_decoded_timestamps(encoder_inputs, sample_idx)
            if timestamps is None:
                raise ValueError(
                    "C2L timestamp interleaving is missing decoded frame times for "
                    f"sample {sample_idx}."
                )

            if self.frame_per_lora == 1 and len(timestamps) % 2:
                timestamps = list(timestamps) + [timestamps[-1]]
            labels = _timestamp_labels_from_frame_times(
                [float(timestamp) for timestamp in timestamps], self.frame_per_lora, terminal_time
            )
            timestamp_labels.extend(labels)
            timestamp_block_owners.extend([sample_idx] * len(labels))

        tokenized = self.tokenizer(timestamp_labels, add_special_tokens=False)
        timestamp_features = [
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            for input_ids, attention_mask in zip(
                tokenized["input_ids"],
                tokenized.get("attention_mask", [[1] * len(input_ids) for input_ids in tokenized["input_ids"]]),
                strict=True,
            )
        ]
        padded_timestamps = self.tokenizer.pad(timestamp_features, padding=True, return_tensors="pt")
        encoder_inputs["timestamp_input_ids"] = padded_timestamps["input_ids"]
        encoder_inputs["timestamp_attention_mask"] = padded_timestamps["attention_mask"]
        encoder_inputs["timestamp_block_owners"] = torch.tensor(timestamp_block_owners, dtype=torch.long)

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        base_features = []
        if self.loss_type == "kl":
            logprobs_indices = []
            logprobs_vals = []
            target_lengths = []
        base_has_media = False

        for feature in features:
            base_feature = {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["labels"],
                "images": feature.get("base_images", []),
                "videos": feature.get("base_videos", []),
                "audios": None,
            }
            base_has_media = base_has_media or bool(base_feature["images"] or base_feature["videos"])
            base_features.append(base_feature)
            if self.loss_type == "kl":
                logprobs_indices.append(feature["logprobs_indices"])
                logprobs_vals.append(feature["logprobs_vals"])
                target_lengths.append(len(feature["logprobs_vals"]))
        # it do the right padding
        if base_has_media:
            base_batch = self.encoder_collator(base_features)
        else:
            labels = [torch.tensor(feature["labels"], dtype=torch.long) for feature in features]
            base_text_features = [
                {
                    "input_ids": feature["input_ids"],
                    "attention_mask": feature["attention_mask"],
                }
                for feature in features
            ]
            base_batch = self.tokenizer.pad(base_text_features, padding=True, return_tensors="pt")
            padded_labels = torch.nn.utils.rnn.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            base_batch["labels"] = padded_labels

        encoder_inputs = self._collate_encoder_inputs(features)
        if self.interleave_timestamp:
            self._add_timestamp_inputs(encoder_inputs, features)

        batch = {
            **base_batch,
            "encoder_inputs": encoder_inputs,
            "n_ctx_chunks": torch.ones(len(features), dtype=torch.int32),
            "n_queries": torch.ones(len(features), dtype=torch.int32),
        }
        if self.loss_type == "kl":
            max_target_len = max(target_lengths)
            max_topk = max(len(token_indices) for sample in logprobs_indices for token_indices in sample)
            padded_indices = torch.zeros((len(features), max_target_len, max_topk), dtype=torch.long)
            padded_vals = torch.full((len(features), max_target_len, max_topk), float("-inf"), dtype=torch.float32)
            target_mask = torch.zeros((len(features), max_target_len), dtype=torch.bool)

            for sample_idx, (sample_indices, sample_vals) in enumerate(zip(logprobs_indices, logprobs_vals, strict=True)):
                for token_idx, (token_indices, token_vals) in enumerate(zip(sample_indices, sample_vals, strict=True)):
                    topk = len(token_indices)
                    padded_indices[sample_idx, token_idx, :topk] = torch.tensor(token_indices, dtype=torch.long)
                    padded_vals[sample_idx, token_idx, :topk] = torch.tensor(token_vals, dtype=torch.float32)
                    target_mask[sample_idx, token_idx] = True

            batch.update(
                {
                    "logprobs_indices": padded_indices,
                    "logprobs_vals": padded_vals,
                    "logprobs_mask": target_mask,
                }
            )

        return batch
