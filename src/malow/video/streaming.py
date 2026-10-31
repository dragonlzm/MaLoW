# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
"""StreamingBench evaluation for Qwen2-VL, Qwen3-VL, and Qwen3.5 models."""
from __future__ import annotations
import argparse
import json
import math
import os
if cache_root := os.environ.get('MALOW_TRITON_CACHE_ROOT'):
    os.environ['TRITON_CACHE_DIR'] = os.path.join(cache_root, f"rank-{os.environ.get('LOCAL_RANK', '0')}")
import re
import sys
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any
from malow.video.sampling import add_c2l_timestamp_inputs, c2l_encoder_frame_count, load_base_last_2seconds_c2l_video_frames, load_base_last_2seconds_c2l_video_frames_with_timestamps, load_split_c2l_video_frames, load_split_c2l_video_frames_with_timestamps, processor_predecoded_video_inputs
from malow.video.interactive_options import (
    add_interactive_arguments, resolve_interactive_arguments, guard_interactive_run,
    init_resolution_log, write_interactive_report, select_verification_session,
)
from malow.video.interactive import (
    advance_session, new_session, interactive_config_from_args, session_generated_loras,
    verify_session_against_single_pass, describe_streaming_path, finish_session,
    clamp_question_time,
)
from malow.video.sampling import load_c2l_base_tail_frames
from malow.video.progress import (PredictionJournal, bind_session, save_session, saved_session,
                                 cache_get, cache_put, digest, file_identity)
OPTION_LETTERS = tuple('ABCD')
STREAMINGBENCH_VIDEO_TOTAL_PIXELS = 469762048
STREAMINGBENCH_VIDEO_MAX_PIXELS = 786432
STREAMINGBENCH_MAX_FRAMES = 768
TASK_FILE_CANDIDATES = {'real': ('questions_real.json', 'questions_real_stream.json'), 'omni': ('questions_omni.json', 'questions_omni_stream.json'), 'sqa': ('questions_sqa_stream.json', 'questions_sqa.json'), 'proactive': ('questions_proactive.json', 'questions_proactive_50.json')}
WEIGHTED_SUMMARY_WEIGHTS = {'real': 2500, 'omni': 1500, 'proactive': 250, 'sqa': 250}
MACHINE_ROOT_PREFIXES = ()
PROMPT_MC = 'You are an advanced video question-answering AI assistant. You have been provided with a video and a multiple-choice question related to the video. Carefully analyze only the visible video context up to the current timestamp and choose the best option. Respond with only the letter (A, B, C, or D) of the correct option.\n\nQuestion: {question}\n\nOptions:\n{options}\n\nThe best option is:'
PROMPT_SQA = 'You are an advanced video question-answering AI assistant. You have been provided with a video, prior contextual question-answer history, and a multiple-choice question related to the video. Answer the current question using the video and the contextual information. Respond with only the letter (A, B, C, or D) of the correct option.\n\n{context}\n\nHere is the question. Answer it and do not confuse it with the previous conversation.\nQuestion: {question}\n\nOptions:\n{options}\n\nThe best option is:'
PROMPT_PROACTIVE_YESNO = 'You are an advanced video question-answering AI assistant. You have been provided with the current video prefix and a question related to it. Decide whether this is the right time to output the requested content. You can only answer yes or no.\n\nQuestion: {question}\n\nAnswer:'
PROMPT_PROACTIVE_FINAL = 'You are an advanced video question-answering AI assistant. You have been provided with the current video prefix and a question related to it. Output the requested answer content directly.\n\nQuestion: {question}\n\nThe answer is:'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate Qwen models on StreamingBench.')
    parser.add_argument('--model-id', required=True, help='Hugging Face model id or local path.')
    parser.add_argument('--model-family', choices=('auto', 'qwen2vl', 'qwen3vl', 'qwen35'), default='auto')
    parser.add_argument('--c2l-checkpoint', default=None, help='Optional Qwen3.5 or Qwen3-VL Video C2L checkpoint directory.')
    parser.add_argument('--data-root', default=None, help='Directory containing StreamingBench question JSON files.')
    parser.add_argument('--data-file', default=None, help='Single StreamingBench question JSON file.')
    parser.add_argument('--task', action='append', choices=tuple(TASK_FILE_CANDIDATES), help='Task to run. Repeatable.')
    parser.add_argument('--output-dir', required=True, help='Directory for rank logs, merged predictions, and stats.')
    parser.add_argument('--context-time', type=int, default=-1, help='Seconds of history before timestamp; <=0 uses [0, timestamp].')
    parser.add_argument('--limit', type=int, default=None, help='Optional per-task entry limit for smoke tests.')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for real/omni multiple-choice tasks.')
    parser.add_argument('--dtype', default='bfloat16', help='Model dtype: bfloat16, float16, float32, or auto.')
    parser.add_argument('--device-map', default='auto', help='Transformers device_map.')
    parser.add_argument('--vision-encoder-only', action='store_true', help='Load only the projected Qwen3.5 or Qwen3-VL vision tower for the frozen C2L context encoder.')
    parser.add_argument('--trust-remote-code', action='store_true')
    parser.add_argument('--system-prompt', default='You are a helpful assistant.')
    parser.add_argument('--enable-thinking', choices=('true', 'false'), default='false')
    parser.add_argument('--video-min-pixels', type=int, default=256 * 32 * 32)
    parser.add_argument('--video-max-pixels', type=int, default=2048 * 32 * 32)
    parser.add_argument('--video-total-pixels', type=int, default=224 * 1024 * 32 * 32)
    parser.add_argument('--video-fps', type=float, default=2.0)
    parser.add_argument('--max-frames', type=int, default=32)
    parser.add_argument('--num-frames', type=int, default=None)
    parser.add_argument('--last-2second-1fps', action='store_true', help='Sample only the last 2 seconds of each question interval at 1 FPS. Intervals shorter than 2 seconds use the original interval at 1 FPS.')
    parser.add_argument('--last-seconds-1fps', type=float, default=None, help='Sample only the last N seconds of each question interval at 1 FPS. Intervals shorter than N seconds use the original interval at 1 FPS.')
    parser.add_argument('--use-streamingbench-frame-sampling', action='store_true', help='Use the original StreamingBench Qwen2-VL frame sampling policy: dynamic FPS by prefix duration, max_frames=768, video_total_pixels=469762048, and greedy decoding.')
    parser.add_argument('--max-new-tokens', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--top-k', type=int, default=None)
    parser.add_argument('--proactive-step', type=int, default=1)
    parser.add_argument('--proactive-extra-seconds', type=int, default=4)
    parser.add_argument('--user-defined-scaling', type=float, default=1.0, help='Multiplier for generated C2L LoRA weights.')
    parser.add_argument('--block-lora', choices=('auto', 'true', 'false'), default='auto', help='Use block-LoRA generation for C2L. `auto` reads checkpoint metadata.')
    parser.add_argument('--frame-per-lora', type=int, default=None, help='Number of raw video frames per generated LoRA block. Defaults to checkpoint metadata or 4.')
    parser.add_argument('--micro-c2l-forward-frame', type=int, default=None, help='Optional raw-frame cap for each micro-batch when generating block-LoRA C2L weights. For frame_per_lora=2, 768 means at most 384 LoRA blocks per micro-forward.')
    parser.add_argument('--base-input-video', action='store_true', help='Include a short tail video observation in C2L base inputs in addition to the text prompt.')
    parser.add_argument('--base-input-frame-number', type=int, default=4, help='Number of tail frames to include in C2L base inputs when --base-input-video is set.')
    parser.add_argument('--base-with-last-2seconds', action='store_true', help='For Video C2L base video input, sample base frames from the tail window (default 2 seconds) and sample the earlier interval at 1 FPS up to the C2L encoder frame cap.')
    parser.add_argument('--progress-ranks', choices=('rank0', 'all', 'none'), default=os.environ.get('PROGRESS_RANKS', 'rank0'), help='Which distributed ranks should show tqdm progress bars.')
    parser.add_argument('--dist-barrier-timeout-seconds', type=int, default=int(os.environ.get('STREAMINGBENCH_BARRIER_TIMEOUT_SECONDS', '86400')), help='Timeout for filesystem rank barriers. Default: 24 hours.')
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--base-tail-seconds', type=float, default=2.0, help='Base observation window in seconds when --base-with-last-2seconds is set.')
    add_interactive_arguments(parser)
    args = parser.parse_args()
    resolve_interactive_arguments(parser, args, worker=True)
    if not math.isfinite(args.base_tail_seconds) or args.base_tail_seconds <= 0:
        parser.error('--base-tail-seconds must be finite and > 0.')
    if args.micro_c2l_forward_frame is not None and args.micro_c2l_forward_frame < 1:
        parser.error('--micro-c2l-forward-frame must be >= 1 when set.')
    if args.base_input_video and args.base_input_frame_number < 1:
        parser.error('--base-input-frame-number must be >= 1 when --base-input-video is set.')
    if args.c2l_checkpoint and args.no_video:
        parser.error('Video C2L evaluation requires video input and cannot be used with --no-video.')
    if args.c2l_checkpoint and args.model_family == 'qwen2vl':
        parser.error('Video C2L evaluation is supported for Qwen3.5 and Qwen3-VL models.')
    if args.last_2second_1fps and args.no_video:
        parser.error('--last-2second-1fps requires video input and cannot be combined with --no-video.')
    if args.last_seconds_1fps is not None and args.last_seconds_1fps <= 0:
        parser.error('--last-seconds-1fps must be positive.')
    if args.last_seconds_1fps is not None and args.no_video:
        parser.error('--last-seconds-1fps requires video input and cannot be combined with --no-video.')
    if args.last_2second_1fps and args.last_seconds_1fps is not None:
        parser.error('--last-2second-1fps cannot be combined with --last-seconds-1fps.')
    if args.last_2second_1fps and args.use_streamingbench_frame_sampling:
        parser.error('--last-2second-1fps cannot be combined with --use-streamingbench-frame-sampling.')
    if args.last_seconds_1fps is not None and args.use_streamingbench_frame_sampling:
        parser.error('--last-seconds-1fps cannot be combined with --use-streamingbench-frame-sampling.')
    if args.last_2second_1fps and args.c2l_checkpoint:
        parser.error('--last-2second-1fps is for base-model evaluation; use --base-with-last-2seconds for Video C2L.')
    if args.last_seconds_1fps is not None and args.c2l_checkpoint:
        parser.error('--last-seconds-1fps is for base-model evaluation.')
    return args

def init_distributed() -> tuple[int, int, int]:
    import torch
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', str(rank)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return (rank, local_rank, world_size)

def barrier(task_dir: Path, tag: str, rank: int, world_size: int, run_token: str, timeout_seconds: int) -> None:
    if world_size <= 1:
        return
    marker = task_dir / f'.barrier_{run_token}_{tag}_rank_{rank:05d}'
    marker.write_text(json.dumps({'rank': rank, 'tag': tag, 'time': time.time()}) + '\n', encoding='utf-8')
    expected = [task_dir / f'.barrier_{run_token}_{tag}_rank_{idx:05d}' for idx in range(world_size)]
    deadline = time.time() + timeout_seconds
    while True:
        missing = [path.name for path in expected if not path.exists()]
        if not missing:
            return
        if time.time() > deadline:
            raise TimeoutError(f"Timed out after {timeout_seconds}s waiting for ranks at barrier '{tag}' in {task_dir}. Missing markers: {', '.join(missing)}")
        time.sleep(5.0)

def distributed_run_token() -> str:
    token_parts = [os.environ.get('STREAMINGBENCH_RUN_TOKEN'), os.environ.get('TORCHELASTIC_RUN_ID'), os.environ.get('SLURM_JOB_ID'), os.environ.get('MASTER_ADDR'), os.environ.get('MASTER_PORT')]
    token = '_'.join((part for part in token_parts if part and part.lower() != 'none')) or 'manual'
    return re.sub('[^A-Za-z0-9_.-]+', '_', token)

def resolve_device_map(device_map: str, local_rank: int, world_size: int):
    if world_size > 1 and device_map == 'auto':
        import torch
        return {'': f'cuda:{local_rank}'} if torch.cuda.is_available() else {'': 'cpu'}
    return device_map

def model_input_device(model) -> Any:
    import torch
    base = getattr(model, 'base_model', model)
    device = getattr(base, 'device', None)
    return torch.device(device) if device is not None else next(model.parameters()).device

def interactive_timestamp(value):
    if isinstance(value, (int, float)):
        result = float(value)
    else:
        parts = str(value).split(":")
        if len(parts) > 3:
            raise ValueError(f"Invalid timestamp: {value!r}")
        result = 0.0
        for part in parts:
            result = 60 * result + float(part)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Invalid timestamp: {value!r}")
    return result


def parse_timestamp(value: Any) -> int:
    parts = [int(float(part)) for part in str(value).strip().split(':')]
    seconds = 0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds

def format_options(options: list[Any]) -> list[str]:
    labels = []
    for idx, option in enumerate(options[:len(OPTION_LETTERS)]):
        text = str(option).strip()
        letter = OPTION_LETTERS[idx]
        if not re.match(f'^{letter}[.)]\\s*', text, flags=re.IGNORECASE):
            text = f'{letter}. {text}'
        labels.append(text)
    return labels

def infer_option(text: str, options: list[str] | None=None) -> str | None:
    text = text.strip()
    if '</think>' in text:
        text = text.rsplit('</think>', 1)[-1].strip()
    match = re.match('^(?:assistant\\s*[:：]\\s*)?\\(?([A-Da-d])\\)?\\s*(?:[.)：:,]|$)', text)
    if match:
        return match.group(1).upper()
    for pattern in ('\\b(?:answer|option|choice|letter)\\s*(?:is|:|：)?\\s*\\(?([A-Da-d])\\)?\\b', '\\b(?:select|choose|pick|chosen)\\s*(?:option|choice)?\\s*\\(?([A-Da-d])\\)?\\b'):
        matches = [m.group(1).upper() for m in re.finditer(pattern, text, flags=re.IGNORECASE)]
        if len(set(matches)) == 1:
            return matches[0]
    if options:
        cleaned = normalize_option_text(text)
        hits = []
        for letter, option in zip(OPTION_LETTERS, options, strict=False):
            opt = normalize_option_text(option)
            if opt and (cleaned == opt or opt in cleaned):
                hits.append(letter)
        if len(set(hits)) == 1:
            return hits[0]
    return None

def normalize_option_text(text: str) -> str:
    text = re.sub('^[A-Da-d][.)]\\s*', '', str(text).strip())
    text = re.sub('\\s+', ' ', text)
    return text.strip(' "\'.,;:!?').lower()

def load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)

def resolve_task_files(args: argparse.Namespace) -> dict[str, Path]:
    if args.data_file:
        tasks = args.task or ['real']
        if len(tasks) != 1:
            raise ValueError('--data-file can only be used with one --task')
        return {tasks[0]: Path(args.data_file).expanduser()}
    if not args.data_root:
        raise ValueError('Set --data-root or --data-file.')
    root = Path(args.data_root).expanduser()
    tasks = args.task or ['real', 'omni', 'sqa', 'proactive']
    files = {}
    for task in tasks:
        for name in TASK_FILE_CANDIDATES[task]:
            candidate = root / name
            if candidate.exists():
                files[task] = candidate
                break
        if task not in files:
            raise FileNotFoundError(f'No annotation file found for task={task} under {root}')
    return files

def parse_video_path_prefix_map() -> list[tuple[str, str]]:
    value = os.environ.get('STREAMINGBENCH_VIDEO_PATH_PREFIX_MAP', '')
    pairs = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        if '=' not in item:
            raise ValueError('STREAMINGBENCH_VIDEO_PATH_PREFIX_MAP entries must use old_prefix=new_prefix, separated by commas.')
        old, new = item.split('=', 1)
        pairs.append((old.rstrip('/'), new.rstrip('/')))
    return pairs

def replace_path_prefix(path: Path, old: str, new: str) -> Path | None:
    path_str = str(path)
    if path_str == old:
        return Path(new)
    prefix = f'{old}/'
    if path_str.startswith(prefix):
        return Path(new) / path_str[len(prefix):]
    return None

def candidate_video_paths(path: Path, annotation_file: Path) -> list[Path]:
    explicit_root = os.environ.get('MALOW_VIDEO_ROOT')
    candidates = ([Path(explicit_root) / path.name] if explicit_root else []) + [path]
    for old, new in parse_video_path_prefix_map():
        remapped = replace_path_prefix(path, old, new)
        if remapped is not None:
            candidates.append(remapped)
    hf_home = os.environ.get('HF_HOME')
    if hf_home:
        marker = '/huggingface/'
        path_str = str(path)
        if marker in path_str:
            candidates.append(Path(hf_home).expanduser() / path_str.split(marker, 1)[1])
    if 'huggingface' in path.parts:
        suffix = Path(*path.parts[path.parts.index('huggingface') + 1:])
        for root in MACHINE_ROOT_PREFIXES:
            candidates.append(Path(root) / 'huggingface' / suffix)
    if path.name:
        candidates.append(annotation_file.parent / path)
        candidates.append(annotation_file.parent.parent / 'videos' / path.name)
    unique = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in seen:
            unique.append(candidate)
            seen.add(candidate_str)
    return unique

def absolutize_video_path(video_path: str, annotation_file: Path) -> str:
    path = Path(video_path).expanduser()
    if not path.is_absolute():
        path = (annotation_file.parent / path).resolve()
    for candidate in candidate_video_paths(path, annotation_file):
        if candidate.exists():
            return str(candidate)
    return str(path)

def validate_video_path(video_path: str) -> None:
    if not Path(video_path).exists():
        raise FileNotFoundError(f'Video not found: {video_path}. If annotations were prepared on another machine, set HF_HOME to the local Hugging Face root or set STREAMINGBENCH_VIDEO_PATH_PREFIX_MAP=old_prefix=new_prefix.')

def entry_questions(entry: dict[str, Any]) -> list[dict[str, Any]]:
    questions = entry.get('questions')
    return questions if isinstance(questions, list) else []

def question_id(task: str, entry_idx: int, question_idx: int, question: dict[str, Any]) -> str:
    entry_idx = question.get("_source_entry_index", entry_idx)
    question_idx = question.get("_source_question_index", question_idx)
    return f"{task}:{entry_idx}:{question_idx}:{question.get('time_stamp', '')}"

def count_questions(entries: list[tuple[int, dict[str, Any]]], task: str, done: set[str] | None=None) -> tuple[int, int]:
    total = 0
    completed = 0
    done = done or set()
    for entry_idx, entry in entries:
        for question_idx, question in enumerate(entry_questions(entry)):
            total += 1
            if question_id(task, entry_idx, question_idx, question) in done:
                completed += 1
    return (total, completed)

def shard_entries(entries: list[Any], rank: int, world_size: int) -> list[tuple[int, Any]]:
    return [(idx, entry) for idx, entry in enumerate(entries) if idx % world_size == rank]

def group_video_entries(data, task, annotation_file):
    """Retain source identities and entry-local SQA context while grouping videos."""
    groups = {}
    for entry_idx, entry in enumerate(data):
        path = str(Path(absolutize_video_path(entry["video_path"], annotation_file)).resolve())
        if path not in groups:
            groups[path] = (entry_idx, {"video_path": path, "questions": []})
        context = "" if task == "sqa" else None
        for question_idx, original in enumerate(entry_questions(entry)):
            question = dict(original, _source_entry_index=entry_idx,
                            _source_question_index=question_idx, _source_context=context)
            groups[path][1]["questions"].append(question)
            if task == "sqa":
                if not context:
                    context = "Here is the contextual information related to the video. "
                opts = format_options(list(original.get("options", [])))
                context += (f"At timestamp {original['time_stamp']}, the following question and answer occurred: "
                            f"Question: {original.get('question')}; Options: {', '.join(opts)}; "
                            f"Answer: {original.get('answer')}. ")
    return [groups[path] for path in sorted(groups)]


def streaming_journal(args, rank_path, entries, task):
    if not getattr(args, "checkpoint_progress", False):
        return None
    expected = {question_id(task, entry_idx, i, q)
                for entry_idx, entry in entries for i, q in enumerate(entry_questions(entry))}
    return PredictionJournal(rank_path, lambda row: row["question_id"], expected)


def streamingbench_fps(start: int, end: int) -> float:
    duration = max(0, int(end) - int(start))
    if duration >= 300 and duration < 600:
        return 0.5
    if duration >= 600:
        return 0.2
    return 1.0

def parse_rate(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    try:
        if '/' in text:
            numerator, denominator = text.split('/', 1)
            denominator_value = float(denominator)
            if denominator_value == 0:
                return None
            return float(numerator) / denominator_value
        return float(text)
    except (TypeError, ValueError):
        return None

@lru_cache(maxsize=1024)
def probe_video_frame_info(video_path: str) -> tuple[int, float] | None:
    local_path = video_path[7:] if video_path.startswith('file://') else video_path
    try:
        from decord import VideoReader
        reader = VideoReader(local_path)
        total_frames = len(reader)
        video_fps = float(reader.get_avg_fps())
        if total_frames > 0 and video_fps > 0:
            return (total_frames, video_fps)
    except Exception:
        pass
    try:
        import ffmpeg
        probe = ffmpeg.probe(local_path)
        stream = next((item for item in probe.get('streams', []) if item.get('codec_type') == 'video'), None)
        if not stream:
            return None
        video_fps = parse_rate(stream.get('avg_frame_rate')) or parse_rate(stream.get('r_frame_rate'))
        if not video_fps or video_fps <= 0:
            return None
        total_frames = None
        if stream.get('nb_frames'):
            total_frames = int(stream['nb_frames'])
        if not total_frames:
            duration = stream.get('duration') or probe.get('format', {}).get('duration')
            if duration:
                total_frames = int(math.floor(float(duration) * video_fps))
        if total_frames and total_frames > 0:
            return (total_frames, video_fps)
    except Exception:
        pass
    return None

def qwen_frame_factor_count(num_frames: int | None) -> int:
    if num_frames is None:
        return 2
    return max(2, int(math.ceil(float(num_frames) / 2.0)) * 2)

def video_source_frame_count(video_path: str, start: int | float, end: int | float) -> int | None:
    frame_info = probe_video_frame_info(video_path)
    if frame_info is None:
        return None
    total_frames, video_fps = frame_info
    if total_frames <= 0 or video_fps <= 0:
        return None
    video_start = max(0.0, float(start))
    video_end = max(0.0, float(end))
    start_frame = math.ceil(video_start * video_fps)
    end_frame = min(math.floor(video_end * video_fps), total_frames - 1)
    return max(0, end_frame - start_frame + 1)

def min_decodable_source_frames(args: argparse.Namespace) -> int:
    if getattr(args, 'c2l_checkpoint', None) and getattr(args, 'base_input_video', False) and getattr(args, 'base_with_last_2seconds', False):
        return qwen_frame_factor_count(getattr(args, 'base_input_frame_number', 2))
    num_frames = None if args.use_streamingbench_frame_sampling else args.num_frames
    required_frames = qwen_frame_factor_count(num_frames)
    if getattr(args, 'c2l_checkpoint', None) and getattr(args, 'base_input_video', False):
        required_frames = max(required_frames, qwen_frame_factor_count(args.base_input_frame_number))
    return required_frames

def can_decode_video_time_range(video_path: str, start: int | float, end: int | float, min_source_frames: int) -> bool:
    adjusted_start, adjusted_end = adjust_video_time_range_for_reader(video_path, start, end, min_source_frames)
    source_frames = video_source_frame_count(video_path, adjusted_start, adjusted_end)
    return source_frames is None or source_frames >= min_source_frames

def adjust_video_time_range_for_reader(video_path: str, start: int | float, end: int | float, min_source_frames: int=2) -> tuple[int | float, int | float]:
    video_start = max(0.0, float(start))
    video_end = max(0.0, float(end))
    frame_info = probe_video_frame_info(video_path)
    if frame_info is None:
        return (start, end)
    total_frames, video_fps = frame_info
    required_frames = max(2, int(min_source_frames))
    if total_frames < required_frames or video_fps <= 0:
        return (start, end)
    max_duration = total_frames / video_fps
    end_clamped = max(0.0, min(video_end, max_duration))
    end_frame = min(math.floor(end_clamped * video_fps), total_frames - 1)
    latest_start_frame = end_frame - required_frames + 1
    if latest_start_frame < 0:
        return (start, end)
    start_clamped = max(0.0, min(video_start, max_duration))
    start_frame = math.ceil(start_clamped * video_fps)
    if start_frame <= latest_start_frame:
        return (start, end)
    adjusted_start = max(0.0, latest_start_frame / video_fps - 0.001)
    return (min(video_start, adjusted_start), video_end)

def last_2second_1fps_window(start: int | float, end: int | float) -> tuple[float, float, float, int]:
    return last_seconds_1fps_window(start, end, 2.0)

def last_seconds_1fps_window(start: int | float, end: int | float, seconds: int | float) -> tuple[float, float, float, int]:
    video_start = max(0.0, float(start))
    video_end = max(video_start, float(end))
    seconds = max(0.0, float(seconds))
    max_frames = max(1, int(math.ceil(seconds)))
    return (max(video_start, video_end - seconds), video_end, 1.0, max_frames)

def make_progress_bar(args: argparse.Namespace, task: str, rank: int, world_size: int, total: int, initial: int):
    if args.progress_ranks == 'none' or (args.progress_ranks == 'rank0' and rank != 0):
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        if rank == 0:
            print('Install tqdm or set PROGRESS_RANKS=none to silence progress-bar setup.', flush=True)
        return None
    desc = f'{task} rank {rank}/{world_size}'
    position = rank if args.progress_ranks == 'all' else 0
    return tqdm(total=total, initial=initial, desc=desc, unit='q', dynamic_ncols=True, position=position)

def make_video_content(video_path: str, start: int, end: int, args: argparse.Namespace) -> dict[str, Any]:
    video_total_pixels = args.video_total_pixels
    video_fps = args.video_fps
    max_frames = args.max_frames
    num_frames = args.num_frames
    raw_start = max(0, int(start))
    raw_end = max(0, int(end))
    if args.last_2second_1fps:
        raw_start, raw_end, video_fps, max_frames = last_2second_1fps_window(start, end)
        num_frames = None
    elif args.last_seconds_1fps is not None:
        raw_start, raw_end, video_fps, max_frames = last_seconds_1fps_window(start, end, args.last_seconds_1fps)
        num_frames = None
    elif args.use_streamingbench_frame_sampling:
        video_total_pixels = STREAMINGBENCH_VIDEO_TOTAL_PIXELS
        video_fps = streamingbench_fps(start, end)
        max_frames = STREAMINGBENCH_MAX_FRAMES
        num_frames = None
    video_start, video_end = adjust_video_time_range_for_reader(video_path, raw_start, raw_end, min_source_frames=qwen_frame_factor_count(num_frames))
    content: dict[str, Any] = {'type': 'video', 'video': video_path, 'video_start': video_start, 'video_end': video_end, 'min_pixels': args.video_min_pixels, 'max_pixels': args.video_max_pixels}
    if video_total_pixels:
        content['total_pixels'] = video_total_pixels
    if num_frames is not None:
        content['nframes'] = num_frames
    else:
        content['fps'] = video_fps
        content['max_frames'] = max_frames
    return content

def make_tail_video_content(video_path: str, start: int, end: int, args: argparse.Namespace) -> dict[str, Any]:
    content = make_video_content(video_path, start, end, args)
    fps = float(args.video_fps or 2.0)
    if args.use_streamingbench_frame_sampling:
        fps = streamingbench_fps(start, end)
    tail_seconds = float(args.base_input_frame_number) / max(fps, 1e-06)
    content['video_start'] = max(float(start), float(end) - tail_seconds)
    content['video_end'] = max(float(start), float(end))
    content.pop('fps', None)
    content.pop('max_frames', None)
    content['nframes'] = args.base_input_frame_number
    content['video_start'], content['video_end'] = adjust_video_time_range_for_reader(video_path, content['video_start'], content['video_end'], min_source_frames=qwen_frame_factor_count(args.base_input_frame_number))
    return content

def build_messages(prompt: str, video_path: str, start: int, end: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if not args.no_video:
        content.append(make_video_content(video_path, start, end, args))
    content.append({'type': 'text', 'text': prompt})
    messages = [{'role': 'user', 'content': content}]
    if args.system_prompt:
        messages.insert(0, {'role': 'system', 'content': args.system_prompt})
    return messages

def build_c2l_encoder_messages(video_path: str, start: int, end: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    return [{'role': 'user', 'content': [make_video_content(video_path, start, end, args)]}]

def build_c2l_base_messages(prompt: str, video_path: str, start: int, end: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.base_input_video:
        content = [make_tail_video_content(video_path, start, end, args), {'type': 'text', 'text': prompt}]
    else:
        content = [{'type': 'text', 'text': prompt}]
    messages = [{'role': 'user', 'content': content}]
    if args.system_prompt:
        messages.insert(0, {'role': 'system', 'content': args.system_prompt})
    return messages

def apply_chat_template(processor, messages: list[list[dict[str, Any]]], args: argparse.Namespace) -> str | list[str]:
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=args.enable_thinking == 'true')
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def annotate_video_temporal_patch_size(processor, encoder_inputs):
    video_grid_thw = encoder_inputs.get('video_grid_thw')
    if video_grid_thw is None or int(video_grid_thw.numel()) == 0:
        return encoder_inputs
    video_processor = getattr(processor, 'video_processor', None) or getattr(processor, 'image_processor', None)
    temporal_patch_size = int(getattr(video_processor, 'temporal_patch_size', 0) or 0)
    if temporal_patch_size < 1:
        return encoder_inputs
    encoder_inputs['video_temporal_patch_size'] = video_grid_thw.new_full((video_grid_thw.shape[0],), temporal_patch_size)
    return encoder_inputs

def processor_qwen35_video_inputs(processor, messages: list[list[dict[str, Any]]], texts: str | list[str], device=None):
    from qwen_vl_utils import process_vision_info
    try:
        images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True, image_patch_size=16)
        video_metadata = None
        if videos is not None and videos and isinstance(videos[0], tuple):
            videos, video_metadata = map(list, zip(*videos))
        inputs = processor(text=texts, images=images, videos=videos, video_metadata=video_metadata, **video_kwargs, do_resize=False, padding=True, padding_side='left', return_metadata=True, return_tensors='pt')
    except TypeError:
        images, videos = process_vision_info(messages)
        inputs = processor(text=texts, images=images, videos=videos, padding=True, padding_side='left', return_tensors='pt')
    return inputs.to(device) if device is not None else inputs

def prepare_c2l_inputs(model, processor, batch: list[tuple[str, str, int, int]], args: argparse.Namespace):
    from malow.video.loader import remove_empty_think_block
    device = model_input_device(model)
    encoder_messages = [build_c2l_encoder_messages(video_path, start, end, args) for _prompt, video_path, start, end in batch]
    encoder_texts = remove_empty_think_block(apply_chat_template(processor, encoder_messages, args))
    interleave_timestamp = bool(getattr(model, 'interleave_timestamp', False))
    if args.base_input_video:
        encoder_frame_count = c2l_encoder_frame_count(args)
        if interleave_timestamp:
            frame_loader = load_base_last_2seconds_c2l_video_frames_with_timestamps if args.base_with_last_2seconds else load_split_c2l_video_frames_with_timestamps
        else:
            frame_loader = load_base_last_2seconds_c2l_video_frames if args.base_with_last_2seconds else load_split_c2l_video_frames
        tail_kwargs = {'tail_seconds': getattr(args, 'base_tail_seconds', 2.0)} if args.base_with_last_2seconds else {}
        split_videos = [frame_loader(video_path, encoder_frame_count, args.base_input_frame_number, clip_start_seconds=max(0.0, float(start)), clip_end_seconds=max(0.0, float(end)), **tail_kwargs) for _prompt, video_path, start, end in batch]
        encoder_videos = [item[0] for item in split_videos]
        base_videos = [item[1] for item in split_videos]
        encoder_timestamps = [item[2] for item in split_videos] if interleave_timestamp else None
        encoder_terminal_times = [item[4] for item in split_videos] if interleave_timestamp else None
        encoder_inputs = processor_predecoded_video_inputs(processor, encoder_texts, encoder_videos, args).to(device)
    else:
        encoder_inputs = processor_qwen35_video_inputs(processor, encoder_messages, encoder_texts)
        encoder_timestamps = None
        encoder_terminal_times = None
    annotate_video_temporal_patch_size(processor, encoder_inputs)
    if interleave_timestamp:
        add_c2l_timestamp_inputs(encoder_inputs, processor.tokenizer, getattr(model, 'frame_per_lora', 4), encoder_timestamps, encoder_terminal_times)
    encoder_inputs = encoder_inputs.to(device)
    base_messages = [build_c2l_base_messages(prompt, video_path, start, end, args) for prompt, video_path, start, end in batch]
    base_texts = remove_empty_think_block(apply_chat_template(processor, base_messages, args))
    if args.base_input_video:
        base_inputs = processor_predecoded_video_inputs(processor, base_texts, base_videos, args, branch="base").to(device)
    else:
        base_inputs = processor(text=base_texts, padding=True, padding_side='left', return_tensors='pt').to(device)
    return (encoder_inputs, base_inputs)

def generate_c2l_batch(model, processor, batch: list[tuple[str, str, int, int]], args: argparse.Namespace) -> list[str]:
    import torch
    encoder_inputs, base_inputs = prepare_c2l_inputs(model, processor, batch, args)
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    generate_kwargs: dict[str, Any] = {'max_new_tokens': args.max_new_tokens, 'eos_token_id': processor.tokenizer.eos_token_id, 'pad_token_id': pad_token_id, 'do_sample': args.temperature > 0}
    if args.temperature > 0:
        generate_kwargs['temperature'] = args.temperature
        if args.top_p is not None:
            generate_kwargs['top_p'] = args.top_p
        if args.top_k is not None:
            generate_kwargs['top_k'] = args.top_k
    with torch.inference_mode():
        generated = model.generate(ctx_ids=encoder_inputs, **base_inputs, **generate_kwargs)
    trimmed = [output[len(input_ids):] for input_ids, output in zip(base_inputs.input_ids, generated)]
    return [text.strip() for text in processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

def generate_batch(model, processor, batch: list[tuple[str, str, int, int]], args: argparse.Namespace) -> list[str]:
    import torch
    if args.c2l_checkpoint:
        return generate_c2l_batch(model, processor, batch, args)
    batched_messages = [build_messages(prompt, video_path, start, end, args) for prompt, video_path, start, end in batch]
    texts = apply_chat_template(processor, batched_messages, args)
    if args.no_video:
        inputs = processor(text=texts, padding=True, padding_side='left', return_tensors='pt')
    else:
        from qwen_vl_utils import process_vision_info
        if args.model_family == 'qwen2vl':
            images, videos = process_vision_info(batched_messages)
            inputs = processor(text=texts, images=images, videos=videos, padding=True, padding_side='left', return_tensors='pt')
        else:
            try:
                images, videos, video_kwargs = process_vision_info(batched_messages, return_video_kwargs=True, return_video_metadata=True, image_patch_size=16)
                video_metadata = None
                if videos is not None and videos and isinstance(videos[0], tuple):
                    videos, video_metadata = map(list, zip(*videos))
                inputs = processor(text=texts, images=images, videos=videos, video_metadata=video_metadata, **video_kwargs, do_resize=False, padding=True, padding_side='left', return_tensors='pt')
            except TypeError:
                images, videos = process_vision_info(batched_messages)
                inputs = processor(text=texts, images=images, videos=videos, padding=True, padding_side='left', return_tensors='pt')
    inputs = inputs.to(model_input_device(model))
    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    generate_kwargs: dict[str, Any] = {'max_new_tokens': args.max_new_tokens, 'eos_token_id': processor.tokenizer.eos_token_id, 'pad_token_id': pad_token_id, 'do_sample': args.temperature > 0}
    if args.temperature > 0:
        generate_kwargs['temperature'] = args.temperature
        if args.top_p is not None:
            generate_kwargs['top_p'] = args.top_p
        if args.top_k is not None:
            generate_kwargs['top_k'] = args.top_k
    with torch.inference_mode():
        generated = model.generate(**inputs, **generate_kwargs)
    trimmed = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, generated)]
    return [text.strip() for text in processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

def load_model(args: argparse.Namespace, local_rank: int, world_size: int):
    import torch
    from transformers import AutoProcessor
    if args.c2l_checkpoint:
        from malow.video.loader import load_c2l_model
        model = load_c2l_model(args, local_rank, world_size)
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
        return (model, processor)
    dtype = args.dtype
    model_kwargs: dict[str, Any] = {'device_map': resolve_device_map(args.device_map, local_rank, world_size), 'trust_remote_code': args.trust_remote_code}
    if dtype != 'auto':
        model_kwargs['torch_dtype'] = getattr(torch, dtype)
    family = args.model_family
    if family == 'auto':
        model_text = str(args.model_id).lower()
        family = 'qwen2vl' if 'qwen2-vl' in model_text else ('qwen3vl' if 'qwen3-vl' in model_text else 'qwen35')
    if family == 'qwen2vl':
        try:
            from transformers import Qwen2VLForConditionalGeneration
            model_cls = Qwen2VLForConditionalGeneration
        except ImportError:
            from transformers import AutoModelForImageTextToText
            model_cls = AutoModelForImageTextToText
    else:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    try:
        model = model_cls.from_pretrained(args.model_id, **model_kwargs).eval()
    except TypeError:
        if 'torch_dtype' in model_kwargs:
            model_kwargs['dtype'] = model_kwargs.pop('torch_dtype')
        model = model_cls.from_pretrained(args.model_id, **model_kwargs).eval()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=args.trust_remote_code)
    return (model, processor)

def prepare_c2l_base_inputs_single(
    model,
    processor,
    prompt: str,
    video_path: str,
    start: int | float,
    end: int | float,
    args: argparse.Namespace,
    summary_prefix: str | None = None,
):
    """Base-model inputs for one interactive turn, matching the offline base branch."""
    from malow.video.loader import remove_empty_think_block

    device = model_input_device(model)
    base_messages = [build_c2l_base_messages(prompt, video_path, start, end, args)]
    if summary_prefix:
        from malow.video.text_summary import prepend_memory
        prepend_memory(base_messages[0], summary_prefix)
    base_texts = apply_chat_template(processor, base_messages, args)
    if not getattr(args, "text_cache_base_only", False):
        base_texts = remove_empty_think_block(base_texts)
    if not args.base_input_video:
        return processor(
            text=base_texts, padding=True, padding_side="left", return_tensors="pt"
        ).to(device)

    clip_start = max(0.0, float(start))
    clip_end = max(clip_start, float(end))
    if args.base_with_last_2seconds:
        base_frames, base_times = load_c2l_base_tail_frames(
            video_path,
            args.base_input_frame_number,
            clip_start_seconds=clip_start,
            clip_end_seconds=clip_end,
            tail_seconds=args.base_tail_seconds,
        )
    else:
        _encoder_frames, base_frames, _encoder_times, base_times, _terminal_time = (
            load_split_c2l_video_frames_with_timestamps(
                video_path,
                c2l_encoder_frame_count(args),
                args.base_input_frame_number,
                clip_start_seconds=clip_start,
                clip_end_seconds=clip_end,
            )
        )
    return processor_predecoded_video_inputs(
        processor,
        base_texts,
        [base_frames],
        args,
        metadata_fps=args.base_input_frame_number / args.base_tail_seconds,
        timestamps=[base_times],
        branch="base",
    ).to(device)

def generate_with_loras_single(model, processor, base_inputs, generated_loras, args: argparse.Namespace) -> str:
    import torch

    pad_token_id = processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id
    generate_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": processor.tokenizer.eos_token_id,
        "pad_token_id": pad_token_id,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generate_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            generate_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            generate_kwargs["top_k"] = args.top_k
    with torch.inference_mode():
        generated = model.generate(generated_loras=generated_loras, **base_inputs, **generate_kwargs)
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(base_inputs.input_ids, generated)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

def require_interactive_support(model) -> None:
    supported, reason = model.supports_interactive_c2l()
    if not supported:
        raise ValueError(f"Interactive evaluation is not supported by this checkpoint: {reason}")

def interactive_answer(
    model,
    processor,
    session,
    encoder_text,
    video_path: str,
    prompt: str,
    base_start: int | float,
    question_time: int | float,
    args: argparse.Namespace,
    verify: bool = False,
) -> tuple[str, int, bool]:
    """Advance the session to ``question_time`` and answer. Returns (text, new blocks, fallback)."""
    args._resolution_video_path = video_path
    question_time = clamp_question_time(video_path, float(question_time))
    device = model_input_device(model)
    new_blocks = advance_session(
        model,
        processor,
        session,
        encoder_text,
        video_path,
        float(question_time),
        args,
        annotate_video_temporal_patch_size,
        device=device,
    )
    summary_kwargs = {}
    if getattr(args, "text_summary", False):
        from malow.video.text_summary import answer_prefix, trace_answer
        summary_kwargs["summary_prefix"] = answer_prefix(session, question_time)
        trace_answer(args, session, question_time, prompt)
    answer_key = None
    if getattr(args, "checkpoint_progress", False):
        answer_key = {"now": question_time, "start": base_start, "prompt": prompt,
                      "video": file_identity(video_path),
                      "memory": digest(getattr(getattr(session, "text_summary", None), "memory", None))}
        cached = cache_get(args, session, "answer", answer_key)
        if cached is not None:
            if verify and new_blocks and session.has_context():
                verify_session_against_single_pass(model, processor, session, encoder_text, video_path,
                                                  args, annotate_video_temporal_patch_size, device=device)
            if cached["fallback"]:
                session.fallback_turns += 1
            return cached["prediction"], new_blocks, cached["fallback"]
    if getattr(args, "text_cache_base_only", False):
        from malow.video.text_summary import generate_base_answer
        inputs = prepare_c2l_base_inputs_single(model, processor, prompt, video_path,
                                               base_start, question_time, args, **summary_kwargs)
        prediction = generate_base_answer(model, processor, inputs, args)
        fallback = not session.has_context()
        session.fallback_turns += int(fallback)
        cache_put(args, session, "answer", answer_key, {"prediction": prediction, "fallback": fallback})
        return prediction, new_blocks, fallback
    if not session.has_context():
        # Not even one temporal block of context yet: fall back to the offline single pass
        # over the short prefix, without committing anything to the streaming state.
        prediction = generate_c2l_batch(model, processor, [(prompt, video_path, base_start, question_time)], args)[0]
        session.fallback_turns += 1
        cache_put(args, session, "answer", answer_key, {"prediction": prediction, "fallback": True})
        return prediction, new_blocks, True

    if verify and new_blocks:
        verify_session_against_single_pass(
            model,
            processor,
            session,
            encoder_text,
            video_path,
            args,
            annotate_video_temporal_patch_size,
            device=device,
        )
    base_inputs = prepare_c2l_base_inputs_single(
        model, processor, prompt, video_path, base_start, question_time, args, **summary_kwargs
    )
    prediction = generate_with_loras_single(
        model, processor, base_inputs, session_generated_loras(model, session), args
    )
    cache_put(args, session, "answer", answer_key, {"prediction": prediction, "fallback": False})
    return prediction, new_blocks, False

def evaluate_mc_task_interactive(
    task: str,
    entries: list[tuple[int, dict[str, Any]]],
    annotation_file: Path,
    model,
    processor,
    args: argparse.Namespace,
    rank_path: Path,
    rank: int,
    world_size: int,
) -> list[Any]:
    """One streaming session per video; questions answered in video-clock order."""
    from malow.video.loader import remove_empty_think_block

    require_interactive_support(model)
    config = interactive_config_from_args(model, args)
    if rank == 0:
        print(f"task={task} {describe_streaming_path(model)}")
    journal = streaming_journal(args, rank_path, entries, task)
    done = set(journal.rows) if journal is not None else done_ids(rank_path)
    def append_result(path, rows):
        if journal is None:
            append_rows(path, rows)
        else:
            save_session(args, session)
            for row in rows:
                journal.append(row)
    total, completed = count_questions(entries, task, done)
    progress = make_progress_bar(args, task, rank, world_size, total, completed)
    tracked: list[Any] = []
    try:
        for session_index, (entry_idx, entry) in enumerate(entries):
            video_path = absolutize_video_path(entry["video_path"], annotation_file)
            validate_video_path(video_path)
            questions = entry_questions(entry)
            recovery_key = [task, video_path] if getattr(args, "streaming_group_by_video", False) else [task, entry_idx, video_path]
            if all(question_id(task, entry_idx, i, q) in done for i, q in enumerate(questions)):
                if journal is not None:
                    tracked.append(saved_session(args, config, recovery_key))
                continue

            # Build the SQA running context in the annotation's own order so the prompts stay
            # identical to offline, then answer in video-clock order so the stream is causal.
            contexts: list[str | None] = []
            context = "" if task == "sqa" else None
            for question in questions:
                contexts.append(context)
                if task == "sqa":
                    if not context:
                        context = "Here is the contextual information related to the video. "
                    opts = format_options(list(question.get("options", [])))
                    context += (
                        f"At timestamp {question['time_stamp']}, the following question and answer occurred: "
                        f"Question: {question.get('question')}; Options: {', '.join(opts)}; "
                        f"Answer: {question.get('answer')}. "
                    )

            contexts = [q.get("_source_context", contexts[i]) for i, q in enumerate(questions)]
            order = sorted(
                range(len(questions)),
                key=lambda index: (interactive_timestamp(questions[index]["time_stamp"]), index),
            )
            session = new_session(model, config, origin=0.0)
            bind_session(args, session, recovery_key)
            tracked.append(session)
            verify = select_verification_session(args)
            encoder_text = remove_empty_think_block(
                apply_chat_template(processor, [build_c2l_encoder_messages(video_path, 0, 0, args)], args)
            )

            for turn_index, question_idx in enumerate(order):
                question = questions[question_idx]
                question_context = contexts[question_idx]
                entry_idx = question.get("_source_entry_index", entry_idx)
                question_idx = question.get("_source_question_index", question_idx)
                qid = question_id(task, entry_idx, question_idx, question)
                timestamp = interactive_timestamp(question["time_stamp"])
                start = max(0, timestamp - args.context_time) if args.context_time > 0 else 0
                prompt, options = make_mc_prompt(question, question_context)
                if qid in done:
                    # Still advance the stream so later questions see the same prefix.
                    advance_session(
                        model,
                        processor,
                        session,
                        encoder_text,
                        video_path,
                        float(timestamp),
                        args,
                        annotate_video_temporal_patch_size,
                        device=model_input_device(model),
                    )
                    session.turns += 1
                    if not session.has_context():
                        session.fallback_turns += 1
                    continue

                prediction, new_blocks, fallback = interactive_answer(
                    model,
                    processor,
                    session,
                    encoder_text,
                    video_path,
                    prompt,
                    start,
                    timestamp,
                    args,
                    verify=verify,
                )
                session.turns += 1
                pred_choice = infer_option(prediction, options)
                answer = str(question.get("answer", "")).strip().upper()[:1]
                append_result(
                    rank_path,
                    [
                        {
                            "question_id": qid,
                            "task": task,
                            "entry_index": entry_idx,
                            "question_index": question_idx,
                            "task_type": question.get("task_type"),
                            "required_ability": question.get("required_ability"),
                            "time_stamp": question.get("time_stamp"),
                            "answer": answer,
                            "question": question.get("question"),
                            "options": options,
                            "video_path": video_path,
                            "video_start": start,
                            "video_end": timestamp,
                            "prompt": prompt,
                            "prediction": prediction,
                            "pred_choice": pred_choice,
                            "hit": int(pred_choice == answer) if answer else None,
                            "rank": rank,
                            "interactive": {
                                "session_index": session_index,
                                "turn_index": turn_index,
                                "blocks_consumed": session.consumed_blocks,
                                "new_blocks": new_blocks,
                                "segments": session.segments,
                                "offline_fallback": fallback,
                            },
                        }
                    ],
                )
                if progress is not None:
                    progress.update(1)
            finish_session(session)
    finally:
        if progress is not None:
            progress.close()
    if journal is not None:
        journal.require_complete()
    return tracked

def evaluate_proactive_task_interactive(
    entries: list[tuple[int, dict[str, Any]]],
    annotation_file: Path,
    model,
    processor,
    args: argparse.Namespace,
    rank_path: Path,
    rank: int,
    world_size: int,
) -> list[Any]:
    """One streaming session per proactive question; each poll advances it by one step."""
    from malow.video.loader import remove_empty_think_block

    require_interactive_support(model)
    config = interactive_config_from_args(model, args)
    if rank == 0:
        print(f"task=proactive {describe_streaming_path(model)}")
    journal = streaming_journal(args, rank_path, entries, "proactive")
    done = set(journal.rows) if journal is not None else done_ids(rank_path)
    def append_result(path, rows):
        if journal is None:
            append_rows(path, rows)
        else:
            save_session(args, session)
            for row in rows:
                journal.append(row)
    total, completed = count_questions(entries, "proactive", done)
    progress = make_progress_bar(args, "proactive", rank, world_size, total, completed)
    tracked: list[Any] = []
    session_index = 0
    try:
        for entry_idx, entry in entries:
            video_path = absolutize_video_path(entry["video_path"], annotation_file)
            validate_video_path(video_path)
            for question_idx, question in enumerate(entry_questions(entry)):
                qid = question_id("proactive", entry_idx, question_idx, question)
                recovery_key = ["proactive", qid, video_path]
                if qid in done:
                    if journal is not None:
                        tracked.append(saved_session(args, config, recovery_key))
                    continue
                start = parse_timestamp(question["time_stamp"])
                gt_time = parse_timestamp(question["ground_truth_time_stamp"])
                end_limit = gt_time + args.proactive_extra_seconds
                query = (
                    f"{question['question']} Is it the right time to output "
                    f"\"{question['ground_truth_output']}\"?"
                )
                yesno_prompt = PROMPT_PROACTIVE_YESNO.format(question=query)
                history = []
                answered_time = None
                final_prediction = ""

                # The proactive context window starts at the question, not at the video start.
                session = new_session(model, config, origin=float(start))
                bind_session(args, session, recovery_key)
                tracked.append(session)
                verify = select_verification_session(args)
                session_index += 1
                encoder_text = remove_empty_think_block(
                    apply_chat_template(processor, [build_c2l_encoder_messages(video_path, start, start, args)], args)
                )

                current = start + args.proactive_step
                while current <= end_limit:
                    if not can_decode_video_time_range(
                        video_path,
                        start,
                        current,
                        min_source_frames=min_decodable_source_frames(args),
                    ):
                        current += args.proactive_step
                        continue
                    t0 = time.time()
                    response, _new_blocks, _fallback = interactive_answer(
                        model,
                        processor,
                        session,
                        encoder_text,
                        video_path,
                        yesno_prompt,
                        start,
                        current,
                        args,
                        verify=verify,
                    )
                    session.turns += 1
                    cost = time.time() - t0
                    history.append({"role": "user", "content": query, "time": current, "cost": cost})
                    history.append({"role": "assistant", "content": response, "time": current, "cost": cost})
                    if "yes" in response.strip().lower():
                        final_prompt = PROMPT_PROACTIVE_FINAL.format(question=question["question"])
                        t0 = time.time()
                        # Same moment, so the stream does not advance; only the prompt changes.
                        final_prediction, _blocks, _fb = interactive_answer(
                            model,
                            processor,
                            session,
                            encoder_text,
                            video_path,
                            final_prompt,
                            start,
                            current,
                            args,
                        )
                        cost = time.time() - t0
                        history.append({"role": "user", "content": question["question"], "time": current, "cost": cost})
                        history.append({"role": "assistant", "content": final_prediction, "time": current, "cost": cost})
                        answered_time = current
                        break
                    current += args.proactive_step

                trigger_hit = answered_time is not None and gt_time <= answered_time <= end_limit
                expected = str(question.get("ground_truth_output", "")).strip().lower()
                output_hit = bool(expected) and expected in final_prediction.strip().lower()
                append_result(
                    rank_path,
                    [
                        {
                            "question_id": qid,
                            "task": "proactive",
                            "entry_index": entry_idx,
                            "question_index": question_idx,
                            "task_type": question.get("task_type"),
                            "required_ability": question.get("required_ability"),
                            "time_stamp": question.get("time_stamp"),
                            "ground_truth_time_stamp": question.get("ground_truth_time_stamp"),
                            "ground_truth_output": question.get("ground_truth_output"),
                            "question": question.get("question"),
                            "video_path": video_path,
                            "video_start": start,
                            "video_end": answered_time or end_limit,
                            "answered_time": answered_time,
                            "prediction": final_prediction,
                            "dialog_history": history,
                            "trigger_hit": int(trigger_hit),
                            "output_hit": int(output_hit),
                            "hit": int(trigger_hit and output_hit),
                            "rank": rank,
                            "interactive": {
                                "polls": session.turns,
                                "blocks_consumed": session.consumed_blocks,
                                "segments": session.segments,
                                "offline_fallback_polls": session.fallback_turns,
                            },
                        }
                    ],
                )
                if progress is not None:
                    progress.update(1)
                finish_session(session)
    finally:
        if progress is not None:
            progress.close()
    if journal is not None:
        journal.require_complete()
    return tracked

def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                ids.add(json.loads(line)['question_id'])
    return ids

def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
        handle.flush()

def make_mc_prompt(question: dict[str, Any], sqa_context: str | None=None) -> tuple[str, list[str]]:
    options = format_options(list(question.get('options', [])))
    template = PROMPT_SQA if sqa_context is not None else PROMPT_MC
    prompt = template.format(context=sqa_context or '', question=question.get('question', ''), options='\n'.join(options))
    return (prompt, options)

def evaluate_mc_task(task: str, entries: list[tuple[int, dict[str, Any]]], annotation_file: Path, model, processor, args: argparse.Namespace, rank_path: Path, rank: int, world_size: int) -> None:
    done = done_ids(rank_path)
    total, completed = count_questions(entries, task, done)
    progress = make_progress_bar(args, task, rank, world_size, total, completed)
    pending: list[tuple[dict[str, Any], tuple[str, str, int, int], list[str]]] = []
    try:
        for entry_idx, entry in entries:
            video_path = absolutize_video_path(entry['video_path'], annotation_file)
            if not args.no_video:
                validate_video_path(video_path)
            context = '' if task == 'sqa' else None
            for question_idx, question in enumerate(entry_questions(entry)):
                qid = question_id(task, entry_idx, question_idx, question)
                timestamp = parse_timestamp(question['time_stamp'])
                start = max(0, timestamp - args.context_time) if args.context_time > 0 else 0
                if qid not in done:
                    prompt, options = make_mc_prompt(question, context)
                    pending.append(({'question_id': qid, 'task': task, 'entry_index': entry_idx, 'question_index': question_idx, 'task_type': question.get('task_type'), 'required_ability': question.get('required_ability'), 'time_stamp': question.get('time_stamp'), 'answer': str(question.get('answer', '')).strip().upper()[:1], 'question': question.get('question'), 'options': options, 'video_path': video_path, 'video_start': start, 'video_end': timestamp, 'prompt': prompt, 'rank': rank}, (prompt, video_path, start, timestamp), options))
                if task == 'sqa':
                    if not context:
                        context = 'Here is the contextual information related to the video. '
                    opts = format_options(list(question.get('options', [])))
                    context += f"At timestamp {question['time_stamp']}, the following question and answer occurred: Question: {question.get('question')}; Options: {', '.join(opts)}; Answer: {question.get('answer')}. "
                if len(pending) >= args.batch_size:
                    flush_mc_pending(pending, model, processor, args, rank_path, progress)
                    pending = []
        if pending:
            flush_mc_pending(pending, model, processor, args, rank_path, progress)
    finally:
        if progress is not None:
            progress.close()

def flush_mc_pending(pending, model, processor, args: argparse.Namespace, rank_path: Path, progress=None) -> None:
    outputs = generate_batch(model, processor, [item[1] for item in pending], args)
    rows = []
    for (row, _request, options), prediction in zip(pending, outputs, strict=False):
        pred_choice = infer_option(prediction, options)
        row['prediction'] = prediction
        row['pred_choice'] = pred_choice
        row['hit'] = int(pred_choice == row['answer']) if row['answer'] else None
        rows.append(row)
    append_rows(rank_path, rows)
    if progress is not None:
        progress.update(len(rows))

def evaluate_proactive_task(entries: list[tuple[int, dict[str, Any]]], annotation_file: Path, model, processor, args: argparse.Namespace, rank_path: Path, rank: int, world_size: int) -> None:
    done = done_ids(rank_path)
    total, completed = count_questions(entries, 'proactive', done)
    progress = make_progress_bar(args, 'proactive', rank, world_size, total, completed)
    try:
        for entry_idx, entry in entries:
            video_path = absolutize_video_path(entry['video_path'], annotation_file)
            if not args.no_video:
                validate_video_path(video_path)
            for question_idx, question in enumerate(entry_questions(entry)):
                qid = question_id('proactive', entry_idx, question_idx, question)
                if qid in done:
                    continue
                start = parse_timestamp(question['time_stamp'])
                gt_time = parse_timestamp(question['ground_truth_time_stamp'])
                end_limit = gt_time + args.proactive_extra_seconds
                query = f'''{question['question']} Is it the right time to output "{question['ground_truth_output']}"?'''
                yesno_prompt = PROMPT_PROACTIVE_YESNO.format(question=query)
                history = []
                answered_time = None
                final_prediction = ''
                current = start + args.proactive_step
                while current <= end_limit:
                    if not args.no_video and (not can_decode_video_time_range(video_path, start, current, min_source_frames=min_decodable_source_frames(args))):
                        current += args.proactive_step
                        continue
                    t0 = time.time()
                    response = generate_batch(model, processor, [(yesno_prompt, video_path, start, current)], args)[0]
                    cost = time.time() - t0
                    history.append({'role': 'user', 'content': query, 'time': current, 'cost': cost})
                    history.append({'role': 'assistant', 'content': response, 'time': current, 'cost': cost})
                    if 'yes' in response.strip().lower():
                        final_prompt = PROMPT_PROACTIVE_FINAL.format(question=question['question'])
                        t0 = time.time()
                        final_prediction = generate_batch(model, processor, [(final_prompt, video_path, start, current)], args)[0]
                        cost = time.time() - t0
                        history.append({'role': 'user', 'content': question['question'], 'time': current, 'cost': cost})
                        history.append({'role': 'assistant', 'content': final_prediction, 'time': current, 'cost': cost})
                        answered_time = current
                        break
                    current += args.proactive_step
                trigger_hit = answered_time is not None and gt_time <= answered_time <= end_limit
                expected = str(question.get('ground_truth_output', '')).strip().lower()
                output_hit = bool(expected) and expected in final_prediction.strip().lower()
                append_rows(rank_path, [{'question_id': qid, 'task': 'proactive', 'entry_index': entry_idx, 'question_index': question_idx, 'task_type': question.get('task_type'), 'required_ability': question.get('required_ability'), 'time_stamp': question.get('time_stamp'), 'ground_truth_time_stamp': question.get('ground_truth_time_stamp'), 'ground_truth_output': question.get('ground_truth_output'), 'question': question.get('question'), 'video_path': video_path, 'video_start': start, 'video_end': answered_time or end_limit, 'answered_time': answered_time, 'prediction': final_prediction, 'dialog_history': history, 'trigger_hit': int(trigger_hit), 'output_hit': int(output_hit), 'hit': int(trigger_hit and output_hit), 'rank': rank}])
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def merge_task_outputs(task_dir: Path, world_size: int) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for rank in range(world_size):
        for row in read_jsonl(task_dir / f'rank_{rank:05d}.jsonl'):
            if row['question_id'] not in seen:
                merged.append(row)
                seen.add(row['question_id'])
    merged.sort(key=lambda item: (item.get('entry_index', 0), item.get('question_index', 0), item['question_id']))
    with (task_dir / 'predictions.jsonl').open('w', encoding='utf-8') as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    (task_dir / 'predictions.json').write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding='utf-8')
    return merged

def grouped_accuracy(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if row.get('hit') is not None and row.get(key) is not None:
            groups[str(row[key])].append(int(row['hit']))
    return {group: sum(values) / len(values) for group, values in sorted(groups.items()) if values}

def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [int(row['hit']) for row in rows if row.get('hit') is not None]
    acc = sum(hits) / len(hits) if hits else 0.0
    return {'num_examples': len(hits), 'correct': sum(hits), 'accuracy': acc, 'accuracy_percent': acc * 100.0, 'task_type_accuracy': grouped_accuracy(rows, 'task_type'), 'required_ability_accuracy': grouped_accuracy(rows, 'required_ability')}

def build_weighted_summary(summary: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if any((task not in summary for task in WEIGHTED_SUMMARY_WEIGHTS)):
        return None
    total_questions = sum(WEIGHTED_SUMMARY_WEIGHTS.values())
    accuracy = sum((summary[task]['accuracy'] * weight for task, weight in WEIGHTED_SUMMARY_WEIGHTS.items())) / total_questions
    return {'weights': dict(WEIGHTED_SUMMARY_WEIGHTS), 'total_questions': total_questions, 'accuracy': accuracy, 'accuracy_percent': accuracy * 100.0}

def main() -> None:
    args = parse_args()
    if args.use_streamingbench_frame_sampling:
        args.video_total_pixels = STREAMINGBENCH_VIDEO_TOTAL_PIXELS
        args.video_max_pixels = min(args.video_max_pixels, STREAMINGBENCH_VIDEO_MAX_PIXELS)
        args.video_min_pixels = min(args.video_min_pixels, args.video_max_pixels)
        args.max_frames = STREAMINGBENCH_MAX_FRAMES
        args.num_frames = None
        args.temperature = 0.0
        args.top_p = None
        args.top_k = None
        args.streamingbench_dynamic_fps = '<300s:1.0, 300-599s:0.5, >=600s:0.2'
    rank, local_rank, world_size = init_distributed()
    run_token = distributed_run_token()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_interactive_run(args, world_size)
    init_resolution_log(args, rank)
    if rank == 0:
        (output_dir / 'run_config.json').write_text(json.dumps({k: v for k, v in vars(args).items() if not k.startswith("_")}, indent=2), encoding='utf-8')
        print(f'StreamingBench output_dir={output_dir}')
        print(f'world_size={world_size}')
        print(f'rank barrier token={run_token}')
    task_files = resolve_task_files(args)
    if getattr(args, "checkpoint_progress", False):
        from malow.video.progress import guard_video_inputs
        paths = []
        for annotation_file in task_files.values():
            entries = load_json(annotation_file)
            if args.limit is not None:
                entries = entries[:args.limit]
            paths.extend(absolutize_video_path(entry["video_path"], annotation_file) for entry in entries)
        guard_video_inputs(args, rank, paths)
    model, processor = load_model(args, local_rank, world_size)
    for task, annotation_file in task_files.items():
        data = load_json(annotation_file)
        if args.limit is not None:
            data = data[:args.limit]
        task_dir = output_dir / task
        task_dir.mkdir(parents=True, exist_ok=True)
        rank_path = task_dir / f'rank_{rank:05d}.jsonl'
        if getattr(args, "streaming_group_by_video", False) and task != "proactive":
            grouped = group_video_entries(data, task, annotation_file)
            shard = [entry for i, entry in enumerate(grouped) if i % world_size == rank]
        else:
            shard = shard_entries(data, rank, world_size)
        if rank == 0:
            print(f'task={task} entries={len(data)} per-rank shard starts with {len(shard)} entries on rank0')
        if args.eval_mode != "offline":
            if task == "proactive":
                tracked = evaluate_proactive_task_interactive(shard, annotation_file, model, processor, args, rank_path, rank, world_size)
            else:
                tracked = evaluate_mc_task_interactive(task, shard, annotation_file, model, processor, args, rank_path, rank, world_size)
            write_interactive_report(args, rank, tracked, task)
        elif task == 'proactive':
            evaluate_proactive_task(shard, annotation_file, model, processor, args, rank_path, rank, world_size)
        else:
            evaluate_mc_task(task, shard, annotation_file, model, processor, args, rank_path, rank, world_size)
        barrier(task_dir, 'eval_done', rank, world_size, run_token, args.dist_barrier_timeout_seconds)
        if rank == 0:
            rows = merge_task_outputs(task_dir, world_size)
            stats = score(rows)
            (task_dir / 'stats.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"task={task} accuracy={stats['accuracy']:.4f} examples={stats['num_examples']}")
        barrier(task_dir, 'merge_done', rank, world_size, run_token, args.dist_barrier_timeout_seconds)
    if rank == 0:
        summary = {}
        for task in task_files:
            stats_path = output_dir / task / 'stats.json'
            if stats_path.exists():
                summary[task] = json.loads(stats_path.read_text(encoding='utf-8'))
        weighted_summary = build_weighted_summary(summary)
        if weighted_summary is not None:
            summary['weighted_average'] = weighted_summary
        (output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
if __name__ == '__main__':
    main()
