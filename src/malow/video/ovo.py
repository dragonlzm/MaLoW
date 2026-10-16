# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
"""Standalone Qwen3.5 / Qwen3-VL Video C2L evaluation for OVO-Bench.

This mirrors the lmms-eval OVO-Bench task prompts and Qwen3.5 video defaults
used by:

FORCE_QWENVL_VIDEO_READER=decord accelerate launch ... -m lmms_eval   --model qwen3_5   --model_args pretrained=Qwen/Qwen3.5-0.8B,max_pixels=2097152,min_pixels=262144,max_num_frames=64,enable_thinking=false   --gen_kwargs max_new_tokens=128,temperature=0,num_beams=1   --tasks ovobench
"""
from __future__ import annotations
import argparse
import copy
import glob
import json
import math
import os
if cache_root := os.environ.get('MALOW_TRITON_CACHE_ROOT'):
    os.environ['TRITON_CACHE_DIR'] = os.path.join(cache_root, f"rank-{os.environ.get('LOCAL_RANK', '0')}")
import re
import shutil
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable
from malow.video.sampling import add_c2l_timestamp_inputs, c2l_encoder_frame_count, load_base_last_2seconds_c2l_video_frames, load_base_last_2seconds_c2l_video_frames_with_timestamps, load_split_c2l_video_frames, load_split_c2l_video_frames_with_timestamps, processor_predecoded_video_inputs
from malow.video.interactive_options import (
    add_interactive_arguments, resolve_interactive_arguments, guard_interactive_run,
    init_resolution_log, write_interactive_report,
)
from malow.video.interactive import (
    advance_session, new_session, interactive_config_from_args, session_generated_loras,
    verify_session_against_single_pass, describe_streaming_path, finish_session,
    clamp_question_time,
)
from malow.video.sampling import load_c2l_base_tail_frames
DATASET_REPO_ID = 'JoeLeelyf/OVO-Bench'
ANNOTATION_REPO_ID = 'kcz358/OVOBench'
HF_HOME = os.path.expanduser(os.getenv('HF_HOME', '~/.cache/huggingface'))
OVO_ROOT = os.path.join(HF_HOME, 'ovobench')
CHUNKS_DIR = os.path.join(OVO_ROOT, 'chunked_videos')
LOCK_PATH = os.path.join(OVO_ROOT, '.ovobench.lock')
OPTION_LETTERS = tuple('ABCD')
TASK_ALIASES = {'ovobench': ('ovo_backward', 'ovo_realtime', 'ovo_forward'), 'all': ('ovo_backward', 'ovo_realtime', 'ovo_forward'), 'ovo_backward': ('ovo_backward',), 'backward': ('ovo_backward',), 'ovo_realtime': ('ovo_realtime',), 'realtime': ('ovo_realtime',), 'ovo_forward': ('ovo_forward',), 'forward': ('ovo_forward',)}
ANNOTATION_SPLITS = {'ovo_backward': 'backward', 'ovo_realtime': 'realtime', 'ovo_forward': 'forward'}
BACKWARD_TASKS = {'EPM', 'ASI', 'HLD'}
REALTIME_TASKS = {'OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR'}
FORWARD_TASKS = {'REC', 'SSR', 'CRR'}
BR_PROMPT_TEMPLATE = '\nQuestion: {}\nOptions:\n{}\n\nRespond only with the letter corresponding to your chosen option (e.g., A, B, C). \nDo not include any additional text or explanation in your response.\n'
REC_PROMPT_TEMPLATE = "\nYou're watching a video in which people may perform a certain type of action repetively. \nThe person performing this kind of action are referred to as 'they' in the following statement.\nYou're task is to count how many times have different people in the video perform this kind of action in total.\nOne complete motion counts as one. \nNow, answer the following question: {}\nProvide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.\nDo not include any additional text or explanation in your response.\n"
SSR_PROMPT_TEMPLATE = "\nYou're watching a tutorial video which contain a sequential of steps. \nThe following is one step from the whole procedures: \n{}\nYour task is to determine if the man or woman in the video is currently performing this step.\nAnswer only with “Yes” or “No”.\nDo not include any additional text or explanation in your response.\n"
CRR_PROMPT_TEMPLATE = "\nYou're responsible of answering questions based on the video content. \nThe following question are relevant to the latest frames, i.e. the end of the video.\n{}\nDecide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.\nAnswer only with “Yes” or “No”.\nDo not include any additional text or explanation in your response.\n"

@dataclass(frozen=True)
class EvalUnit:
    task_name: str
    doc: dict[str, Any]
    prompt: str
    video_path: str | None
    round_idx: int | None

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluate Qwen3.5 or Qwen3-VL Video C2L checkpoints on OVO-Bench.')
    parser.add_argument('--tasks', default='ovobench', help='Comma-separated tasks: ovobench, ovo_backward, ovo_realtime, ovo_forward.')
    parser.add_argument('--model-id', default='Qwen/Qwen3.5-0.8B', help='Base Hugging Face model id or local path.')
    parser.add_argument('--c2l-checkpoint', default=None, help='Optional Video C2L checkpoint directory or state file.')
    parser.add_argument('--output-dir', default='outputs/ovo/qwen35_0_8b_ovobench', help='Output directory.')
    parser.add_argument('--video-root', default=None, help='Pre-extracted OVO chunked_videos directory.')
    parser.add_argument('--annotation-root', default=None, help='Optional local OVO annotation root with backward/realtime/forward test parquet files.')
    parser.add_argument('--no-auto-download-videos', action='store_true', help='Disable automatic chunked video download.')
    parser.add_argument('--no-video', action='store_true', help='Evaluate with text prompts only, without video input.')
    parser.add_argument('--limit', type=int, default=None, help='Optional document limit per OVO subtask for smoke tests.')
    parser.add_argument('--batch-size', type=int, default=1, help='Generation batch size.')
    parser.add_argument('--dtype', default='bfloat16', help='Model dtype passed to Transformers.')
    parser.add_argument('--device-map', default='auto', help='Transformers device_map value.')
    parser.add_argument('--vision-encoder-only', action='store_true', help='Load only the projected Qwen3.5 or Qwen3-VL vision tower for the frozen C2L context encoder.')
    parser.add_argument('--video-min-pixels', type=int, default=262144, help='Qwen video min_pixels. Matches lmms-eval command.')
    parser.add_argument('--video-max-pixels', type=int, default=2097152, help='Qwen video max_pixels. Matches lmms-eval command.')
    parser.add_argument('--num-frames', type=int, default=64, help='Maximum video frame count. Matches lmms-eval max_num_frames=64.')
    parser.add_argument('--video-fps', type=float, default=None, help='Optional FPS sampling mode. Leave unset to match lmms-eval nframes.')
    parser.add_argument('--max-frames', type=int, default=64, help='Maximum frames when --video-fps is set.')
    parser.add_argument('--last-2second-1fps', action='store_true', help='For base-model evaluation, sample the last 2 seconds of each OVO chunk at 1 FPS. Chunks shorter than 2 seconds are sampled over the full chunk at 1 FPS.')
    parser.add_argument('--last-seconds-1fps', type=float, default=None, help='For base-model evaluation, sample the last N seconds of each OVO chunk at 1 FPS. Chunks shorter than N seconds are sampled over the full chunk at 1 FPS.')
    parser.add_argument('--video-total-pixels', type=int, default=0, help='Optional total_pixels budget. Default 0 matches the user command.')
    parser.add_argument('--max-new-tokens', type=int, default=128, help='Generation max_new_tokens. Matches lmms-eval.')
    parser.add_argument('--temperature', type=float, default=0.0, help='Generation temperature. 0 disables sampling.')
    parser.add_argument('--top-p', type=float, default=None, help='Optional nucleus sampling top-p value.')
    parser.add_argument('--top-k', type=int, default=None, help='Optional top-k sampling value.')
    parser.add_argument('--num-beams', type=int, default=1, help='Generation num_beams. Matches lmms-eval.')
    parser.add_argument('--system-prompt', default='You are a helpful assistant.', help='System prompt used by lmms-eval Qwen3.5.')
    parser.add_argument('--enable-thinking', choices=('true', 'false'), default='false', help='Qwen chat-template thinking mode.')
    parser.add_argument('--user-defined-scaling', type=float, default=1.0, help='Multiplier for generated C2L LoRA weights.')
    parser.add_argument('--block-lora', choices=('auto', 'true', 'false'), default='auto', help='Use block-LoRA generation. `auto` reads the value from C2L checkpoint metadata.')
    parser.add_argument('--frame-per-lora', type=int, default=None, help='Number of raw video frames per generated LoRA block. Defaults to checkpoint metadata or 4.')
    parser.add_argument('--micro-c2l-forward-frame', type=int, default=None, help='Optional raw-frame cap for each micro-batch when generating block-LoRA C2L weights. For frame_per_lora=2, 768 means at most 384 LoRA blocks per micro-forward.')
    parser.add_argument('--base-input-video', action='store_true', help='Include a tail video observation in C2L base inputs in addition to the text prompt.')
    parser.add_argument('--base-input-frame-number', type=int, default=4, help='Number of tail video frames to include in C2L base inputs when --base-input-video is set.')
    parser.add_argument('--base-with-last-2seconds', action='store_true', help='For Video C2L base video input, sample base frames from the tail window (default 2 seconds) and sample the earlier interval at 1 FPS up to the C2L encoder frame cap.')
    parser.add_argument('--trust-remote-code', action='store_true', help='Pass trust_remote_code=True to Transformers.')
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
    if args.last_2second_1fps and args.no_video:
        parser.error('--last-2second-1fps requires video input and cannot be combined with --no-video.')
    if args.last_seconds_1fps is not None and args.last_seconds_1fps <= 0:
        parser.error('--last-seconds-1fps must be positive.')
    if args.last_seconds_1fps is not None and args.no_video:
        parser.error('--last-seconds-1fps requires video input and cannot be combined with --no-video.')
    if args.last_2second_1fps and args.last_seconds_1fps is not None:
        parser.error('--last-2second-1fps cannot be combined with --last-seconds-1fps.')
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
    if world_size > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            # Workers only synchronize file writes; they never exchange GPU tensors.
            # NCCL lazily allocates CUDA resources at the final barrier, which can
            # fail after inference has filled the CUDA allocator's cache.
            torch.distributed.init_process_group(backend='gloo')
    return (rank, local_rank, world_size)

def is_main_process(rank: int) -> bool:
    return rank == 0

def resolve_device_map(device_map: str, local_rank: int, world_size: int):
    if world_size <= 1 or device_map != 'auto':
        return device_map
    import torch
    if torch.cuda.is_available():
        return {'': f'cuda:{local_rank}'}
    return {'': 'cpu'}

def model_input_device(model) -> Any:
    import torch
    base = getattr(model, 'base_model', model)
    device = getattr(base, 'device', None)
    if device is not None:
        return torch.device(device)
    return next(model.parameters()).device

def normalize_task_list(tasks: str) -> list[str]:
    normalized: list[str] = []
    for raw in tasks.split(','):
        key = raw.strip()
        if not key:
            continue
        if key not in TASK_ALIASES:
            raise ValueError(f'Unknown OVO-Bench task {key!r}. Expected one of {sorted(TASK_ALIASES)}.')
        normalized.extend(TASK_ALIASES[key])
    return list(dict.fromkeys(normalized))

def is_forward_task(doc: dict[str, Any]) -> bool:
    return str(doc['task']) in FORWARD_TASKS

def get_task_type(task_name: str) -> str:
    if task_name in BACKWARD_TASKS:
        return 'backward'
    if task_name in REALTIME_TASKS:
        return 'realtime'
    if task_name in FORWARD_TASKS:
        return 'forward'
    raise ValueError(f'Unknown OVO-Bench task code: {task_name}')

def _chunks_ready(path: str=CHUNKS_DIR) -> bool:
    return os.path.isdir(path) and any((name.endswith('.mp4') for name in os.listdir(path)))

def _concat_parts(parts: Iterable[str], out_path: str) -> None:
    with open(out_path, 'wb') as out:
        for part in parts:
            with open(part, 'rb') as handle:
                shutil.copyfileobj(handle, out, length=1 << 22)

def _safe_extract_tar(tar: tarfile.TarFile, path: str) -> None:
    root = Path(path).resolve()
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if root not in target.parents and target != root:
            raise RuntimeError(f'Refusing to extract tar member outside {root}: {member.name}')
    tar.extractall(path)

def _normalize_extracted_dir() -> None:
    if os.path.isdir(CHUNKS_DIR):
        return
    for name in os.listdir(OVO_ROOT):
        candidate = os.path.join(OVO_ROOT, name)
        if os.path.isdir(candidate) and _chunks_ready(candidate):
            os.rename(candidate, CHUNKS_DIR)
            return

def get_chunked_videos_dir() -> str:
    """Return the extracted OVO-Bench chunked video directory, downloading if needed."""
    from filelock import FileLock
    from huggingface_hub import snapshot_download
    if _chunks_ready():
        return CHUNKS_DIR
    os.makedirs(OVO_ROOT, exist_ok=True)
    with FileLock(LOCK_PATH):
        if _chunks_ready():
            return CHUNKS_DIR
        snapshot = snapshot_download(DATASET_REPO_ID, repo_type='dataset', allow_patterns=['chunked_videos.tar.parta*'], cache_dir=HF_HOME)
        parts = sorted(glob.glob(os.path.join(snapshot, 'chunked_videos.tar.parta*')))
        if not parts:
            raise RuntimeError(f'No chunked_videos.tar.parta* files found under {snapshot}')
        tar_path = os.path.join(OVO_ROOT, 'chunked_videos.tar')
        _concat_parts(parts, tar_path)
        with tarfile.open(tar_path) as tar:
            _safe_extract_tar(tar, OVO_ROOT)
        os.remove(tar_path)
        _normalize_extracted_dir()
    if not _chunks_ready():
        raise RuntimeError(f'OVO-Bench video extraction failed: {CHUNKS_DIR} is missing or empty.')
    return CHUNKS_DIR

def resolve_video_root(video_root: str | None, auto_download: bool) -> str:
    if video_root:
        root = str(Path(video_root).expanduser())
        if _chunks_ready(root):
            return root
        nested_root = os.path.join(root, 'chunked_videos')
        if _chunks_ready(nested_root):
            return nested_root
        raise FileNotFoundError(f'--video-root does not contain OVO chunk mp4 files: {root}')
    if auto_download:
        return get_chunked_videos_dir()
    raise FileNotFoundError('OVO-Bench chunked videos were not provided. Set --video-root or omit --no-auto-download-videos.')

def decode_forward_fields(doc: dict[str, Any]) -> dict[str, Any]:
    if 'test_info' in doc and isinstance(doc['test_info'], list):
        doc['test_info'] = [json.loads(item) if isinstance(item, str) else item for item in doc['test_info']]
    return doc

def ovobench_features(task_name: str):
    import datasets
    if task_name in {'ovo_backward', 'ovo_realtime'}:
        return datasets.Features({'id': datasets.Value('int64'), 'task': datasets.Value('string'), 'video': datasets.Value('string'), 'realtime': datasets.Value('float64'), 'question': datasets.Value('string'), 'answer': datasets.Value('string'), 'options': datasets.Sequence(datasets.Value('string')), 'gt': datasets.Value('int64')})
    return datasets.Features({'id': datasets.Value('int64'), 'task': datasets.Value('string'), 'video': datasets.Value('string'), 'question': datasets.Value('string'), 'answer': datasets.Value('string'), 'ask_time': datasets.Value('int64'), 'clue_time': datasets.Value('int64'), 'test_info': datasets.Sequence(datasets.Value('large_string')), 'tutorial': datasets.Value('string'), 'all_steps': datasets.Sequence(datasets.Value('string')), 'start_time': datasets.Sequence(datasets.Value('int64')), 'end_time': datasets.Sequence(datasets.Value('int64')), 'activity': datasets.Value('string'), 'start_times': datasets.Sequence(datasets.Value('int64')), 'end_times': datasets.Sequence(datasets.Value('int64'))})

def resolve_annotation_files(task_name: str, annotation_root: str | None) -> str | list[str]:
    split_name = ANNOTATION_SPLITS[task_name]
    if annotation_root:
        root = Path(annotation_root).expanduser()
        candidates = sorted((root / split_name).glob('test-*.parquet'))
        if not candidates:
            candidates = sorted(root.glob(f'{split_name}/test-*.parquet'))
        if not candidates:
            raise FileNotFoundError(f'No OVO-Bench annotation parquet files found under {root / split_name}')
        return str(candidates[0]) if len(candidates) == 1 else [str(path) for path in candidates]
    return f'hf://datasets/{ANNOTATION_REPO_ID}/{split_name}/test-*.parquet'

def load_ovobench_task(task_name: str, limit: int | None, annotation_root: str | None=None) -> list[dict[str, Any]]:
    from datasets import load_dataset
    data_file = resolve_annotation_files(task_name, annotation_root)
    dataset = load_dataset('parquet', data_files={'test': data_file}, features=ovobench_features(task_name), split='test')
    rows = [decode_forward_fields(dict(doc)) for doc in dataset]
    if limit is not None:
        rows = rows[:limit]
    return rows

def build_prompt(doc: dict[str, Any], index: int | None=None) -> str:
    task = str(doc['task'])
    if not is_forward_task(doc):
        options = doc['options']
        formatted_options = '; '.join((f'{chr(65 + i)}. {option}' for i, option in enumerate(options))) + ';'
        return BR_PROMPT_TEMPLATE.format(doc['question'], formatted_options)
    if task == 'REC':
        question = 'How many times did they ' + str(doc['activity']) + '?'
        return REC_PROMPT_TEMPLATE.format(question)
    if task == 'SSR':
        if index is None:
            raise ValueError('SSR prompt requires a forward round index.')
        return SSR_PROMPT_TEMPLATE.format(doc['test_info'][index]['step'])
    if task == 'CRR':
        return CRR_PROMPT_TEMPLATE.format(doc['question'])
    raise ValueError(f'Unknown forward task: {task}')

def resolve_video_path(doc: dict[str, Any], video_root: str, round_idx: int | None=None) -> str:
    if round_idx is not None:
        path = Path(video_root) / f"{doc['id']}_{round_idx}.mp4"
    elif is_forward_task(doc):
        path = Path(video_root) / f"{doc['id']}_0.mp4"
    else:
        path = Path(video_root) / f"{doc['id']}.mp4"
    if not path.exists():
        raise FileNotFoundError(f'Missing OVO-Bench video chunk: {path}. Run the data preparation step in evaluation/OVO_Bench/readme.md or set --video-root.')
    return str(path)

def build_eval_units(task_names: list[str], video_root: str | None, limit: int | None, no_video: bool=False, annotation_root: str | None=None) -> list[EvalUnit]:
    units: list[EvalUnit] = []
    for task_name in task_names:
        for doc in load_ovobench_task(task_name, limit, annotation_root):
            if task_name == 'ovo_forward':
                for round_idx in range(len(doc['test_info'])):
                    video_path = None if no_video else resolve_video_path(doc, str(video_root), round_idx)
                    units.append(EvalUnit(task_name=task_name, doc=doc, prompt=build_prompt(doc, round_idx), video_path=video_path, round_idx=round_idx))
            else:
                video_path = None if no_video else resolve_video_path(doc, str(video_root))
                units.append(EvalUnit(task_name=task_name, doc=doc, prompt=build_prompt(doc), video_path=video_path, round_idx=None))
    return units

def unit_question_time(unit: EvalUnit) -> float:
    """Absolute end of the context available to this question, in seconds."""
    if unit.video_path is None:
        raise ValueError("Interactive OVO evaluation requires video input.")
    duration = video_duration_seconds(unit.video_path)
    if duration is None:
        raise RuntimeError(f"Cannot determine the duration of OVO chunk {unit.video_path}.")
    return float(duration)

def annotated_question_time(unit: EvalUnit) -> float | None:
    """The annotation's own timestamp, kept only for the output rows."""
    doc = unit.doc
    if unit.round_idx is not None:
        rounds = doc.get("test_info") or []
        if unit.round_idx < len(rounds) and isinstance(rounds[unit.round_idx], dict):
            value = rounds[unit.round_idx].get("realtime")
            return float(value) if value is not None else None
        return None
    value = doc.get("realtime")
    return float(value) if value is not None else None

def interactive_session_key(unit: EvalUnit, args: argparse.Namespace) -> tuple:
    doc = unit.doc
    if getattr(args, "interactive_session_key", "video") == "doc":
        return (unit.task_name, str(doc.get("video")), str(doc.get("id")))
    return (unit.task_name, str(doc.get("video")))

def build_interactive_sessions(units: list[EvalUnit], args: argparse.Namespace) -> list[list[EvalUnit]]:
    """Group questions by source video and order each group along the video clock."""
    groups: dict[tuple, list[EvalUnit]] = defaultdict(list)
    for unit in units:
        groups[interactive_session_key(unit, args)].append(unit)
    sessions = []
    for key in sorted(groups, key=lambda item: tuple(str(part) for part in item)):
        sessions.append(
            sorted(
                groups[key],
                key=lambda unit: (
                    unit_question_time(unit),
                    int(unit.doc.get("id") or 0),
                    unit.round_idx if unit.round_idx is not None else -1,
                ),
            )
        )
    return sessions

def shard_sessions(sessions: list[list[EvalUnit]], rank: int, world_size: int) -> list[list[EvalUnit]]:
    """Whole sessions per rank: a session's questions must share one streaming state."""
    if world_size <= 1:
        return sessions
    return [session for index, session in enumerate(sessions) if index % world_size == rank]

def prepare_c2l_base_inputs(model, processor, unit: EvalUnit, args: argparse.Namespace):
    """Base-model inputs for one question, matching the offline base branch."""
    from malow.video.loader import remove_empty_think_block

    if args.base_input_video:
        base_messages = [build_video_text_messages(unit, args)]
    else:
        base_messages = [build_text_only_messages(unit.prompt, args)]
    base_texts = remove_empty_think_block(apply_chat_template(processor, base_messages, args))
    if not args.base_input_video:
        return processor(text=base_texts, padding=True, padding_side="left", return_tensors="pt")

    if args.base_with_last_2seconds:
        base_frames, base_times = load_c2l_base_tail_frames(
            str(unit.video_path),
            args.base_input_frame_number,
            tail_seconds=args.base_tail_seconds,
        )
    else:
        _encoder_frames, base_frames, _encoder_times, base_times, _terminal_time = (
            load_split_c2l_video_frames_with_timestamps(
                str(unit.video_path),
                c2l_encoder_frame_count(args),
                args.base_input_frame_number,
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
    )

def generate_with_loras(model, processor, base_inputs, generated_loras, args: argparse.Namespace) -> str:
    import torch

    device = model_input_device(model)
    base_inputs = base_inputs.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(
            generated_loras=generated_loras,
            **base_inputs,
            **build_generate_kwargs(processor, args),
        )
    return decode_generated_outputs(processor, base_inputs.input_ids, generated_ids, args)[0]

def run_interactive_sessions(model, processor, sessions: list[list[EvalUnit]], args: argparse.Namespace, rank: int):
    """Answer every question by streaming its video once, question by question."""
    from malow.video.loader import remove_empty_think_block

    supported, reason = model.supports_interactive_c2l()
    if not supported:
        raise ValueError(f"Interactive evaluation is not supported by this checkpoint: {reason}")

    config = interactive_config_from_args(model, args)
    if is_main_process(rank):
        print(f"Rank {rank} {describe_streaming_path(model)}")
    device = model_input_device(model)
    rows: list[dict[str, Any]] = []
    tracked: list[Any] = []

    from tqdm import tqdm

    iterator = tqdm(sessions, desc=f"Rank {rank} OVO-Bench interactive", disable=not is_main_process(rank))
    for session_index, session_units in enumerate(iterator):
        session = new_session(model, config)
        verify = session_index < args.interactive_verify_sessions
        tracked.append(session)
        for turn_index, unit in enumerate(session_units):
            args._resolution_video_path = unit.video_path
            question_time = unit_question_time(unit)
            encoder_text = remove_empty_think_block(
                apply_chat_template(processor, [build_video_only_messages(unit.video_path, args)], args)
            )
            new_blocks = advance_session(
                model,
                processor,
                session,
                encoder_text,
                str(unit.video_path),
                question_time,
                args,
                annotate_video_temporal_patch_size,
                device=device,
            )
            session.turns += 1

            if session.has_context():
                if verify and new_blocks:
                    verify_session_against_single_pass(
                        model,
                        processor,
                        session,
                        encoder_text,
                        str(unit.video_path),
                        args,
                        annotate_video_temporal_patch_size,
                        device=device,
                    )
                base_inputs = prepare_c2l_base_inputs(model, processor, unit, args)
                prediction = generate_with_loras(
                    model, processor, base_inputs, session_generated_loras(model, session), args
                )
                fallback = False
            else:
                # Nothing has been observed yet: the question arrives before a full temporal
                # block of context exists. Answer it the offline way, from the short prefix,
                # without committing anything to the streaming state.
                prediction = generate_c2l_batch(model, processor, [unit], args)[0]
                session.fallback_turns += 1
                fallback = True

            row = make_result_row(unit, prediction)
            row["interactive"] = {
                "session_index": session_index,
                "turn_index": turn_index,
                "question_time": question_time,
                "annotated_time": annotated_question_time(unit),
                "blocks_consumed": session.consumed_blocks,
                "new_blocks": new_blocks,
                "segments": session.segments,
                "offline_fallback": fallback,
            }
            rows.append(to_jsonable(row))
        finish_session(session)
    return rows, tracked

def shard_units(units: list[EvalUnit], rank: int, world_size: int) -> list[EvalUnit]:
    if world_size <= 1:
        return units
    return [unit for index, unit in enumerate(units) if index % world_size == rank]

@lru_cache(maxsize=8192)
def video_frame_info(video_path: str) -> tuple[int, float] | None:
    try:
        import decord
        reader = decord.VideoReader(video_path)
        total_frames = len(reader)
        fps = float(reader.get_avg_fps() or 0.0)
        if total_frames > 0 and fps > 0:
            return (total_frames, fps)
    except Exception:
        pass
    try:
        import av
        container = av.open(video_path, 'r')
        try:
            stream = next((item for item in container.streams if item.type == 'video'), None)
            if stream is None:
                return None
            fps = float(stream.average_rate or 0.0)
            total_frames = int(getattr(stream, 'frames', 0) or 0)
            if total_frames > 0 and fps > 0:
                return (total_frames, fps)
        finally:
            container.close()
    except Exception:
        pass
    return None

@lru_cache(maxsize=8192)
def video_total_frames(video_path: str) -> int | None:
    info = video_frame_info(video_path)
    if info is None:
        return None
    return info[0]

@lru_cache(maxsize=8192)
def video_duration_seconds(video_path: str) -> float | None:
    info = video_frame_info(video_path)
    if info is not None:
        total_frames, fps = info
        return total_frames / fps
    try:
        import decord
        reader = decord.VideoReader(video_path)
        fps = float(reader.get_avg_fps() or 0.0)
        if fps > 0:
            return len(reader) / fps
    except Exception:
        pass
    try:
        import av
        container = av.open(video_path, 'r')
        try:
            stream = next((item for item in container.streams if item.type == 'video'), None)
            if stream is None:
                return None
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
            fps = float(stream.average_rate or 0.0)
            if fps > 0 and getattr(stream, 'frames', 0):
                return float(stream.frames) / fps
        finally:
            container.close()
    except Exception:
        return None
    return None

def qwen_frame_factor() -> int:
    try:
        from qwen_vl_utils.vision_process import FRAME_FACTOR
        return int(FRAME_FACTOR)
    except Exception:
        return 2

def floor_by_factor(value: int, factor: int) -> int:
    return max(factor, value // factor * factor)

def last_2second_1fps_window(start: float, end: float) -> tuple[float, float, float, int]:
    return last_seconds_1fps_window(start, end, 2.0)

def last_seconds_1fps_window(start: float, end: float, seconds: float) -> tuple[float, float, float, int]:
    video_start = max(0.0, float(start))
    video_end = max(video_start, float(end))
    seconds = max(0.0, float(seconds))
    max_frames = max(1, int(math.ceil(seconds)))
    return (max(video_start, video_end - seconds), video_end, 1.0, max_frames)

def last_2second_1fps_window_for_video(video_path: str) -> tuple[float, float, float, int]:
    return last_seconds_1fps_window_for_video(video_path, 2.0)

def last_seconds_1fps_window_for_video(video_path: str, seconds: float) -> tuple[float, float, float, int]:
    duration = video_duration_seconds(video_path)
    if duration is None:
        raise RuntimeError(f'Cannot determine video duration for tail 1 FPS sampling: {video_path}')
    return last_seconds_1fps_window(0.0, duration, seconds)

def effective_video_frame_count(video_path: str, video_start: float | None=None, video_end: float | None=None) -> int | None:
    info = video_frame_info(video_path)
    if info is None:
        return None
    total_frames, fps = info
    if video_start is None and video_end is None:
        return total_frames
    max_duration = total_frames / fps
    if video_start is not None:
        start_frame = math.ceil(max(0.0, min(float(video_start), max_duration)) * fps)
    else:
        start_frame = 0
    if video_end is not None:
        end_frame = math.floor(max(0.0, min(float(video_end), max_duration)) * fps)
        end_frame = min(end_frame, total_frames - 1)
    else:
        end_frame = total_frames - 1
    return max(0, end_frame - start_frame + 1)

def capped_nframes(video_path: str, requested_frames: int, video_start: float | None=None, video_end: float | None=None) -> int | None:
    total_frames = effective_video_frame_count(video_path, video_start, video_end)
    if total_frames is None:
        return None
    factor = qwen_frame_factor()
    if total_frames < factor:
        return None
    return floor_by_factor(min(int(requested_frames), total_frames), factor)

def video_content(video_path: str, args: argparse.Namespace, frame_count: int | None=None, tail: bool=False) -> dict[str, Any]:
    video_start = None
    video_end = None
    last_2second_1fps = bool(getattr(args, 'last_2second_1fps', False)) and (not tail)
    last_seconds_1fps = getattr(args, 'last_seconds_1fps', None) is not None and (not tail)
    if tail:
        info = video_frame_info(video_path)
        if info is not None:
            total_frames, fps = info
            factor = qwen_frame_factor()
            requested = max(factor, int(frame_count or 1))
            start_frame = max(0, total_frames - requested)
            video_start = start_frame / fps
            video_end = total_frames / fps
        else:
            duration = video_duration_seconds(video_path)
            if duration is not None:
                fps = float(args.video_fps or 2.0)
                video_start = max(0.0, duration - float(frame_count or 1) / max(fps, 1e-06))
                video_end = duration
    elif last_2second_1fps:
        video_start, video_end, video_fps, max_frames = last_2second_1fps_window_for_video(video_path)
    elif last_seconds_1fps:
        video_start, video_end, video_fps, max_frames = last_seconds_1fps_window_for_video(video_path, args.last_seconds_1fps)
    content: dict[str, Any] = {'type': 'video', 'video': video_path, 'min_pixels': args.video_min_pixels, 'max_pixels': args.video_max_pixels}
    if args.video_total_pixels:
        content['total_pixels'] = args.video_total_pixels
    if last_2second_1fps or last_seconds_1fps:
        content['fps'] = video_fps
        content['max_frames'] = max_frames
    elif frame_count is not None:
        nframes = capped_nframes(video_path, frame_count, video_start, video_end)
        if nframes is not None:
            content['nframes'] = nframes
        else:
            content['fps'] = 1000000.0
            content['max_frames'] = frame_count
    elif args.video_fps is not None:
        content['fps'] = args.video_fps
        content['max_frames'] = args.max_frames
    elif args.video_total_pixels:
        content['max_frames'] = args.max_frames
    else:
        nframes = capped_nframes(video_path, args.num_frames)
        if nframes is not None:
            content['nframes'] = nframes
        else:
            content['fps'] = 1000000.0
            content['max_frames'] = args.num_frames
    if video_start is not None:
        content['video_start'] = video_start
    if video_end is not None:
        content['video_end'] = video_end
    return content

def text_part(text: str) -> dict[str, str]:
    return {'type': 'text', 'text': text}

def add_system_message(messages: list[dict[str, Any]], system_prompt: str | None) -> list[dict[str, Any]]:
    if not system_prompt:
        return messages
    return [{'role': 'system', 'content': system_prompt}] + messages

def build_baseline_messages(unit: EvalUnit, args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.no_video:
        return build_text_only_messages(unit.prompt, args)
    if unit.video_path is None:
        raise ValueError('OVO-Bench video input requires a resolved video path. Use --no-video for text-only evaluation.')
    return add_system_message([{'role': 'user', 'content': [video_content(unit.video_path, args), text_part(unit.prompt)]}], args.system_prompt)

def build_video_only_messages(video_path: str | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    if video_path is None:
        raise ValueError('Video C2L evaluation requires video input and cannot be used with --no-video.')
    return [{'role': 'user', 'content': [video_content(video_path, args)]}]

def build_video_text_messages(unit: EvalUnit, args: argparse.Namespace) -> list[dict[str, Any]]:
    if unit.video_path is None:
        raise ValueError('Video C2L base input requires video input and cannot be used with --no-video.')
    return add_system_message([{'role': 'user', 'content': [video_content(unit.video_path, args, frame_count=args.base_input_frame_number, tail=True), text_part(unit.prompt)]}], args.system_prompt)

def build_text_only_messages(prompt: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    return add_system_message([{'role': 'user', 'content': [text_part(prompt)]}], args.system_prompt)

def apply_chat_template(processor, messages: list[list[dict[str, Any]]], args: argparse.Namespace) -> str | list[str]:
    enable_thinking = args.enable_thinking == 'true'
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
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

def processor_video_inputs(processor, messages: list[list[dict[str, Any]]], texts: str | list[str], args: argparse.Namespace):
    from qwen_vl_utils import process_vision_info
    images, videos, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True, image_patch_size=16)
    video_metadata = None
    if videos is not None:
        videos, video_metadata = map(list, zip(*videos))
    processor_kwargs: dict[str, Any] = {'text': texts, 'images': images, 'videos': videos, 'video_metadata': video_metadata, **video_kwargs, 'do_resize': False, 'return_metadata': True, 'return_tensors': 'pt'}
    if len(messages) > 1:
        processor_kwargs.update({'padding': True, 'padding_side': 'left'})
    return processor(**processor_kwargs)

def processor_text_inputs(processor, texts: str | list[str]):
    kwargs: dict[str, Any] = {'text': texts, 'return_tensors': 'pt'}
    if isinstance(texts, list) and len(texts) > 1:
        kwargs.update({'padding': True, 'padding_side': 'left'})
    return processor(**kwargs)

def build_generate_kwargs(processor, args: argparse.Namespace) -> dict[str, Any]:
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    kwargs: dict[str, Any] = {'max_new_tokens': args.max_new_tokens, 'eos_token_id': processor.tokenizer.eos_token_id, 'pad_token_id': pad_token_id, 'do_sample': args.temperature > 0, 'num_beams': args.num_beams}
    if args.temperature > 0:
        kwargs['temperature'] = args.temperature
        if args.top_p is not None:
            kwargs['top_p'] = args.top_p
        if args.top_k is not None:
            kwargs['top_k'] = args.top_k
    return kwargs

def strip_prediction(text: str, enable_thinking: str) -> str:
    text = text.strip()
    if enable_thinking == 'true' and '</think>' in text:
        text = text.rsplit('</think>', maxsplit=1)[-1].strip()
    return text

def strip_option_label(option: str) -> str:
    return re.sub('^[A-Da-d][\\.\\)]\\s*', '', str(option).strip()).strip()

def normalize_option_match_text(text: str) -> str:
    text = strip_option_label(text)
    text = re.sub('\\s+', ' ', text)
    text = re.sub('^[\\"\'“”‘’]+|[\\"\'“”‘’.,;:!?]+$', '', text.strip())
    return text.lower()

def extract_option_letter(response: str | None, options: list[Any] | None=None) -> str | None:
    if response is None:
        return None
    text = str(response).strip()
    if '</think>' in text:
        text = text.rsplit('</think>', maxsplit=1)[-1].strip()
    match = re.match('^(?:assistant\\s*[:：]\\s*)?\\(?([A-Da-d])\\)?\\s*[\\.\\):：,]', text)
    if match:
        return match.group(1).upper()
    match = re.match('^(?:assistant\\s*[:：]\\s*)?\\(?([A-Da-d])\\)?\\s*$', text)
    if match:
        return match.group(1).upper()
    patterns = ('\\b(?:answer|option|choice|letter)\\s*(?:is|:|：)?\\s*\\(?([A-Da-d])\\)?\\b', '\\b(?:select|choose|pick|chosen)\\s*(?:option|choice)?\\s*\\(?([A-Da-d])\\)?\\b')
    for pattern in patterns:
        matches = [match.group(1).upper() for match in re.finditer(pattern, text, flags=re.IGNORECASE)]
        matches = [letter for letter in matches if letter in OPTION_LETTERS]
        if len(set(matches)) == 1:
            return matches[0]
    standalone = []
    tokens = [token for token in re.split('\\s+', text) if token]
    for token in tokens:
        cleaned = token.strip('()[]{}"\'“”‘’.,;:：')
        if len(tokens) == 1 and cleaned in OPTION_LETTERS or (token[:1] in '([{' and cleaned.upper() in OPTION_LETTERS):
            standalone.append(cleaned.upper())
    if len(set(standalone)) == 1:
        return standalone[0]
    if options:
        normalized_response = normalize_option_match_text(text)
        option_matches = []
        for letter, option in zip(OPTION_LETTERS, options, strict=False):
            option_text = normalize_option_match_text(str(option))
            if option_text and (normalized_response == option_text or option_text in normalized_response):
                option_matches.append(letter)
        if len(set(option_matches)) == 1:
            return option_matches[0]
    return None

def extract_count(response: str | None) -> str | None:
    if response is None:
        return None
    text = str(response).strip()
    if '</think>' in text:
        text = text.rsplit('</think>', maxsplit=1)[-1].strip()
    text = re.sub('^[A-Da-d][\\.\\)]\\s*', '', text).strip()
    explicit = re.search('\\b(?:answer|count|total)\\s*(?:is|:|：)?\\s*(\\d+)\\b', text, flags=re.IGNORECASE)
    if explicit:
        return str(int(explicit.group(1)))
    numbers = re.findall('\\d+', text)
    if len(numbers) == 1:
        return str(int(numbers[0]))
    return None

def extract_yes_no(response: str | None) -> str | None:
    if response is None:
        return None
    text = str(response).strip()
    if '</think>' in text:
        text = text.rsplit('</think>', maxsplit=1)[-1].strip()
    lowered = text.lower()
    match = re.search('\\b(?:answer\\s*(?:is|:|：)?\\s*)?(yes|no|y|n)\\b', lowered)
    if not match:
        return None
    value = match.group(1)
    if value == 'y':
        return 'Yes'
    if value == 'n':
        return 'No'
    return value.capitalize()

def preprocess_baseline_batch(processor, units: list[EvalUnit], args: argparse.Namespace):
    messages = [build_baseline_messages(unit, args) for unit in units]
    texts = apply_chat_template(processor, messages, args)
    if args.no_video:
        return processor_text_inputs(processor, texts)
    return processor_video_inputs(processor, messages, texts, args)

def decode_generated_outputs(processor, input_ids, generated_ids, args: argparse.Namespace) -> list[str]:
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)]
    outputs = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [strip_prediction(output, args.enable_thinking) for output in outputs]

def generate_preprocessed_baseline_batch(model, processor, inputs, args: argparse.Namespace) -> list[str]:
    import torch
    inputs.pop('video_metadata', None)
    inputs = inputs.to(model_input_device(model))
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, **build_generate_kwargs(processor, args))
    return decode_generated_outputs(processor, inputs.input_ids, generated_ids, args)

def generate_baseline_batch(model, processor, units: list[EvalUnit], args: argparse.Namespace) -> list[str]:
    inputs = preprocess_baseline_batch(processor, units, args)
    return generate_preprocessed_baseline_batch(model, processor, inputs, args)

def prepare_c2l_inputs(model, processor, units: list[EvalUnit], args: argparse.Namespace):
    from malow.video.loader import remove_empty_think_block
    encoder_messages = [build_video_only_messages(unit.video_path, args) for unit in units]
    encoder_texts = remove_empty_think_block(apply_chat_template(processor, encoder_messages, args))
    interleave_timestamp = bool(getattr(model, 'interleave_timestamp', False))
    if args.base_input_video:
        encoder_frame_count = c2l_encoder_frame_count(args)
        if interleave_timestamp:
            frame_loader = load_base_last_2seconds_c2l_video_frames_with_timestamps if args.base_with_last_2seconds else load_split_c2l_video_frames_with_timestamps
        else:
            frame_loader = load_base_last_2seconds_c2l_video_frames if args.base_with_last_2seconds else load_split_c2l_video_frames
        tail_kwargs = {'tail_seconds': getattr(args, 'base_tail_seconds', 2.0)} if args.base_with_last_2seconds else {}
        split_videos = [frame_loader(str(unit.video_path), encoder_frame_count, args.base_input_frame_number, **tail_kwargs) for unit in units]
        encoder_videos = [item[0] for item in split_videos]
        base_videos = [item[1] for item in split_videos]
        encoder_timestamps = [item[2] for item in split_videos] if interleave_timestamp else None
        encoder_terminal_times = [item[4] for item in split_videos] if interleave_timestamp else None
        encoder_inputs = processor_predecoded_video_inputs(processor, encoder_texts, encoder_videos, args)
    else:
        encoder_inputs = processor_video_inputs(processor, encoder_messages, encoder_texts, args)
        encoder_timestamps = None
        encoder_terminal_times = None
    annotate_video_temporal_patch_size(processor, encoder_inputs)
    if interleave_timestamp:
        add_c2l_timestamp_inputs(encoder_inputs, processor.tokenizer, getattr(model, 'frame_per_lora', 4), encoder_timestamps, encoder_terminal_times)
    if args.base_input_video:
        base_messages = [build_video_text_messages(unit, args) for unit in units]
    else:
        base_messages = [build_text_only_messages(unit.prompt, args) for unit in units]
    base_texts = remove_empty_think_block(apply_chat_template(processor, base_messages, args))
    if args.base_input_video:
        base_inputs = processor_predecoded_video_inputs(processor, base_texts, base_videos, args, branch="base")
    else:
        base_inputs = processor(text=base_texts, padding=True, padding_side='left', return_tensors='pt')
    return (encoder_inputs, base_inputs)

def generate_preprocessed_c2l_batch(model, processor, inputs, args: argparse.Namespace) -> list[str]:
    import torch
    device = model_input_device(model)
    encoder_inputs, base_inputs = inputs
    encoder_inputs = encoder_inputs.to(device)
    base_inputs = base_inputs.to(device)
    with torch.inference_mode():
        generated_ids = model.generate(ctx_ids=encoder_inputs, **base_inputs, **build_generate_kwargs(processor, args))
    return decode_generated_outputs(processor, base_inputs.input_ids, generated_ids, args)

def generate_c2l_batch(model, processor, units: list[EvalUnit], args: argparse.Namespace) -> list[str]:
    inputs = prepare_c2l_inputs(model, processor, units, args)
    return generate_preprocessed_c2l_batch(model, processor, inputs, args)

def load_model(args: argparse.Namespace, local_rank: int, world_size: int):
    if args.c2l_checkpoint:
        from malow.video.loader import load_c2l_model
        return load_c2l_model(args, local_rank, world_size)
    from transformers import AutoModelForImageTextToText
    model_kwargs: dict[str, Any] = {'dtype': args.dtype, 'device_map': resolve_device_map(args.device_map, local_rank, world_size)}
    if args.trust_remote_code:
        model_kwargs['trust_remote_code'] = True
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    model.eval()
    return model

def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, 'item') and callable(value.item):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value

def make_result_row(unit: EvalUnit, prediction: str) -> dict[str, Any]:
    doc = unit.doc
    row = {'task_name': unit.task_name, 'id': doc.get('id'), 'task': doc.get('task'), 'round_idx': unit.round_idx, 'video_path': unit.video_path, 'prompt': unit.prompt, 'prediction': prediction, 'doc': doc}
    if unit.task_name in {'ovo_backward', 'ovo_realtime'}:
        row['question'] = doc.get('question')
        row['ground_truth'] = chr(65 + int(doc['gt']))
        row['answer'] = doc.get('answer')
        row['pred_choice'] = extract_option_letter(prediction, doc.get('options'))
    return to_jsonable(row)

def score_backward_realtime(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    results: list[dict[str, Any]] = []
    raw_scores: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        response = row.get('prediction')
        ground_truth = row.get('ground_truth')
        doc = row.get('doc') if isinstance(row.get('doc'), dict) else {}
        pred_choice = row.get('pred_choice') or extract_option_letter(response, doc.get('options'))
        hit = int(pred_choice == ground_truth)
        result = {'id': row['id'], 'task': row['task'], 'question': row.get('question'), 'response': response, 'pred_choice': pred_choice, 'ground_truth': ground_truth, 'score': hit}
        results.append(result)
        raw_scores[str(row['task'])].append(hit)
    scores = {task: 100.0 * sum(values) / len(values) for task, values in sorted(raw_scores.items()) if values}
    if scores:
        scores['average'] = 100.0 * sum((sum(values) / len(values) for values in raw_scores.values())) / len(raw_scores)
    return (results, scores)

def forward_score_rec(response: str | None, gt: Any) -> int:
    if response is None:
        return 0
    count = extract_count(response)
    return int(count == str(gt))

def forward_score_yes_no(response: str | None, type_value: Any) -> int:
    if response is None:
        return 0
    ground_truth = 'No' if int(type_value) == 0 else 'Yes'
    return int(extract_yes_no(response) == ground_truth)

def score_forward(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    docs: dict[Any, dict[str, Any]] = {}
    for row in rows:
        doc_id = row['id']
        if doc_id not in docs:
            docs[doc_id] = copy.deepcopy(row['doc'])
        round_idx = row['round_idx']
        if round_idx is not None and round_idx < len(docs[doc_id]['test_info']):
            docs[doc_id]['test_info'][round_idx]['response'] = row.get('prediction')
    results = [to_jsonable(doc) for doc in docs.values()]
    raw_scores: dict[str, list[int]] = {task: [] for task in sorted({str(result['task']) for result in results})}
    for result in results:
        task = str(result['task'])
        for test_info in result['test_info']:
            response = test_info.get('response')
            if response is None:
                continue
            if task == 'REC':
                raw_scores['REC'].append(forward_score_rec(response, test_info['count']))
            elif task in {'SSR', 'CRR'}:
                raw_scores[task].append(forward_score_yes_no(response, test_info['type']))
    scores = {task: 100.0 * sum(values) / len(values) for task, values in sorted(raw_scores.items()) if values}
    if scores:
        scores['average'] = 100.0 * sum((sum(values) / len(values) for values in raw_scores.values() if values)) / len(scores)
    return (results, scores)

def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(to_jsonable(row), ensure_ascii=False) + '\n')

def rank_output_path(args: argparse.Namespace, rank: int) -> Path:
    return Path(args.output_dir) / f'rank_{rank}_predictions.jsonl'

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def merge_rank_outputs(args: argparse.Namespace, world_size: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        rows.extend(read_jsonl(rank_output_path(args, rank)))
    return rows

def sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (str(row.get('task_name')), int(row.get('id') or 0), row.get('round_idx') or -1))

def write_outputs(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    rows = sort_rows(rows)
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / 'predictions.jsonl', rows)
    write_json(output_dir / 'predictions.json', rows)
    summary: dict[str, Any] = {'num_prediction_rows': len(rows), 'tasks': {}, 'average': None}
    task_averages: list[float] = []
    for task_type, task_name in (('backward', 'ovo_backward'), ('realtime', 'ovo_realtime')):
        task_rows = [row for row in rows if row.get('task_name') == task_name]
        if not task_rows:
            continue
        results, scores = score_backward_realtime(task_rows)
        write_json(output_dir / f'{task_type}_acc_results.json', results)
        write_json(output_dir / f'{task_type}_acc_scores.json', scores)
        summary['tasks'][task_name] = scores
        if 'average' in scores:
            task_averages.append(scores['average'])
    forward_rows = [row for row in rows if row.get('task_name') == 'ovo_forward']
    if forward_rows:
        results, scores = score_forward(forward_rows)
        write_json(output_dir / 'forward_acc_results.json', results)
        write_json(output_dir / 'forward_acc_scores.json', scores)
        summary['tasks']['ovo_forward'] = scores
        if 'average' in scores:
            task_averages.append(scores['average'])
    if task_averages:
        summary['average'] = sum(task_averages) / len(task_averages)
    write_json(output_dir / 'summary_scores.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary

def run_batches(model, processor, units: list[EvalUnit], args: argparse.Namespace, rank: int) -> list[dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm
    rows: list[dict[str, Any]] = []
    preprocess_batch = prepare_c2l_inputs if args.c2l_checkpoint else preprocess_baseline_batch
    preprocess_prefix = (model, processor) if args.c2l_checkpoint else (processor,)
    generate_preprocessed_batch = generate_preprocessed_c2l_batch if args.c2l_checkpoint else generate_preprocessed_baseline_batch
    batches = [units[start:start + args.batch_size] for start in range(0, len(units), args.batch_size)]
    if not batches:
        return rows
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(preprocess_batch, *preprocess_prefix, batches[0], args)
        iterator = tqdm(range(len(batches)), desc=f'Rank {rank} OVO-Bench', disable=not is_main_process(rank))
        for index in iterator:
            inputs = future.result()
            if index + 1 < len(batches):
                future = executor.submit(preprocess_batch, *preprocess_prefix, batches[index + 1], args)
            outputs = generate_preprocessed_batch(model, processor, inputs, args)
            rows.extend((make_result_row(unit, output) for unit, output in zip(batches[index], outputs)))
    return rows

def main() -> None:
    import torch
    from transformers import AutoProcessor
    args = parse_args()
    if args.no_video and args.c2l_checkpoint:
        raise ValueError('--no-video is only supported for base model evaluation; Video C2L requires video encoder input.')
    rank, local_rank, world_size = init_distributed()
    task_names = normalize_task_list(args.tasks)
    video_root = None if args.no_video else resolve_video_root(args.video_root, auto_download=not args.no_auto_download_videos)
    all_units = build_eval_units(task_names, video_root, args.limit, no_video=args.no_video, annotation_root=args.annotation_root)
    guard_interactive_run(args, world_size)
    init_resolution_log(args, rank)
    sessions = shard_sessions(build_interactive_sessions(all_units, args), rank, world_size) if args.eval_mode != "offline" else None
    units = shard_units(all_units, rank, world_size)
    if is_main_process(rank):
        mode = 'Video C2L' if args.c2l_checkpoint else 'base'
        print(f'Loading {mode} Qwen model: {args.model_id}')
        if args.c2l_checkpoint:
            print(f'C2L checkpoint: {args.c2l_checkpoint}')
        print(f"OVO video root: {(video_root if video_root is not None else '<none; no-video>')}")
        print(f'Evaluation units: total={len(all_units)}, rank0={len(units)}, world_size={world_size}')
    model = load_model(args, local_rank, world_size)
    processor = AutoProcessor.from_pretrained(args.model_id, max_pixels=args.video_max_pixels, min_pixels=args.video_min_pixels, trust_remote_code=args.trust_remote_code)
    if args.eval_mode == "offline":
        rows = run_batches(model, processor, units, args, rank)
    else:
        rows, tracked = run_interactive_sessions(model, processor, sessions, args, rank)
        write_interactive_report(args, rank, tracked)
    write_jsonl(rank_output_path(args, rank), rows)
    print(f'Rank {rank} wrote {len(rows)} rows: {rank_output_path(args, rank)}')
    if world_size > 1:
        torch.distributed.barrier()
        if is_main_process(rank):
            write_outputs(merge_rank_outputs(args, world_size), args)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()
    else:
        write_outputs(rows, args)
if __name__ == '__main__':
    main()
