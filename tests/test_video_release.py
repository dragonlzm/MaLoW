"""Release behavior tests: presets, metrics, portable paths, and preprocessing."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import shutil
import subprocess
import sys

import pytest

from malow.video import sampling, streaming
from malow.video.eval import parse_args, build_command
from malow.video.data import downsample_ovo, downsample_streaming


def args(benchmark="ovo", **extra):
    extra.setdefault("preset", "last2s")
    raw = ["--benchmark", benchmark, "--checkpoint", "checkpoint", "--base-model", "base",
           "--data-root", "annotations", "--video-root", "videos", "--output-dir", "outputs"]
    raw += [token for key, value in extra.items() for token in ("--" + key.replace("_", "-"), str(value))]
    return parse_args(raw)


@pytest.mark.parametrize("benchmark", ["ovo", "streaming"])
def test_preset_roundtrips_through_worker_parser(benchmark, monkeypatch):
    from malow.video import ovo
    worker = ovo if benchmark == "ovo" else streaming
    cmd = build_command(args(benchmark))
    monkeypatch.setattr(sys, "argv", ["worker", *cmd[3:]])
    parsed = worker.parse_args()
    assert parsed.num_frames == parsed.max_frames == 766
    assert parsed.base_input_frame_number == 2
    assert parsed.base_with_last_2seconds and parsed.base_input_video
    assert parsed.base_tail_seconds == 2.0
    assert parsed.frame_per_lora == 2 and parsed.block_lora == "true"
    assert parsed.vision_encoder_only
    assert parsed.micro_c2l_forward_frame is None
    assert parsed.system_prompt == "" and parsed.temperature == 0
    assert parsed.video_total_pixels == 0
    assert parsed.max_new_tokens == (128 if benchmark == "ovo" else 16)


def test_multi_process_uses_module_dispatch():
    command = build_command(args(num_processes=4))
    assert "--nproc_per_node=4" in command
    assert command[command.index("--module") + 1] == "malow.video.ovo"


def test_frame_count_precedence():
    assert sampling.c2l_encoder_frame_count(SimpleNamespace(num_frames=766, max_frames=256)) == 766
    assert sampling.c2l_encoder_frame_count(SimpleNamespace(num_frames=None, max_frames=766)) == 766


@pytest.mark.parametrize("duration,expected", [(1.5,2),(4.5,4),(1002,766)])
def test_long_short_odd_sampling(monkeypatch, duration, expected):
    monkeypatch.setattr(sampling, "_resolve_clip_bounds", lambda *args: (0.0,duration))
    calls = []
    def load(_path, count, clip_start_seconds, clip_end_seconds):
        calls.append((count,clip_start_seconds,clip_end_seconds))
        return list(range(count)), [float(i) for i in range(count)], clip_end_seconds
    monkeypatch.setattr(sampling, "load_video_clip_frames_with_timestamps", load)
    enc,base,*_ = sampling.load_base_last_2seconds_c2l_video_frames_with_timestamps("clip",766,2)
    assert len(enc) == expected and len(base) == 2
    if duration <= 2:
        assert enc is base and len(calls) == 1
    else:
        assert calls[-1] == (2,duration-2,duration)


@pytest.mark.parametrize("scores,expected", [
    ([82.16,62.06,58.8,63.6],73.13),
    ([80.8,60.2,56.8,55.2],71.18),
    ([80.28,54.3,54.4,55.6],68.81),
    ([76.68,48.86,26,48.4],63.02),
])
def test_paper_weighted_metrics(scores, expected):
    row = {name:{"accuracy":value/100} for name,value in zip(("real","omni","proactive","sqa"),scores)}
    assert round(streaming.build_weighted_summary(row)["accuracy_percent"],2) == expected


def test_streaming_video_root_override(tmp_path, monkeypatch):
    annotation = tmp_path/"annotations"/"questions_real.json"
    root = tmp_path/"relocated-videos"
    monkeypatch.setenv("MALOW_VIDEO_ROOT",str(root))
    paths=streaming.candidate_video_paths(Path("old/video.mp4"), annotation)
    assert paths[0] == root/"video.mp4"


def test_downsampling_policies_remain_distinct():
    ovo = downsample_ovo.VideoMetadata(width=100,height=100,fps=30,duration=10,frames=300,codec="h264")
    stream = downsample_streaming.VideoMetadata(width=100,height=100,fps=30,duration=10,frames=300,codec="h264")
    assert downsample_ovo.choose_target_fps(ovo,2,64)[0] == 6.4
    assert downsample_streaming.selected_fps(stream,2,768,64,True)[0] is None
    stream_long = downsample_streaming.VideoMetadata(width=100,height=100,fps=30,duration=1000,frames=30000,codec="h264")
    assert downsample_streaming.selected_fps(stream_long,2,768,64,True)[0] == 2


def test_runtime_has_no_training_or_original_checkout_dependencies():
    root=Path(sampling.__file__).parent
    for path in root.rglob("*.py"):
        text=path.read_text()
        assert "from llamafactory" not in text
        assert "from ctx_to_lora" not in text
        assert "from run_mmbench" not in text
        assert "torch.load(" not in text
        assert "/mnt/scratch/" not in text
