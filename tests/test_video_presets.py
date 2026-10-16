"""Checkpoint-selected observation windows and their sampling boundaries."""
import json
import subprocess
import sys

import pytest

from malow.checkpoints import FORMAT, FORMAT_VERSION, METADATA_FILENAME
from malow.video.eval import build_command, parse_args, resolve_observation_preset


def metadata_dir(tmp_path, metadata):
    checkpoint = tmp_path / "arbitrary-checkpoint-name"
    checkpoint.mkdir()
    (checkpoint / METADATA_FILENAME).write_text(json.dumps({
        "format": FORMAT, "version": FORMAT_VERSION, "metadata": metadata,
    }))
    return checkpoint


def launcher_args(checkpoint, benchmark="ovo", extra=()):
    return ["--benchmark", benchmark, "--checkpoint", str(checkpoint),
            "--base-model", "base", "--data-root", "annotations",
            "--video-root", "videos", "--output-dir", "outputs", *extra]


@pytest.mark.parametrize("benchmark", ["ovo", "streaming"])
@pytest.mark.parametrize("tail,frames", [(None, 2), (4.0, 4), (8.0, 8)])
def test_auto_metadata_reaches_worker_and_preserves_evaluation_cap(tmp_path, monkeypatch, benchmark, tail, frames):
    from malow.video import ovo, streaming
    metadata = {"base_input_frame_number": frames, "base_last_2seconds_max_frames": 256}
    if tail is not None:
        metadata["base_tail_seconds"] = tail
    checkpoint = metadata_dir(tmp_path, metadata)
    args = parse_args(launcher_args(checkpoint, benchmark))
    assert args.preset == "auto"
    command = build_command(args)
    monkeypatch.setattr(sys, "argv", ["worker", *command[3:]])
    parsed = (ovo if benchmark == "ovo" else streaming).parse_args()
    assert parsed.base_tail_seconds == (2.0 if tail is None else tail)
    assert parsed.base_input_frame_number == frames
    assert parsed.base_input_video and parsed.base_with_last_2seconds
    assert parsed.num_frames == parsed.max_frames == 766
    assert parsed.frame_per_lora == 2
    assert parsed.micro_c2l_forward_frame is None


@pytest.mark.parametrize("preset,tail,frames", [("last2s", 2.0, 2), ("last4s", 4.0, 4), ("last8s", 8.0, 8)])
def test_explicit_preset_overrides_metadata(tmp_path, preset, tail, frames):
    checkpoint = metadata_dir(tmp_path, {"base_tail_seconds": 6.0, "base_input_frame_number": 6})
    command = build_command(parse_args(launcher_args(checkpoint, extra=("--preset", preset))))
    assert command[command.index("--base-tail-seconds") + 1] == str(tail)
    assert command[command.index("--base-input-frame-number") + 1] == str(frames)
    assert resolve_observation_preset(tmp_path / "not-downloaded", preset) == preset


def test_missing_metadata_requires_explicit_preset(tmp_path):
    with pytest.raises(FileNotFoundError, match="choose --preset last2s.*--preset last8s"):
        resolve_observation_preset(tmp_path)


def test_legacy_observation_defaults(tmp_path):
    assert resolve_observation_preset(metadata_dir(tmp_path, {})) == "last2s"


@pytest.mark.parametrize("metadata", [
    {"base_tail_seconds": 4.0, "base_input_frame_number": 2},
    {"base_input_frame_number": 4},
    {"base_tail_seconds": "4.0", "base_input_frame_number": 4},
])
def test_ambiguous_metadata_does_not_silently_choose_wrong_window(tmp_path, metadata):
    with pytest.raises(ValueError, match="Unsupported observation settings"):
        resolve_observation_preset(metadata_dir(tmp_path, metadata))


@pytest.mark.parametrize("tail", [4.0, 8.0])
@pytest.mark.parametrize("mode", ["offline", "interactive", "fixed-gap"])
def test_auto_dry_run_does_not_import_model_libraries(tmp_path, tail, mode):
    checkpoint = metadata_dir(tmp_path, {"base_tail_seconds": tail, "base_input_frame_number": int(tail)})
    script = (
        "import sys; from malow.video.eval import main; "
        "assert main(sys.argv[1:]) == 0; "
        "assert not {'torch', 'transformers', 'safetensors'} & set(sys.modules)"
    )
    result = subprocess.run([sys.executable, "-c", script, *launcher_args(checkpoint, extra=("--dry-run", "--eval-mode", mode))],
                            check=True, capture_output=True, text=True)
    assert f"--base-tail-seconds {tail}" in result.stdout
    assert f"--base-input-frame-number {int(tail)}" in result.stdout
    assert "--num-frames 766" in result.stdout


@pytest.mark.parametrize("tail", [4.0, 8.0])
@pytest.mark.parametrize("offset,expected_prefix", [(-2.5, None), (0.0, None), (0.5, 1), (3.0, 3), (1000.0, 766)])
def test_tail_sampling_boundaries(tmp_path, monkeypatch, tail, offset, expected_prefix):
    from malow.video import sampling
    start = 5.0
    end = start + tail + offset
    monkeypatch.setattr(sampling, "_resolve_clip_bounds", lambda *args: (start, end))
    calls = []

    def load(_path, count, clip_start_seconds, clip_end_seconds):
        calls.append((count, clip_start_seconds, clip_end_seconds))
        return list(range(count)), [clip_start_seconds + i for i in range(count)], clip_end_seconds

    monkeypatch.setattr(sampling, "load_video_clip_frames_with_timestamps", load)
    enc, base, enc_times, base_times, terminal = sampling.load_base_last_2seconds_c2l_video_frames_with_timestamps(
        "clip", 766, int(tail), clip_start_seconds=start, clip_end_seconds=end, tail_seconds=tail,
    )
    assert len(base) == int(tail)
    if expected_prefix is None:
        assert enc is base and enc_times is base_times
        assert calls == [(int(tail), start, end)]
        assert terminal == end
    else:
        assert calls == [(expected_prefix, start, end - tail), (int(tail), end - tail, end)]
        assert len(enc) == expected_prefix + expected_prefix % 2
        assert terminal == end - tail
