# Train MaLoW video memory modules

This guide trains the MaLoW and evaluates the resulting checkpoints
using the existing OVO-Bench and StreamingBench protocols. Training data and
evaluation data are prepared separately. The launchers are portable Python/Bash
entrypoints; a Slurm installation or PlatformR account is not required.

## Quickstart for MaLoW (Qwen3.5-9B) on eight GPUs

Run these Bash commands from the training checkout with the `malow-video`
environment active. Install the [training dependencies](env_prep.md#training-additions)
and FFmpeg (`ffprobe`), place the base model at `checkpoints/Qwen3.5-9B`, and
[prepare the evaluation data](video_evaluation.md) under `data/`. If the assets
live in another worktree, use their absolute paths in the commands below.

### Prepare training data

The public [MaLoW-Instruct dataset](https://huggingface.co/datasets/zhuomingliu/MaLoW-Instruct)
contains the prepared videos and the filtered SFT replay schedule. Use a fresh
`data/training` directory for the download and audit.

```bash
python -m malow.training.data download --data-root data/training
python -m malow.training.data audit --data-root data/training
```

The downloader verifies and extracts 15 video archives under
`data/training/videos/` and installs the manifest, eight replay shards, and
their offset indexes under `data/training/annotations/`. The audit verifies
the replay checksums and step ordering, then probes each distinct referenced
video with `ffprobe`.

### Train on the filtered replay

Pass the downloaded manifest to select replay mode and use a fresh output
directory. With one sample per GPU and eight gradient accumulation steps,
each optimizer step uses its scheduled 64 occurrences. The run processes
179,200 occurrences in 2,800 steps.

```bash
python -m malow.training.train --model video-9b \
  --replay-manifest data/training/annotations/manifest.json \
  --videos-root data/training/videos \
  --base-model checkpoints/Qwen3.5-9B \
  --output-dir outputs/train-9b-filtered-replay \
  --num-processes 8 --gradient-accumulation-steps 8 --offline
```

### Evaluate the reproduced checkpoint
Prepare OVO-Bench and StreamingBench following the existing
[evaluation data instructions](video_evaluation.md).

After training saves `checkpoint-2800`, run OVO-Bench and StreamingBench in
offline evaluation mode, sequentially on all eight GPUs.

```bash
set -e
common=(
  --checkpoint outputs/train-9b-filtered-replay/checkpoint-2800
  --base-model checkpoints/Qwen3.5-9B
  --num-processes 8 --eval-mode offline --offline
)

python -m malow.video.eval "${common[@]}" --benchmark ovo \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir outputs/eval-9b-filtered-replay/ovo

python -m malow.video.eval "${common[@]}" --benchmark streaming \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir outputs/eval-9b-filtered-replay/streaming
```

Read the OVO-Bench average in `ovo/summary_scores.json` and the StreamingBench
weighted accuracy in `streaming/summary.json` under the evaluation output directory.

