# Video evaluation: OVO-Bench and StreamingBench

This guide evaluates the released MaLoW memory modules with Qwen3.5-0.8B, 2B, 4B, or 9B, including a 4B variant with a four-second current-observation window, and Qwen3-VL-4B-Instruct with an eight-second window. The base Qwen model is a separate download. Use a Linux machine with an NVIDIA GPU. The reference runs use four data-parallel GPU workers. Each worker holds a complete model and frozen vision encoder, so adding workers increases throughput rather than dividing model memory.

## 1. Environment

Please follow the instruction [here](./env_prep.md) to finish the environment preparation.

## 2. Download checkpoints and base models

Use the release downloader, which records immutable model revisions in the release manifest:

```bash
python -m malow.download --model video-9b --output checkpoints --with-base
```

Replace `video-9b` with `video-4b`, `video-4b-last4s`, `video-2b`, `video-0.8b`, or `video-qwen3vl-4b-last8s`. To inspect the download destinations first, append `--dry-run`. The public memory module directories are:

| Model | Memory module | Base model |
|---|---|---|
| MaLoW (Qwen3.5-9B) | `checkpoints/MaLoW-Qwen3.5-9B` | `checkpoints/Qwen3.5-9B` |
| MaLoW (Qwen3.5-4B) | `checkpoints/MaLoW-Qwen3.5-4B` | `checkpoints/Qwen3.5-4B` |
| MaLoW (Qwen3.5-4B), last 4 seconds | `checkpoints/MaLoW-Qwen3.5-4B-Last4s` | `checkpoints/Qwen3.5-4B` |
| MaLoW (Qwen3.5-Qwen3-VL-4B), last 8 seconds | `checkpoints/MaLoW-Qwen3-VL-4B-Last8s` | `checkpoints/Qwen3-VL-4B-Instruct` |
| MaLoW (Qwen3.5-2B) | `checkpoints/MaLoW-Qwen3.5-2B` | `checkpoints/Qwen3.5-2B` |
| MaLoW (Qwen3.5-0.8B) | `checkpoints/MaLoW-Qwen3.5-0.8B` | `checkpoints/Qwen3.5-0.8B` |

Memory checkpoints contain `model.safetensors`, `c2l_metadata.json`, and a small `c2l_metadata.safetensors` companion when needed. The runtime never unpickles model metadata. Always pair a memory module with its matching base-model architecture and size.

## 3. Prepare OVO-Bench

Download the evaluator-compatible annotation parquets from the [kcz358/OVOBench mirror](https://huggingface.co/datasets/kcz358/OVOBench) for either video option below:

```bash
hf download kcz358/OVOBench --repo-type dataset \
  --include 'backward/*.parquet' 'realtime/*.parquet' 'forward/*.parquet' \
  --local-dir data/ovo-annotations
```

The annotation root must contain `backward/test-00000-of-00001.parquet`, `realtime/test-00000-of-00001.parquet`, and `forward/test-00000-of-00001.parquet`. Choose **one** of the following video options.

### Option A: download the prepared videos

To skip local video downsampling, download [ovobench-2fps-min64.tar](https://huggingface.co/datasets/zhuomingliu/MaLoW_others/blob/main/ovobench-2fps-min64.tar), approximately 41.92 GB. It contains the temporally downsampled videos used to speed up evaluation:

```bash
mkdir -p data/downloads
hf download zhuomingliu/MaLoW_others ovobench-2fps-min64.tar \
  --repo-type dataset --revision 6ed5b7129b966c22535a988d7ba20cc320f7abea \
  --local-dir data/downloads
tar -xf data/downloads/ovobench-2fps-min64.tar -C data
```

Pass `data/ovobench-2fps-min64/chunked_videos` as `--video-root`. Keep the separate annotation download above. This option does not require running the downsampling utility.

### Option B: download official videos and downsample locally

The [official OVO-Bench repository](https://github.com/JoeLeelyf/OVO-Bench) describes the videos and benchmark protocol. Download its original pre-chunked videos from [JoeLeelyf/OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench), then apply the reference temporal-downsampling policy:

```bash
mkdir -p data/ovo-download data/ovo-original
hf download JoeLeelyf/OVO-Bench --repo-type dataset \
  --include 'chunked_videos.tar.parta*' --local-dir data/ovo-download
cat data/ovo-download/chunked_videos.tar.parta* | tar -xf - -C data/ovo-original
python -m malow.video.data.downsample_ovo \
  --input-root data/ovo-original --output-root data/ovobench-2fps-min64 \
  --target-fps 2 --min-frames 64 --workers 8 --ffmpeg-threads 1 \
  --crf 28 --preset veryfast
```

This preserves image dimensions and targets 2 FPS, increasing the rate for short clips when needed to retain 64 frames. Inputs with fewer than 64 source frames remain unchanged. A processing manifest records copied and re-encoded files. Check it for conversion failures: the implementation copies an original when conversion fails.

Both options produce `data/ovobench-2fps-min64/chunked_videos`. Original official videos can also be evaluated by passing `data/ovo-original/chunked_videos`; the reported table uses temporally downsampled videos.

## 4. Prepare StreamingBench

Choose **one** of the following options.

### Option A: download the prepared videos and annotations

To skip local video downsampling, download [streamingbench-2fps-min64.tar](https://huggingface.co/datasets/zhuomingliu/MaLoW_others/blob/main/streamingbench-2fps-min64.tar), approximately 68.23 GB. This archive includes the prepared videos and evaluation annotations:

```bash
mkdir -p data/downloads
hf download zhuomingliu/MaLoW_others streamingbench-2fps-min64.tar \
  --repo-type dataset --revision 6ed5b7129b966c22535a988d7ba20cc320f7abea \
  --local-dir data/downloads
tar -xf data/downloads/streamingbench-2fps-min64.tar -C data
```

Pass `data/streamingbench-2fps-min64/annotations` as `--data-root` and `data/streamingbench-2fps-min64/videos` as `--video-root`. The latter overrides the machine-specific absolute video paths saved in the annotations. No local downsampling or separate annotation download is required for this option.

### Option B: download official sources and downsample locally

Get annotations from the [official repository](https://github.com/THUNLP-MT/StreamingBench), and original video ZIPs from [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench):

```bash
mkdir -p third_party
git clone https://github.com/THUNLP-MT/StreamingBench.git third_party/StreamingBench
hf download mjuicem/StreamingBench --repo-type dataset \
  --include '*.zip' --local-dir data/streamingbench-download
python -m malow.video.data.streaming_archives \
  --annotation-root third_party/StreamingBench/src/data \
  --archive-root data/streamingbench-download \
  --output-root data/streamingbench-original --strict
python -m malow.video.data.downsample_streaming \
  --source-root data/streamingbench-original \
  --output-root data/streamingbench-2fps-min64 \
  --annotation-video-root "$PWD/data/streamingbench-2fps-min64/videos" \
  --target-fps 2 --min-frames 64 --no-max-frame-cap --referenced-only \
  --workers 8 --ffmpeg-threads 1 --crf 28 --preset veryfast
```

The archive helper extracts only referenced videos, renames archive members to annotation-compatible filenames, and checks that every video exists. The prepared tree has **899 videos**: 898 downsampled videos and one original copy. Unlike the OVO utility, the StreamingBench policy copies a video unchanged when 2 FPS would produce fewer than 64 frames. There is no preprocessing maximum-frame cap.

Both options provide this evaluation layout (the prepared archive also contains additional annotation variants):

```text
data/streamingbench-2fps-min64/
  annotations/questions_real.json
  annotations/questions_omni.json
  annotations/questions_sqa_stream.json
  annotations/questions_proactive.json
  videos/*.mp4
  manifest.json
  downsample_manifest.jsonl
```

Prepared archives are evaluation-speed conveniences; official videos remain available from the benchmark sources and their upstream terms still apply.

## 5. Run evaluation

Set the model size and number of workers. The same commands work with `SIZE=4B`, `2B`, or `0.8B` after downloading that pair:

```bash
SIZE=9B
WORKERS=4
python -m malow.video.eval --benchmark ovo \
  --checkpoint "checkpoints/MaLoW-Qwen3.5-$SIZE" \
  --base-model "checkpoints/Qwen3.5-$SIZE" \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir "outputs/ovo-$SIZE" --num-processes "$WORKERS" --offline

python -m malow.video.eval --benchmark streaming \
  --checkpoint "checkpoints/MaLoW-Qwen3.5-$SIZE" \
  --base-model "checkpoints/Qwen3.5-$SIZE" \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir "outputs/streaming-$SIZE" --num-processes "$WORKERS" --offline
```

For the **MaLoW (Qwen3.5-4B), last 4 seconds** checkpoint, use its own memory-module directory with the same Qwen3.5-4B base model. The default `auto` preset selects four seconds and four frames from its metadata:

```bash
python -m malow.download --model video-4b-last4s --output checkpoints --with-base
python -m malow.video.eval --benchmark ovo \
  --checkpoint checkpoints/MaLoW-Qwen3.5-4B-Last4s \
  --base-model checkpoints/Qwen3.5-4B \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir outputs/ovo-4B-last4s --num-processes 4 --offline
python -m malow.video.eval --benchmark streaming \
  --checkpoint checkpoints/MaLoW-Qwen3.5-4B-Last4s \
  --base-model checkpoints/Qwen3.5-4B \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir outputs/streaming-4B-last4s --num-processes 4 --offline
```

For the **MaLoW (Qwen3.5-Qwen3-VL-4B), last 8 seconds** checkpoint
```bash
python -m malow.download --model video-qwen3vl-4b-last8s --output checkpoints --with-base
python -m malow.video.eval --benchmark ovo \
  --checkpoint checkpoints/MaLoW-Qwen3-VL-4B-Last8s \
  --base-model checkpoints/Qwen3-VL-4B-Instruct \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir outputs/ovo-qwen3vl-4B-last8s --num-processes 4 --offline
python -m malow.video.eval --benchmark streaming \
  --checkpoint checkpoints/MaLoW-Qwen3-VL-4B-Last8s \
  --base-model checkpoints/Qwen3-VL-4B-Instruct \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir outputs/streaming-qwen3vl-4B-last8s --num-processes 4 --offline
```

The default `--preset auto` reads eight seconds/eight frames from the checkpoint. 
An explicit `--preset last2s`, `--preset last4s`, or `--preset last8s` overrides checkpoint metadata for controlled comparisons. Use the matching preset to reproduce the reference settings. For a dry run before downloading a checkpoint, specify the preset explicitly; `auto` requires the checkpoint's JSON metadata.

Use `WORKERS=1` for one GPU, or set `CUDA_VISIBLE_DEVICES` to select the devices. Add `--dry-run` to print all resolved arguments without loading models or writing results. Add `--limit 1` and use a fresh output directory for a smoke test. 


Outputs include per-rank predictions and benchmark `summary.json`. OVO also writes split/subtask score files; StreamingBench writes per-task `stats.json` and merged predictions. SQA uses previous gold QA history, matching the original benchmark protocol. Proactive evaluation polls every second through four seconds after the target time and requires both trigger timing and output correctness.

## 6. Interactive evaluation

The commands above still use **offline evaluation**, the default. The two opt-in modes carry video memory between questions and apply to both benchmarks:

- `--eval-mode interactive`: Rollout and consume streaming video base on the question timestamps.
- `--eval-mode fixed-gap --rollout-step-seconds 2`: Rollout and consume streaming video on ticks every two seconds.

Both use a fixed **1 FPS** encoder grid, **two-frame** temporal blocks, and the current-observation window selected by `--preset auto`. At video time `t`, the encoder commits only complete blocks before `t - base_tail_seconds`; incomplete blocks wait for more footage. The base model receives the recent two seconds/two frames, four seconds/four frames for `video-4b-last4s`, or eight seconds/eight frames for `video-qwen3vl-4b-last8s`. 

The rollout interval controls **when memory updates happen**. Sampling FPS controls **which frames enter memory**. The observation window controls **which recent frames the base model sees directly**. These are separate settings. A two-second tick need not update memory if no complete historical block has become available. Evaluation advances along the recorded video clock. It does not generate answers at ticks without question.

### Run both variants

After the same environment, checkpoint, and data preparation above, run these Bash commands from the repository root. Each mode gets a separate output directory:

```bash
SIZE=9B
WORKERS=4
COMMON=(
  --checkpoint "checkpoints/MaLoW-Qwen3.5-$SIZE"
  --base-model "checkpoints/Qwen3.5-$SIZE"
  --num-processes "$WORKERS" --offline
)

# Question-driven OVO-Bench.
python -m malow.video.eval "${COMMON[@]}" \
  --benchmark ovo --eval-mode interactive \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir "outputs/ovo-$SIZE-interactive"

# Two-second rollout OVO-Bench.
python -m malow.video.eval "${COMMON[@]}" \
  --benchmark ovo --eval-mode fixed-gap --rollout-step-seconds 2 \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir "outputs/ovo-$SIZE-fixed-gap-2s"

# Question-driven StreamingBench.
python -m malow.video.eval "${COMMON[@]}" \
  --benchmark streaming --eval-mode interactive \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir "outputs/streaming-$SIZE-interactive"

# Two-second rollout StreamingBench.
python -m malow.video.eval "${COMMON[@]}" \
  --benchmark streaming --eval-mode fixed-gap --rollout-step-seconds 2 \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir "outputs/streaming-$SIZE-fixed-gap-2s"
```

`--offline` here means **no network model downloads**; `--eval-mode offline` selects the original evaluation protocol. They are independent.

For the **MaLoW (Qwen3.5-4B), last 4 seconds** checkpoint, replace `COMMON` before running the commands, and use new output directories with a `last4s` suffix:

```bash
COMMON=(
  --checkpoint checkpoints/MaLoW-Qwen3.5-4B-Last4s
  --base-model checkpoints/Qwen3.5-4B
  --num-processes 4 --offline
)
```

`auto` selects the four-second observation window; the rollout step can still be two seconds. Other released sizes use the same commands with their matching checkpoint/base-model pair.

Append `--dry-run` to inspect the resolved worker command without loading models or writing files. Append `--limit 1` and use a fresh output directory for a smoke test. For example:

```bash
python -m malow.video.eval \
  --benchmark ovo --eval-mode fixed-gap --rollout-step-seconds 2 \
  --checkpoint checkpoints/MaLoW-Qwen3.5-4B-Last4s \
  --base-model checkpoints/Qwen3.5-4B \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir outputs/ovo-4B-last4s-fixed-gap-smoke \
  --num-processes 1 --limit 1 --offline
```

For the **MaLoW (Qwen3-VL-4B), last 8 seconds** checkpoint this Bash loop runs both modes on both benchmarks with separate outputs:

```bash
COMMON=(
  --checkpoint checkpoints/MaLoW-Qwen3-VL-4B-Last8s
  --base-model checkpoints/Qwen3-VL-4B-Instruct
  --num-processes 4 --offline
)
for MODE in interactive fixed-gap; do
  ROLLOUT=()
  if [[ "$MODE" == fixed-gap ]]; then
    ROLLOUT=(--rollout-step-seconds 2)
  fi
  python -m malow.video.eval "${COMMON[@]}" \
    --benchmark ovo --eval-mode "$MODE" "${ROLLOUT[@]}" \
    --data-root data/ovo-annotations \
    --video-root data/ovobench-2fps-min64/chunked_videos \
    --output-dir "outputs/ovo-qwen3vl-4B-last8s-$MODE"
  python -m malow.video.eval "${COMMON[@]}" \
    --benchmark streaming --eval-mode "$MODE" "${ROLLOUT[@]}" \
    --data-root data/streamingbench-2fps-min64/annotations \
    --video-root data/streamingbench-2fps-min64/videos \
    --output-dir "outputs/streaming-qwen3vl-4B-last8s-$MODE"
done
```

In both modes, the base model sees eight frames from the latest eight seconds. History enters memory in two-frame blocks before that observation window. `interactive` updates at question timestamps; `fixed-gap` advances every two video seconds and catches up at the question timestamp. The eight-second observation window does not change the two-second rollout step. Set `CUDA_VISIBLE_DEVICES=0` and use `--num-processes 1` for a single selected GPU; append `--limit 1` for a smoke run or `--dry-run` to inspect the command. 

### Measured evaluation time
The following evaluation time profiling runs on **eight H100 80GB GPUs**, two-second rollout took **13.4% longer overall** than question-timestamp rollout across all five checkpoints and both benchmarks: **9h 45m versus 8h 36m**. The mean of the ten individual runtime ratios was **1.136×** (13.6% longer).

For **MaLoW 9B**, including the default offline evaluation:

| Evaluation mode | Time |
| --- | ---: |
| Default offline | 1h 45m 31s |
| Interactive, question-timestamp rollout | 1h 46m 48s |
| Interactive, two-second rollout | 2h 1m 11s |


## 7. Results and scoring

Our re-evaluation results show that the average deviation from the reported results below is less than one percentage point per checkpoint in both offline and interactive evaluation settings. On narrow screens, scroll each table horizontally to view all columns.

### OVO-Bench

| Model | Backward | Real&#8209;time | Forward | Overall |
| --- | ---: | ---: | ---: | ---: |
| MaLoW&nbsp;(Qwen3.5&#8209;9B) | 66.35 | 76.16 | 51.48 | 64.64 |
| MaLoW&nbsp;(Qwen3.5&#8209;4B) | 61.15 | 70.52 | 50.82 | 60.83 |
| MaLoW&nbsp;(Qwen3.5&#8209;4B),&nbsp;last&nbsp;4&nbsp;seconds | 62.54 | 74.67 | 49.32 | 62.17 |
| MaLoW&nbsp;(Qwen3&#8209;VL&#8209;4B),&nbsp;last&nbsp;8&nbsp;seconds | 63.78 | 74.68 | 50.68 | 63.04 |
| MaLoW&nbsp;(Qwen3.5&#8209;2B) | 63.08 | 70.14 | 49.33 | 60.85 |
| MaLoW&nbsp;(Qwen3.5&#8209;0.8B) | 54.70 | 67.70 | 48.65 | 57.00 |

### StreamingBench

| Model | Real&#8209;time | Omni&#8209;source | Proactive | Sequential&nbsp;QA | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| MaLoW&nbsp;(Qwen3.5&#8209;9B) | 82.16 | 62.06 | 58.80 | 63.60 | 73.13 |
| MaLoW&nbsp;(Qwen3.5&#8209;4B) | 80.80 | 60.20 | 56.80 | 55.20 | 71.18 |
| MaLoW&nbsp;(Qwen3.5&#8209;4B),&nbsp;last&nbsp;4&nbsp;seconds | 82.28 | 63.70 | 68.00 | 50.00 | 73.51 |
| MaLoW&nbsp;(Qwen3&#8209;VL&#8209;4B),&nbsp;last&nbsp;8&nbsp;seconds | 84.84 | 61.80 | 56.80 | 58.80 | 74.15 |
| MaLoW&nbsp;(Qwen3.5&#8209;2B) | 80.28 | 54.30 | 54.40 | 55.60 | 68.82 |
| MaLoW&nbsp;(Qwen3.5&#8209;0.8B) | 76.68 | 48.86 | 26.00 | 48.40 | 63.02 |

OVO averages subtask accuracies within each split, then averages the three splits. StreamingBench uses **question-weighted accuracy** with weights real:omni:proactive:SQA = **2500:1500:250:250**. Its `weighted_average.accuracy_percent` is the release metric.
