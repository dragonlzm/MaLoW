# MaLoW with text-based memory: evaluation

## 1. Motivation

Text summaries are widely used as memory in streaming and agentic pipelines. This experiment demonstrates that text-based memory can complement MaLoW's visual LoRA memory, further improving model performance. Specifically, video advance in a fixed rollout intervals to generate captions, which are subsequently summarized and provided as additional context during question answering. In parallel, the same video stream is used to construct MaLoW's visual LoRA memory. We evaluate this combined approach on OVO-Bench and StreamingBench using **MaLoW (Qwen3.5-9B)** exclusively.

## 2. Environment preparation

Follow the [environment preparation instructions](./video_evaluation.md#1-environment).
Run the commands below from the repository root on a Linux machine with an NVIDIA GPU.
The examples use four GPU workers; set `WORKERS=1` for a single GPU.

## 3. Download checkpoint

Download MaLoW (Qwen3.5-9B) and its matching base model:

```bash
python -m malow.download --model video-9b --output checkpoints --with-base
```

This creates `checkpoints/MaLoW-Qwen3.5-9B` and `checkpoints/Qwen3.5-9B`.
See [checkpoint preparation](./video_evaluation.md#2-download-checkpoints-and-base-models)
for more information.

## 4. Prepare OVO-Bench

Follow [Prepare OVO-Bench](./video_evaluation.md#3-prepare-ovo-bench).
The commands below use `data/ovo-annotations` for annotations and
`data/ovobench-2fps-min64/chunked_videos` for videos.

## 5. Prepare StreamingBench

Follow [Prepare StreamingBench](./video_evaluation.md#4-prepare-streamingbench).
The commands below use `data/streamingbench-2fps-min64/annotations` for annotations
and `data/streamingbench-2fps-min64/videos` for videos.


## 6. Evaluate from scratch

To generate captions and summaries directly from the videos, both benchmarks process videos in 30-second rollout intervals, with question timestamps potentially triggering additional memory updates.
Specifically, the base model, Qwen3.5-9B, generates video captions and subsequently summarizes them into text-based memory. In parallel, the same video stream is processed by MaLoW to generate visual LoRA memory. During question answering, the base model incorporates the generated LoRA weights and takes the text-based memory, the most recent 2-second video observation, and the question as inputs to produce an answer.
StreamingBench uses the same video grouping as the replay above.

```bash
WORKERS=4
python -m malow.video.eval --benchmark ovo \
  --checkpoint checkpoints/MaLoW-Qwen3.5-9B \
  --base-model checkpoints/Qwen3.5-9B \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir outputs/ovo-9b-text-summary-30s-scratch \
  --num-processes "$WORKERS" --offline \
  --eval-mode fixed-gap --rollout-step-seconds 30 \
  --text-summary --checkpoint-progress

python -m malow.video.eval --benchmark streaming \
  --checkpoint checkpoints/MaLoW-Qwen3.5-9B \
  --base-model checkpoints/Qwen3.5-9B \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir outputs/streaming-9b-text-summary-30s-scratch \
  --num-processes "$WORKERS" --offline \
  --eval-mode fixed-gap --rollout-step-seconds 30 \
  --text-summary --checkpoint-progress --streaming-group-by-video
```

Generating captions and summaries takes longer than reusing the prepared summaries.


## 7. Evaluate with prepared summaries
Generating video captions through sequential video rollouts and subsequently compressing them into text-based summaries can be time-consuming. To accelerate evaluation, you can download the [precomputed captions and summaries](https://huggingface.co/datasets/zhuomingliu/testDataset/tree/main/eval/malow-text-summary-30s-20261027) and extract the cache for each benchmark into its respective directory:

```bash
hf download zhuomingliu/testDataset --repo-type dataset \
  --include 'eval/malow-text-summary-30s-20261027/*' \
  --local-dir data/text-summary-download
mkdir -p data/text-summary-cache/ovo data/text-summary-cache/streaming
unzip data/text-summary-download/eval/malow-text-summary-30s-20261027/ovo-text-cache.zip \
  -d data/text-summary-cache/ovo
unzip data/text-summary-download/eval/malow-text-summary-30s-20261027/streaming-text-cache.zip \
  -d data/text-summary-cache/streaming
```

Use the prepared datasets from sections 4–5, with unchanged annotations
and video contents. Replay checks their compatibility with the cache. The commands
reuse saved summaries while rebuilding visual memory and generating new answers;
the cache does not contain model weights, videos, or saved answers.

```bash
WORKERS=4
python -m malow.video.eval --benchmark ovo \
  --checkpoint checkpoints/MaLoW-Qwen3.5-9B \
  --base-model checkpoints/Qwen3.5-9B \
  --data-root data/ovo-annotations \
  --video-root data/ovobench-2fps-min64/chunked_videos \
  --output-dir outputs/ovo-9b-text-summary-30s-replay \
  --num-processes "$WORKERS" --offline \
  --eval-mode fixed-gap --rollout-step-seconds 30 \
  --text-summary --checkpoint-progress \
  --text-cache-source data/text-summary-cache/ovo --text-cache-mode summaries

python -m malow.video.eval --benchmark streaming \
  --checkpoint checkpoints/MaLoW-Qwen3.5-9B \
  --base-model checkpoints/Qwen3.5-9B \
  --data-root data/streamingbench-2fps-min64/annotations \
  --video-root data/streamingbench-2fps-min64/videos \
  --output-dir outputs/streaming-9b-text-summary-30s-replay \
  --num-processes "$WORKERS" --offline \
  --eval-mode fixed-gap --rollout-step-seconds 30 \
  --text-summary --checkpoint-progress --streaming-group-by-video \
  --text-cache-source data/text-summary-cache/streaming --text-cache-mode summaries
```

Use a fresh output directory for each new experiment. Append `--dry-run` to inspect
the resolved command without loading models. Here, `--offline` prevents network
model downloads; it does not select the evaluation mode.


## 8. Results and scoring

Scores are percentages. Each evaluation writes predictions and a `summary.json`
under its output directory; StreamingBench also writes per-task `stats.json` files.

### OVO-Bench

| Model | Backward | Real&#8209;time | Forward | Overall |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5&#8209;9B (last 2s) | 51.2 | 75.5 | 40.0 | 55.6 |
| Qwen3.5&#8209;9B (last 2s) with text summary | 59.4 | 78.1 | 42.1 | 59.7 |
| MaLoW&nbsp;(Qwen3.5&#8209;9B) | 66.4 | 76.2 | 51.5 | 64.6 |
| MaLoW&nbsp;(Qwen3.5&#8209;9B) with text summary | 69.7 | 77.3 | 52.2 | 66.4 |

### StreamingBench

| Model | Real&#8209;time | Omni&#8209;source | Proactive | Sequential&nbsp;QA | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5&#8209;9B (last 2s) | 82.0 | 61.0 | 38.8 | 58.0 | 71.3 |
| Qwen3.5&#8209;9B (last 2s) with text summary | 83.1 | 62.6 | 39.6 | 57.2 | ≈72.4† |
| MaLoW&nbsp;(Qwen3.5&#8209;9B) | 82.2 | 62.1 | 58.8 | 63.6 | 73.1 |
| MaLoW&nbsp;(Qwen3.5&#8209;9B) with text summary | 83.6 | 61.2 | 60.0 | 60.4 | 73.6 |

OVO averages subtask accuracies within each split, then averages the three splits. StreamingBench uses **question-weighted accuracy** with weights real:omni:proactive:SQA = **2500:1500:250:250**. Its `weighted_average.accuracy_percent` is the release metric.

† The Qwen3.5-9B with text summary StreamingBench overall score is approximately 72.4,
recalculated from the displayed task scores using the weights above.
