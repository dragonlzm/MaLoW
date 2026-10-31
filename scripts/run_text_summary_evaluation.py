#!/usr/bin/env python3
"""Supervise the 9B text-summary experiment and restore the existing GPU burner.

Run with the prepared GPU Python, from any directory. --smoke-only checks both
conditions on both benchmarks. After inspecting those outputs, --skip-smoke runs
the four full evaluations. Status and logs persist independently of the terminal.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ROOT / "outputs/reproduction/evaluations"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def now():
    return datetime.now(timezone.utc).isoformat()


def burner_pids(binary):
    found = []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            if (proc / "exe").resolve() == binary.resolve():
                found.append(int(proc.name))
        except (FileNotFoundError, PermissionError, OSError):
            continue
    return found


def gpu_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    return {int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()}


def stop_burner(binary):
    pids = set(burner_pids(binary))
    unrelated = gpu_processes() - pids
    if unrelated:
        raise RuntimeError(f"GPUs have unrelated workloads; refusing to interrupt PIDs {sorted(unrelated)}")
    for pid in pids:
        # Recheck identity immediately before signalling; never use broad pkill.
        try:
            if Path(f"/proc/{pid}/exe").resolve() == binary.resolve():
                os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 30
    while gpu_processes():
        if time.monotonic() > deadline:
            raise RuntimeError("GPU burner did not release the devices within 30 seconds")
        time.sleep(1)
    return sorted(pids)


def command(benchmark, output, workers, summary, smoke, rollout_step_seconds=10, checkpoint_progress=False, streaming_group_by_video=False):
    if benchmark == "ovo":
        data = "data/ovo-annotations"
        videos = "data/ovobench-2fps-min64/chunked_videos"
    else:
        data = "data/streamingbench-2fps-min64/annotations"
        videos = "data/streamingbench-2fps-min64/videos"
    cmd = [sys.executable, "-m", "malow.video.eval", "--benchmark", benchmark,
           "--checkpoint", str(ROOT / "checkpoints/MaLoW-Qwen3.5-9B"),
           "--base-model", str(ROOT / "checkpoints/Qwen3.5-9B"),
           "--data-root", str(ROOT / data), "--video-root", str(ROOT / videos),
           "--output-dir", str(output), "--num-processes", str(workers),
           "--eval-mode", "fixed-gap", "--rollout-step-seconds", str(rollout_step_seconds), "--offline"]
    if summary:
        cmd.append("--text-summary")
    if checkpoint_progress:
        cmd.append("--checkpoint-progress")
    if streaming_group_by_video and benchmark == "streaming":
        cmd.append("--streaming-group-by-video")
    if smoke and not (streaming_group_by_video and benchmark == "streaming"):
        cmd += ["--limit", "1"]
    return cmd


def metrics(path, benchmark):
    name = "summary_scores.json" if benchmark == "ovo" else "summary.json"
    value = json.loads((path / name).read_text())
    if benchmark == "ovo":
        return {"overall": value["average"], "questions": value["num_prediction_rows"],
                **{key: item["average"] for key, item in value["tasks"].items()}}
    return {"overall": value["weighted_average"]["accuracy_percent"],
            # The benchmark's weighted total is always 4,500, even for a smoke
            # subset. Completeness must use the actual scored examples instead.
            "questions": sum(value[key]["num_examples"] for key in ("real", "omni", "sqa", "proactive")),
            **{key: value[key]["accuracy_percent"] for key in ("real", "omni", "sqa", "proactive")}}


def prepare_smoke_annotations(root, grouped=False):
    """Keep complete real entries; avoid a ten-minute first SQA entry for preflight."""
    source = ROOT / "data/streamingbench-2fps-min64/annotations"
    destination = root / "smoke_annotations"
    destination.mkdir()
    selection = {}
    def seconds(timestamp):
        total = 0.0
        for part in str(timestamp).split(":"):
            total = total * 60 + float(part)
        return total
    for name in ("questions_real.json", "questions_omni.json", "questions_sqa_stream.json", "questions_proactive.json"):
        entries = json.loads((source / name).read_text())
        index = 0
        if name == "questions_sqa_stream.json":
            candidates = [(max(seconds(q["time_stamp"]) for q in entry["questions"]), i)
                          for i, entry in enumerate(entries) if len(entry["questions"]) >= 2]
            _, index = min((end, i) for end, i in candidates if end >= 30)
        selected = [index]
        if grouped and name in ("questions_real.json", "questions_omni.json"):
            by_video = {}
            for i, entry in enumerate(entries):
                end = max(seconds(q["time_stamp"]) for q in entry["questions"])
                by_video.setdefault(entry["video_path"], []).append((end, i))
            candidates = [sorted(values)[:2] for values in by_video.values() if len(values) >= 2]
            selected = [i for _, i in min(candidates, key=lambda pair: (pair[-1][0], pair[0][1]))]
        write_json(destination / name, [entries[i] for i in selected])
        selection[name] = {"source_entry_index": selected[0], "source_entry_indices": selected, "video": entries[selected[0]]["video_path"]}
    write_json(root / "smoke_selection.json", selection)
    return destination


def prepare_ovo_smoke_annotations(root):
    """Choose short complete documents, retaining every round of a forward task."""
    import pyarrow.parquet as pq
    destination = root / "ovo_smoke_annotations"
    selection = {}
    for split in ("backward", "realtime", "forward"):
        table = pq.read_table(ROOT / f"data/ovo-annotations/{split}/test-00000-of-00001.parquet")
        choices = []
        for index, doc in enumerate(table.to_pylist()):
            if split == "forward":
                rounds = [json.loads(item) if isinstance(item, str) else item for item in doc["test_info"]]
                if len(rounds) < 2:
                    continue
                end = max(float(item["realtime"]) for item in rounds)
            else:
                end = float(doc["realtime"])
            if end >= 20:
                choices.append((end, index))
        end, index = min(choices)
        path = destination / split / "test-00000-of-00001.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(table.take([index]), path)
        selection[split] = {"source_entry_index": index, "last_annotated_time": end}
    write_json(root / "smoke_ovo_selection.json", selection)
    return destination


def memory_stats(output):
    result = {"captions": 0, "summaries": 0, "answers": 0, "invalid_attempts": 0,
              "normalized_summaries": 0,
              "caption_seconds": 0.0, "compression_seconds": 0.0,
              "summary_tokens_total": 0, "summary_tokens_max": 0}
    for path in output.glob("text_summary_rank_*.jsonl"):
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row["kind"] == "caption":
                    result["captions"] += 1
                    result["caption_seconds"] += row["seconds"]
                elif row["kind"] == "summary":
                    result["summaries"] += 1
                    result["normalized_summaries"] += bool(row.get("normalizations"))
                    result["compression_seconds"] += row["seconds"]
                    result["summary_tokens_total"] += row["tokens"]
                    result["summary_tokens_max"] = max(result["summary_tokens_max"], row["tokens"])
                elif row["kind"] == "invalid_summary":
                    result["invalid_attempts"] += 1
                elif row["kind"] in ("external_caption", "external_summary", "external_cache_miss"):
                    key = row["kind"] + "_count"
                    result[key] = result.get(key, 0) + 1
                elif row["kind"] in ("cached_caption", "cached_summary"):
                    key = row["kind"] + "_reuses"
                    result[key] = result.get(key, 0) + 1
                elif row["kind"] == "answer":
                    result["answers"] += 1
    recovery = output / "recovery"
    if recovery.exists():
        captions = [json.loads(p.read_text())["value"] for p in recovery.glob("*/caption/*.json")]
        summaries = [json.loads(p.read_text())["value"] for p in recovery.glob("*/summary/*.json")]
        result.update(captions=len(captions), summaries=len(summaries),
                      caption_seconds=sum(r["seconds"] for r in captions),
                      compression_seconds=sum(r["seconds"] for r in summaries),
                      summary_tokens_total=sum(r["tokens"] for r in summaries),
                      summary_tokens_max=max((r["tokens"] for r in summaries), default=0),
                      normalized_summaries=sum(bool(r.get("normalizations")) for r in summaries))
        prediction_files = list(output.glob("rank_*_predictions.jsonl")) + list(output.glob("*/rank_*.jsonl"))
        result["answers"] = sum(len(p.read_text().splitlines()) for p in prediction_files)
    return result


def validate_completed_job(job, workers):
    """Validate identity and the exact durable prediction set, not just score counts."""
    sys.path.insert(0, str(ROOT / "src"))
    from argparse import Namespace
    from malow.video.interactive_options import guard_interactive_run
    from malow.video.progress import read_journal, ovo_key, validate_video_inputs
    output = Path(job["output"])
    validate_video_inputs(output)
    manifest = json.loads((output / "interactive_run_config.json").read_text())
    # Recreate worker arguments only; derived identity fields are recomputed.
    derived = {"world_size", "interactive_sample_fps", "interactive_format_version", "text_summary_settings",
               "video_environment", "checkpoint_metadata_sha256", "checkpoint_files", "streaming_grouping_version",
               "recovery_version", "recovery_implementation_sha256", "annotation_fingerprints", "model_file_identities"}
    args = Namespace(**{key: value for key, value in manifest.items() if key not in derived}, output_dir=str(output))
    saved_env = {key: os.environ.get(key) for key in manifest["video_environment"]}
    try:
        os.environ.update(manifest["video_environment"])
        guard_interactive_run(args, workers)
    finally:
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    if job["benchmark"] == "ovo":
        from malow.video.ovo import build_eval_units, normalize_task_list
        units = build_eval_units(normalize_task_list(args.tasks), Path(args.video_root), args.limit,
                                 no_video=False, annotation_root=args.annotation_root)
        expected = {(u.task_name, u.doc['id'], u.round_idx) for u in units}
        rows = [row for rank in range(workers) for row in read_journal(output / f"rank_{rank}_predictions.jsonl")]
        actual = [ovo_key(row) for row in rows]
    else:
        from malow.video.streaming import resolve_task_files, entry_questions, question_id
        expected, actual = set(), []
        for task, annotation_file in resolve_task_files(args).items():
            entries = json.loads(annotation_file.read_text())
            if args.limit is not None:
                entries = entries[:args.limit]
            expected.update(question_id(task, i, j, q) for i, entry in enumerate(entries)
                            for j, q in enumerate(entry_questions(entry)))
            actual.extend(row['question_id'] for rank in range(workers)
                          for row in read_journal(output / task / f"rank_{rank:05d}.jsonl"))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError(f"Incomplete or duplicate predictions: {len(actual)} rows for {len(expected)} expected IDs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true", help="Resume a saved run using its recorded settings.")
    parser.add_argument("--text-cache-root", help="Directory containing exported ovo/ and streaming/ bundles; run LoRA replay then base-only replay.")
    parser.add_argument("--text-cache-mode", choices=("summaries", "captions"), default="summaries")
    parser.add_argument("--checkpoint-progress", action="store_true")
    parser.add_argument("--streaming-group-by-video", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--rollout-step-seconds", type=int, default=10,
                        help="Fixed-gap interval for both summary and matched control runs (default: 10).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--smoke-only", action="store_true")
    group.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-summary-benchmarks", nargs="+", choices=("streaming", "ovo"),
                        default=["streaming", "ovo"], help="Select summary smoke cases; controls still cover both benchmarks.")
    parser.add_argument("--burner", type=Path, default=Path("/fsx/linz/env/gpu-burn/gpu_burn"))
    args = parser.parse_args()
    if args.text_cache_root:
        args.text_cache_root = str(Path(args.text_cache_root).resolve())
        args.checkpoint_progress = True
        args.streaming_group_by_video = True
    if not args.run_id.replace("-", "").replace("_", "").isalnum() or args.workers < 1:
        parser.error("Use an alphanumeric run ID (hyphens/underscores allowed) and positive workers")
    if args.rollout_step_seconds < 1:
        parser.error("--rollout-step-seconds must be positive")
    root = ROOT / "outputs/text_sum_evaluation" / args.run_id
    if args.resume and not (root / "status.json").is_file():
        parser.error("Resume requires an existing run with status.json")
    root.mkdir(parents=True, exist_ok=args.resume)
    lock = (root.parent / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    setting_names = ("workers", "rollout_step_seconds", "smoke_only", "skip_smoke",
                     "smoke_summary_benchmarks", "burner", "checkpoint_progress", "streaming_group_by_video", "text_cache_root", "text_cache_mode")
    if args.resume:
        state = json.loads((root / "status.json").read_text())
        if "settings" not in state or not state["settings"].get("checkpoint_progress"):
            raise ValueError("This run has no durable recovery metadata; use a fresh run ID")
        for name in setting_names:
            saved = state["settings"].get(name, getattr(args, name))
            flag = "--" + name.replace("_", "-")
            if any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
                supplied = str(getattr(args, name)) if name == "burner" else getattr(args, name)
                if supplied != saved:
                    raise ValueError(f"Cannot change {flag} while resuming")
            setattr(args, name, Path(saved) if name == "burner" else saved)
    else:
        state = {"started": now(), "jobs": [], "settings": {
            name: str(getattr(args, name)) if name == "burner" else getattr(args, name) for name in setting_names}}
    if args.text_cache_root:
        if not args.skip_smoke:
            parser.error("Cached replay queues require --skip-smoke; run focused replay checks first")
        for benchmark in ("ovo", "streaming"):
            if not (Path(args.text_cache_root) / benchmark / "manifest.json").is_file():
                parser.error(f"Missing {benchmark} text-cache manifest")
    if not args.burner.is_file():
        parser.error("GPU burner binary is missing")
    attempt = len(state.get("attempts", [])) + 1
    state.setdefault("attempts", []).append({"number": attempt, "started": now(), "pid": os.getpid()})
    state.update(pid=os.getpid(), status="preflight")
    for field in ("error", "finished", "burner_error", "burner_running", "burner_pid"):
        state.pop(field, None)
    child = None
    restore_burner = False

    def save():
        write_json(root / "status.json", state)

    def interrupted(signum, frame):
        raise RuntimeError(f"Supervisor interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save()
    try:
        # Refuse unrelated GPU workloads before taking responsibility for restoration.
        if gpu_processes() - set(burner_pids(args.burner)):
            raise RuntimeError("Unrelated GPU jobs are active")
        restore_burner = True
        state["stopped_burner_pids"] = stop_burner(args.burner)
        phases = ([True] if args.smoke_only else [False] if args.skip_smoke else [True, False])
        smoke_annotations = ovo_smoke_annotations = None
        if True in phases:
            smoke_annotations = root / "smoke_annotations"
            ovo_smoke_annotations = root / "ovo_smoke_annotations"
            if not smoke_annotations.exists():
                smoke_annotations = (prepare_smoke_annotations(root, grouped=True) if args.streaming_group_by_video
                                     else prepare_smoke_annotations(root))
            if not ovo_smoke_annotations.exists():
                ovo_smoke_annotations = prepare_ovo_smoke_annotations(root)
        for smoke in phases:
            for summary in (True, False):
                for benchmark in ("ovo", "streaming"):
                    if smoke and summary and benchmark not in args.smoke_summary_benchmarks:
                        continue
                    condition = ("lora-text-replay" if summary else "base-text-replay") if args.text_cache_root else ("text-summary" if summary else "lora-control")
                    label = f"{'smoke-' if smoke else ''}released-9b-{benchmark}-{condition}-{args.rollout_step_seconds}s-{args.run_id}"
                    if args.streaming_group_by_video and benchmark == "streaming":
                        label += "-grouped"
                    output = EVALUATIONS / label
                    cmd = command(benchmark, output, 1 if smoke else args.workers, summary or bool(args.text_cache_root), smoke,
                                  args.rollout_step_seconds, args.checkpoint_progress, args.streaming_group_by_video)
                    if args.text_cache_root:
                        cmd += ["--text-cache-source", str(Path(args.text_cache_root) / benchmark),
                                "--text-cache-mode", args.text_cache_mode,
                                "--text-cache-miss", "error" if summary else "generate"]
                        if not summary:
                            cmd.append("--text-cache-base-only")
                    if smoke and benchmark == "streaming":
                        cmd[cmd.index("--data-root") + 1] = str(smoke_annotations)
                    elif smoke:
                        cmd[cmd.index("--data-root") + 1] = str(ovo_smoke_annotations)
                    previous = next((job for job in state["jobs"] if job["name"] == label), None)
                    if previous is not None and previous["command"] != cmd:
                        raise ValueError(f"Saved command changed for {label}")
                    if previous is not None and previous["status"] == "complete":
                        validate_completed_job(previous, 1 if smoke else args.workers)
                        previous["metrics"] = metrics(output, benchmark)
                        save()
                        continue
                    job = {"name": label, "benchmark": benchmark, "summary": summary or bool(args.text_cache_root),
                           "smoke": smoke, "command": cmd, "output": str(output),
                           "started": now(), "status": "running"}
                    if args.text_cache_root:
                        job["base_only"] = not summary
                    if previous is not None:
                        job["attempt_history"] = previous.get("attempt_history", []) + [
                            {k: v for k, v in previous.items() if k != "attempt_history"}]
                        state["jobs"][state["jobs"].index(previous)] = job
                    else:
                        state["jobs"].append(job)
                    state["status"] = "running"
                    env = os.environ.copy()
                    env["PYTHONUNBUFFERED"] = "1"
                    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
                    env["CUDA_VISIBLE_DEVICES"] = "0" if smoke else ",".join(map(str, range(args.workers)))
                    start = time.monotonic()
                    with (root / f"{label}.attempt-{attempt}.log").open("w") as log:
                        child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        job["pid"] = child.pid
                        save()
                        returncode = child.wait()
                    child = None
                    job.update(returncode=returncode, seconds=time.monotonic() - start, finished=now(),
                               status="validating" if returncode == 0 else "failed")
                    save()
                    if returncode:
                        raise RuntimeError(f"{label} exited with code {returncode}")
                    if args.checkpoint_progress:
                        validate_completed_job(job, 1 if smoke else args.workers)
                    job["metrics"] = metrics(output, benchmark)
                    if job["summary"]:
                        job["memory"] = memory_stats(output)
                        if not job["memory"]["summaries"] or not job["memory"]["answers"]:
                            raise RuntimeError("Summary condition produced no memory or answer traces")
                    if not smoke and job["metrics"]["questions"] != (3035 if benchmark == "ovo" else 4500):
                        raise RuntimeError("Full evaluation question count differs from the reference run")
                    job["status"] = "complete"
                    save()
        comparisons = {}
        if not args.smoke_only:
            for benchmark in ("ovo", "streaming"):
                jobs = [job for job in state["jobs"] if not job["smoke"] and job["benchmark"] == benchmark]
                experiment, control = jobs
                comparisons[benchmark] = {
                    "summary": experiment["metrics"], "control": control["metrics"],
                    "summary_minus_control_pp": {key: value - control["metrics"][key]
                        for key, value in experiment["metrics"].items() if key != "questions"},
                    "reference_2s": metrics(EVALUATIONS / f"released-9b-{benchmark}-fixed-gap", benchmark),
                }
        if args.text_cache_root:
            for benchmark, comparison in comparisons.items():
                comparison["lora_replay"] = comparison.pop("summary")
                comparison["base_replay"] = comparison.pop("control")
                comparison["lora_minus_base_pp"] = comparison.pop("summary_minus_control_pp")
        write_json(root / "comparison.json", comparisons)
        state["status"] = "complete"
    except BaseException as error:
        state.update(status="failed", error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        state["finished"] = now()
        state["attempts"][-1].update(finished=state["finished"], status=state["status"], error=state.get("error"))
        save()
        if restore_burner:
            # Do not overlap a burner with any surviving worker or unrelated GPU job.
            deadline = time.monotonic() + 60
            while gpu_processes() and time.monotonic() < deadline:
                time.sleep(1)
            if gpu_processes():
                state["burner_error"] = "GPU processes remain; burner was not started"
            else:
                with (root / "burner.log").open("w") as log:
                    burner = subprocess.Popen([str(args.burner), "1209600"], cwd=args.burner.parent,
                                              stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                state["burner_pid"] = burner.pid
                deadline = time.monotonic() + 60
                while burner.poll() is None and len(gpu_processes()) < args.workers and time.monotonic() < deadline:
                    time.sleep(1)
                state["burner_running"] = burner.poll() is None and len(gpu_processes()) >= args.workers
                if not state["burner_running"]:
                    state["burner_error"] = "Burner failed startup verification; inspect burner.log"
            save()
        lock.close()


if __name__ == "__main__":
    main()
