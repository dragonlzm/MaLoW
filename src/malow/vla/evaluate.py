"""Run the four Simpler WidowX tasks against independent policy servers."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TASKS = (
    "StackGreenCubeOnYellowCubeBakedTexInScene-v0",
    "PutCarrotOnPlateInScene-v0",
    "PutSpoonOnTableClothInScene-v0",
    "PutEggplantInBasketScene-v0",
)


def build_plan(args):
    root = Path(args.simpler_env).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    selected = getattr(args, "gpus", None)
    if selected is None:
        selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_ids = [v.strip() for v in selected.split(",")] if selected is not None else [str(i) for i in range(args.workers)]
    if len(gpu_ids) < args.workers or any(not gpu or gpu == "-1" for gpu in gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus / CUDA_VISIBLE_DEVICES must contain at least one distinct GPU ID per worker")
    gpu_ids = gpu_ids[:args.workers]
    servers, tasks = [], []
    for worker in range(args.workers):
        seed = args.seed + worker
        cmd = [args.server_python, "-m", "malow.vla.server", "--kind", args.kind,
               "--checkpoint", args.checkpoint, "--base-model", args.base_model,
               "--port", str(args.port + worker), "--seed", str(seed)]
        if args.kind == "c2l":
            cmd += ["--stage1-checkpoint", args.stage1_checkpoint]
        servers.append({"worker": worker, "argv": cmd,
                        "env": {"CUDA_VISIBLE_DEVICES": gpu_ids[worker], "PYTHONHASHSEED": str(seed),
                                "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}})
    for index, name in enumerate(TASKS):
        worker = index % args.workers
        sink = index == 3
        cmd = [args.sim_python, "-m", "malow.vla.simpler.start_simpler_env_lora_parallel",
               "--simpler-env-path", str(root), "--ckpt-path", args.checkpoint,
               "--port", str(args.port + worker), "--robot", "widowx_sink_camera_setup" if sink else "widowx",
               "--policy-setup", "widowx_bridge", "--control-freq", "5", "--sim-freq", "500",
               "--max-episode-steps", "120", "--env-name", name,
               "--scene-name", "bridge_table_1_v2" if sink else "bridge_table_1_v1",
               "--rgb-overlay-path", str(root / "ManiSkill2_real2sim/data/real_inpainting" / ("bridge_sink.png" if sink else "bridge_real_eval_1.png")),
               "--robot-init-x-range", *(["0.127", "0.127", "1"] if sink else ["0.147", "0.147", "1"]),
               "--robot-init-y-range", *(["0.06", "0.06", "1"] if sink else ["0.028", "0.028", "1"]),
               "--obj-variation-mode", "episode", "--obj-episode-range", "0", "24",
               "--robot-init-rot-quat-center", "0", "0", "0", "1",
               "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
               "--action-ensemble-horizon", "4", "--worker-id", str(worker),
               "--num-workers", str(args.workers), "--results-dir", str(output),
               "--logging-dir", str(output / "episodes"),
               "--deterministic", "--seed", str(args.seed + index)]
        if args.kind == "c2l":
            cmd += ["--interactive-c2l", "--c2l-action-feedback"]
        env = {"CUDA_VISIBLE_DEVICES": gpu_ids[worker], "PYTHONHASHSEED": str(args.seed + index),
               "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TF_DETERMINISTIC_OPS": "1",
               "TF_CUDNN_DETERMINISTIC": "1", "DISPLAY": "",
               "SIMPLER_INTERACTIVE_C2L": "1" if args.kind == "c2l" else "0",
               "VLA_C2L_ACTION_FEEDBACK": "1" if args.kind == "c2l" else "0"}
        tasks.append({"worker": worker, "index": index, "env_name": name, "argv": cmd, "env": env})
    return {"kind": args.kind, "seed": args.seed, "workers": args.workers, "gpus": gpu_ids,
            "ensemble_horizon": 4, "step_seconds": 0.2, "episodes_per_task": 24,
            "servers": servers, "tasks": tasks}


def aggregate_results(output, workers):
    """Reject incomplete, duplicated, malformed or partially successful runs."""
    output = Path(output)
    expected = {f"worker_{i % workers}_{name}_results.json" for i, name in enumerate(TASKS)}
    found = {p.name for p in output.glob("worker_*_results.json")}
    if found != expected:
        raise ValueError(f"Expected exactly four task result files; missing={sorted(expected-found)}, extra={sorted(found-expected)}")
    results = {}
    for index, name in enumerate(TASKS):
        worker = index % workers
        data = json.loads((output / f"worker_{worker}_{name}_results.json").read_text())
        outcomes = data.get("success_array")
        if data.get("env_name") != name or data.get("worker_id") != worker:
            raise ValueError(f"Mismatched task identity: {name}")
        if not isinstance(outcomes, list) or len(outcomes) != 24 or data.get("num_episodes") != 24:
            raise ValueError(f"{name}: expected 24 completed episodes")
        if any(type(v) not in (bool, int, float) or not math.isfinite(v) or v not in (0, 1) for v in outcomes):
            raise ValueError(f"{name}: success_array must contain binary outcomes")
        rate = sum(outcomes) / 24
        if not math.isclose(data.get("avg_success", float("nan")), rate, abs_tol=1e-10):
            raise ValueError(f"{name}: stored average disagrees with episode outcomes")
        results[name] = {"successes": int(sum(outcomes)), "episodes": 24, "success_rate": rate}
    return {"per_task": results, "overall": sum(v["success_rate"] for v in results.values()) / 4,
            "total_episodes": 96}


def _stop(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + 10
    for process in processes:
        try:
            process.wait(timeout=max(0.01, deadline-time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run(args, plan):
    output = Path(args.output).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be empty to prevent reuse of stale results")
    if not (Path(args.simpler_env) / "simpler_env/evaluation/maniskill2_evaluator.py").is_file():
        raise FileNotFoundError("--simpler-env must point to a populated SimplerEnv checkout")
    for ckpt in [args.checkpoint] + ([args.stage1_checkpoint] if args.kind == "c2l" else []):
        if not Path(ckpt).is_dir():
            raise FileNotFoundError(ckpt)
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_config.json").write_text(json.dumps(plan, indent=2) + "\n")
    # Fail before starting any server if a requested port is already in use.
    for port in range(args.port, args.port + args.workers):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    processes, handles = [], []
    def launch(item, log):
        handle = (output / log).open("w")
        handles.append(handle)
        env = dict(os.environ, **item["env"])
        env["TRITON_CACHE_DIR"] = str(output / "triton_cache" / f"worker_{item['worker']}")
        # Keep the model runtime isolated from simulator ABI/package constraints.
        env["TOKENIZERS_PARALLELISM"] = "false"
        env["SIMPLER_ENV_PATH"] = str(Path(args.simpler_env).resolve())
        env["SimplerEnv_PATH"] = env["SIMPLER_ENV_PATH"]
        if args.vulkan_icd:
            env["VK_ICD_FILENAMES"] = args.vulkan_icd
        process = subprocess.Popen(item["argv"], env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        processes.append(process)
        return process
    try:
        servers = [launch(item, f"server_{item['worker']}.log") for item in plan["servers"]]
        deadline = time.monotonic() + args.server_timeout
        ready = set()
        while len(ready) < len(servers):
            for index, process in enumerate(servers):
                if process.poll() is not None:
                    raise RuntimeError(f"Policy server {index} exited; inspect server_{index}.log")
                if index not in ready:
                    try:
                        with socket.create_connection(("127.0.0.1", args.port + index), timeout=0.2):
                            ready.add(index)
                    except OSError:
                        pass
            if time.monotonic() >= deadline:
                raise TimeoutError("Policy server readiness timed out")
            time.sleep(0.2)
        queues = {w: [t for t in plan["tasks"] if t["worker"] == w] for w in range(args.workers)}
        active = {}
        while any(queues.values()) or active:
            if any(server.poll() is not None for server in servers):
                raise RuntimeError("A policy server exited during evaluation")
            for worker, queue in queues.items():
                if worker not in active and queue:
                    task = queue.pop(0)
                    active[worker] = (launch(task, f"task_{task['index']}.log"), task)
            for worker, (process, task) in list(active.items()):
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"Task {task['env_name']} failed with exit {code}; inspect task_{task['index']}.log")
                    del active[worker]
            time.sleep(0.2)
        summary = aggregate_results(output, args.workers)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
    finally:
        _stop(processes)
        for handle in handles:
            handle.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("c2l", "baseline"), default="c2l")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stage1-checkpoint")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--simpler-env", required=True)
    p.add_argument("--server-python", default=sys.executable)
    p.add_argument("--sim-python", required=True, help="Interpreter of the separate SimplerEnv environment")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    p.add_argument("--gpus", help="Comma-separated GPU indices or UUIDs; defaults to CUDA_VISIBLE_DEVICES, then 0..workers-1")
    p.add_argument("--port", type=int, default=10100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True)
    p.add_argument("--vulkan-icd", help="NVIDIA Vulkan ICD JSON path, if not configured system-wide")
    p.add_argument("--server-timeout", type=float, default=1800)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if args.kind == "c2l" and not args.stage1_checkpoint:
        p.error("--stage1-checkpoint is required for C2L")
    if not 1 <= args.port <= 65536 - args.workers:
        p.error("Invalid port range")
    for field in ("checkpoint", "stage1_checkpoint", "simpler_env", "output"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, str(Path(value).expanduser().resolve()))
    for field in ("sim_python", "server_python", "vulkan_icd"):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, os.path.expanduser(value))
    if Path(args.base_model).expanduser().exists() or args.base_model.startswith(("/", "~", "./", "../")):
        args.base_model = str(Path(args.base_model).expanduser().resolve())
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    run(args, plan)


if __name__ == "__main__":
    main()
