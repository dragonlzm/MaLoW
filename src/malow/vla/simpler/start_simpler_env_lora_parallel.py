"""
start_simpler_env_lora_parallel.py

Parallel evaluation script for SimplerEnv benchmark using InternVLA M1 with LoRA adapters.
This script distributes tasks across multiple GPUs for faster evaluation.

Usage:
    # Single worker (specify which tasks to evaluate)
    python start_simpler_env_lora_parallel.py \
        --ckpt-path /path/to/ckpt \
        --port 10093 \
        --worker-id 0 \
        --num-workers 4 \
        --env-name StackGreenCubeOnYellowCubeBakedTexInScene-v0 \
        ...

    # Or use the shell script to launch all workers automatically
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
import os
import pathlib
import random
import sys
from pathlib import Path

import numpy as np


def configure_reproducibility(seed: int | None, deterministic: bool) -> None:
    if seed is None and deterministic:
        seed = 0
    if seed is None:
        return

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
        os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            cuda_backend = getattr(torch.backends, "cuda", None)
            cuda_matmul = getattr(cuda_backend, "matmul", None)
            if hasattr(cuda_matmul, "allow_tf32"):
                cuda_matmul.allow_tf32 = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
    except ImportError:
        pass

    if "tensorflow" in sys.modules:
        try:
            sys.modules["tensorflow"].random.set_seed(seed)
        except AttributeError:
            pass

    logging.getLogger(__name__).info("Reproducibility configured: seed=%s, deterministic=%s", seed, deterministic)


_pre_parser = argparse.ArgumentParser(add_help=False)
_pre_parser.add_argument("--simpler-env-path", type=str, default=os.environ.get("SIMPLER_ENV_PATH"))
_pre_parser.add_argument("--seed", type=int, default=None)
_pre_parser.add_argument("--deterministic", action="store_true")
_pre_args, _ = _pre_parser.parse_known_args()
configure_reproducibility(_pre_args.seed, _pre_args.deterministic)
if _pre_args.simpler_env_path:
    simpler_env_path = Path(_pre_args.simpler_env_path).expanduser().resolve()
    os.environ["SIMPLER_ENV_PATH"] = str(simpler_env_path)
    os.environ["SimplerEnv_PATH"] = str(simpler_env_path)
    sys.path.insert(0, str(simpler_env_path))

from sapien.core import Pose
from transforms3d.euler import euler2quat

from simpler_env.utils.io import DictAction
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from malow.vla.simpler.model2simpler_interface_lora import M1InferenceLora
from malow.vla.simpler.model2simpler_interface_c2l_lora import M1InferenceC2LLora


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-model",
        type=str,
        default="rt1",
        help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'",
    )
    parser.add_argument(
        "--policy-setup",
        type=str,
        default="google_robot",
        help="Policy model setup; e.g., 'google_robot', 'widowx_bridge'",
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument(
        "--additional-env-save-tags",
        type=str,
        default=None,
        help="Additional tags to save the environment eval results",
    )
    parser.add_argument("--scene-name", type=str, default="google_pick_coke_can_1_v4")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--robot", type=str, default="google_robot_static")
    parser.add_argument(
        "--obs-camera-name",
        type=str,
        default=None,
        help="Obtain image observation from this camera for policy input. None = default",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--action-ensemble-horizon", dest="action_ensemble_horizon", type=int, default=None)
    parser.add_argument("--control-freq", type=int, default=3)
    parser.add_argument("--sim-freq", type=int, default=513)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--rgb-overlay-path", type=str, default=None)
    parser.add_argument(
        "--robot-init-x-range",
        type=float,
        nargs=3,
        default=[0.35, 0.35, 1],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--robot-init-y-range",
        type=float,
        nargs=3,
        default=[0.20, 0.20, 1],
        help="[ymin, ymax, num]",
    )
    parser.add_argument(
        "--robot-init-rot-quat-center",
        type=float,
        nargs=4,
        default=[1, 0, 0, 0],
        help="[x, y, z, w]",
    )
    parser.add_argument(
        "--robot-init-rot-rpy-range",
        type=float,
        nargs=9,
        default=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        help="[rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]",
    )
    parser.add_argument(
        "--obj-variation-mode",
        type=str,
        default="xy",
        choices=["xy", "episode"],
        help="Whether to vary the xy position of a single object, or to vary predetermined episodes",
    )
    parser.add_argument("--obj-episode-range", type=int, nargs=2, default=[0, 60], help="[start, end]")
    parser.add_argument(
        "--obj-init-x-range",
        type=float,
        nargs=3,
        default=[-0.35, -0.12, 5],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--obj-init-y-range",
        type=float,
        nargs=3,
        default=[-0.02, 0.42, 5],
        help="[ymin, ymax, num]",
    )

    parser.add_argument(
        "--additional-env-build-kwargs",
        nargs="+",
        action=DictAction,
        help="Additional env build kwargs in xxx=yyy format. If the value "
        'is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument("--logging-dir", type=str, default="./results")
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit")
    parser.add_argument("--octo-init-rng", type=int, default=0, help="Octo init rng seed")
    parser.add_argument("--seed", type=int, default=None, help="Seed Python, NumPy, torch, and TensorFlow RNGs")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic library settings in addition to seeding RNGs",
    )
    parser.add_argument("--async-freq", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--simpler-env-path", type=str, default=os.environ.get("SIMPLER_ENV_PATH"))
    parser.add_argument(
        "--output-prior-action",
        dest="output_prior_action",
        action="store_true",
        help="Use prior action from VLM instead of flow model output",
    )
    parser.add_argument(
        "--interactive-c2l",
        dest="interactive_c2l",
        action="store_true",
        default=os.environ.get("SIMPLER_INTERACTIVE_C2L", "0") == "1",
        help=(
            "Evaluate an interactive Context-to-LoRA VLA server: forwards the per-episode "
            "reset so the server's streaming context does not leak between episodes."
        ),
    )

    parser.add_argument(
        "--c2l-action-feedback",
        dest="c2l_action_feedback",
        action="store_true",
        default=os.environ.get("VLA_C2L_ACTION_FEEDBACK", "0").strip().lower() in {"1", "true", "yes", "on"},
        help=(
            "Interactive C2L only: send the executed (post-ensemble) action back to the "
            "server so the next step's LoRA is generated from what the environment actually "
            "did, instead of this step's unensembled prediction. Off by default. The server "
            "must be started with --c2l_action_feedback as well; the client checks the "
            "server's advertised metadata and refuses to run if it is not."
        ),
    )

    # Parallel evaluation parameters
    parser.add_argument("--worker-id", type=int, default=0, help="Worker ID (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1, help="Total number of workers")
    parser.add_argument(
        "--results-dir", type=str, default="./results/simpler_parallel", help="Directory to save worker results"
    )
    parser.add_argument(
        "--aggregate-only", action="store_true", help="Only aggregate results from workers, don't run evaluation"
    )

    args = parser.parse_args()

    # env args: robot pose
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                args.robot_init_quats.append((Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q)
    # env args: object position
    if args.obj_variation_mode == "xy":
        args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
    # update logging info (args.additional_env_save_tags) if using a different camera from default
    if args.obs_camera_name is not None:
        if args.additional_env_save_tags is None:
            args.additional_env_save_tags = f"obs_camera_{args.obs_camera_name}"
        else:
            args.additional_env_save_tags = args.additional_env_save_tags + f"_obs_camera_{args.obs_camera_name}"

    return args


def eval_simpler_parallel(args) -> dict:
    """
    Evaluate SimplerEnv tasks assigned to this worker.

    Returns:
        Dictionary with evaluation results for this worker
    """
    logging.info(f"Worker {args.worker_id}/{args.num_workers} starting...")

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    output_prior_action = getattr(args, "output_prior_action", False)

    # The interactive C2L policy forwards the per-episode reset to the server and can
    # optionally feed the executed action back; otherwise the two take identical
    # constructor arguments.
    interactive_c2l = getattr(args, "interactive_c2l", False)
    interface_cls = M1InferenceC2LLora if interactive_c2l else M1InferenceLora
    interface_kwargs = dict(
        policy_ckpt_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,
        action_ensemble_horizon=args.action_ensemble_horizon,
        output_prior_action=output_prior_action,
    )
    if interactive_c2l:
        interface_kwargs["action_feedback"] = getattr(args, "c2l_action_feedback", False)
    elif getattr(args, "c2l_action_feedback", False):
        # Silently dropping it would score the stateless policy and label it a feedback run.
        raise ValueError(
            "--c2l-action-feedback / VLA_C2L_ACTION_FEEDBACK only applies to the interactive "
            "C2L policy; pass --interactive-c2l (or SIMPLER_INTERACTIVE_C2L=1) as well."
        )
    model = interface_cls(**interface_kwargs)

    success_arr = maniskill2_evaluator(model, args)
    avg_success = float(np.mean(success_arr))

    worker_result = {
        "worker_id": args.worker_id,
        "env_name": args.env_name,
        "success_array": success_arr.tolist() if hasattr(success_arr, "tolist") else list(success_arr),
        "avg_success": avg_success,
        "num_episodes": len(success_arr),
    }

    # Save worker results
    pathlib.Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    result_file = pathlib.Path(args.results_dir) / f"worker_{args.worker_id}_{args.env_name}_results.json"
    with open(result_file, "w") as f:
        json.dump(worker_result, f, indent=2)

    logging.info(f"Worker {args.worker_id} completed. Results saved to {result_file}")
    logging.info(f"Worker {args.worker_id} - Env: {args.env_name}, Average success: {avg_success:.4f}")

    return worker_result


def aggregate_results(results_dir: str, num_workers: int, env_names: list[str] = None) -> dict:
    """
    Aggregate results from all workers.

    Args:
        results_dir: Path where worker result files are stored
        num_workers: Number of workers to aggregate from
        env_names: List of environment names to aggregate (optional)

    Returns:
        Aggregated results dictionary
    """
    results_path = pathlib.Path(results_dir)
    all_results = []

    # Find all worker result files
    for result_file in results_path.glob("worker_*_results.json"):
        with open(result_file, "r") as f:
            worker_result = json.load(f)
        all_results.append(worker_result)
        logging.info(f"Loaded results from {result_file}")

    if not all_results:
        logging.warning("No result files found!")
        return {}

    # Group by environment
    env_results = {}
    for result in all_results:
        env_name = result["env_name"]
        if env_name not in env_results:
            env_results[env_name] = {
                "success_arrays": [],
                "avg_successes": [],
            }
        env_results[env_name]["success_arrays"].extend(result["success_array"])
        env_results[env_name]["avg_successes"].append(result["avg_success"])

    # Calculate overall statistics per environment
    aggregated = {
        "per_env_results": {},
        "overall_avg_success": 0.0,
    }

    total_sum = 0.0
    total_count = 0

    for env_name, data in env_results.items():
        all_successes = data["success_arrays"]
        env_avg = float(np.mean(all_successes)) if all_successes else 0.0
        aggregated["per_env_results"][env_name] = {
            "avg_success": env_avg,
            "num_episodes": len(all_successes),
            "worker_avgs": data["avg_successes"],
        }
        total_sum += env_avg
        total_count += 1

    aggregated["overall_avg_success"] = total_sum / total_count if total_count > 0 else 0.0

    # Save aggregated results
    agg_file = results_path / "aggregated_results.json"
    with open(agg_file, "w") as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    logging.info(f"\n{'=' * 60}")
    logging.info(f"AGGREGATED RESULTS")
    logging.info(f"{'=' * 60}")
    for env_name, data in aggregated["per_env_results"].items():
        logging.info(f"  {env_name}: {data['avg_success']:.4f} ({data['num_episodes']} episodes)")
    logging.info(f"{'=' * 60}")
    logging.info(f"Overall average success: {aggregated['overall_avg_success']:.4f}")
    logging.info(f"Results saved to {agg_file}")

    return aggregated


if __name__ == "__main__":
    args = get_args()
    configure_reproducibility(args.seed, args.deterministic)

    if args.aggregate_only:
        aggregate_results(args.results_dir, args.num_workers)
    else:
        result = eval_simpler_parallel(args)
        print(args)
        print(" " * 10, "Average success", result["avg_success"])
