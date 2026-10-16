#!/usr/bin/env python
"""Websocket policy server for interactive Context-to-LoRA VLA checkpoints.

The plain VLA server (`server_policy_vla.py`) is stateless: every request is a fresh
forward over one frame. This one carries an episode's context. The policy object holds a
streaming C2L session, folds each observation into it, regenerates the adapter, and
answers with that step's action -- so the wire protocol is unchanged and only the
`reset` message, sent at every episode boundary, is new.

The stage-1 checkpoint is a required argument, not an inferred one: the frozen base here
must be raw Qwen with *that* checkpoint's LoRA merged in, because that is the model the
hypernet was trained to modulate.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import socket
import sys
from pathlib import Path

import numpy as np
import torch


from malow.vla.simpler.tools.websocket_policy_server import WebsocketPolicyServer
from malow.vla.c2l.blocks import SIMPLER_BRIDGE_STEP_SECONDS
from malow.vla.c2l.loading import load_c2l_vla_model
from malow.vla.c2l.policy import C2LVLAInteractivePolicy
from malow.vla.constants import VLA_ADAPTER_DIR


def env_flag(name: str) -> bool:
    """Read a launcher-set boolean env var, matching the shell launchers' `1` convention."""
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def configure_reproducibility(seed: int | None, deterministic: bool) -> None:
    if seed is None and deterministic:
        seed = 0
    if seed is None:
        return

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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

    logging.getLogger(__name__).info("Reproducibility configured: seed=%s, deterministic=%s", seed, deterministic)


def main(args: argparse.Namespace) -> None:
    configure_reproducibility(args.seed, args.deterministic)

    stage1 = Path(args.stage1_ckpt_path).expanduser().resolve()
    if not (stage1 / VLA_ADAPTER_DIR).is_dir():
        raise FileNotFoundError(
            f"--stage1_ckpt_path {stage1} has no {VLA_ADAPTER_DIR}/. Point it at the VLA "
            "checkpoint this C2L run was trained on top of."
        )

    logging.info("Loading C2L-VLA checkpoint %s on stage-1 base %s", args.ckpt_path, stage1)
    model, processor = load_c2l_vla_model(
        checkpoint_path=args.ckpt_path,
        stage1_checkpoint_path=str(stage1),
        model_name_or_path=args.model_name_or_path,
        device=args.device,
        dtype=torch.bfloat16 if args.use_bf16 else torch.float32,
    )
    logging.info("Interactive C2L clock: %.6f s per env step", args.c2l_step_seconds)
    logging.info("Interactive C2L action feedback: %s", args.c2l_action_feedback)
    policy = C2LVLAInteractivePolicy(
        model=model,
        processor=processor,
        step_seconds=args.c2l_step_seconds,
        action_feedback=args.c2l_action_feedback,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata={
            "env": "simpler_env",
            "model_type": "hippocampus_c2l_vla",
            "interactive": True,
            # Advertised so a client asked for action feedback can verify the server
            # actually honours it. `predict_action(**payload)` ignores unknown keys, so
            # without this handshake a new client against an old server would run happily
            # with the feature silently doing nothing.
            "action_feedback": bool(args.c2l_action_feedback),
        },
        output_prior_action=args.output_prior_action,
    )
    logging.info("Server running with interactive C2L-VLA model...")
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hippocampus interactive C2L-VLA websocket policy server")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Trained C2L-VLA checkpoint directory")
    parser.add_argument(
        "--stage1_ckpt_path",
        "--stage1-ckpt-path",
        dest="stage1_ckpt_path",
        type=str,
        required=True,
        help="VLA checkpoint whose LoRA adapter is merged into the frozen base",
    )
    parser.add_argument(
        "--model_name_or_path",
        "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default=None,
        help="Base VLM path override; defaults to the one recorded in the C2L metadata",
    )
    parser.add_argument(
        "--c2l_step_seconds",
        "--c2l-step-seconds",
        dest="c2l_step_seconds",
        type=float,
        default=float(os.environ.get("VLA_C2L_STEP_SECONDS", SIMPLER_BRIDGE_STEP_SECONDS)),
        help=(
            "Seconds per env step, i.e. the clock the interleaved C2L timestamps are "
            "written on. Defaults to SimplerEnv's WidowX bridge control rate of 5 Hz "
            f"({SIMPLER_BRIDGE_STEP_SECONDS}s); override for a 3 Hz Google-robot setup. "
            "Also reads VLA_C2L_STEP_SECONDS so a launcher can set it without editing "
            "the server command line."
        ),
    )
    parser.add_argument(
        "--c2l_action_feedback",
        "--c2l-action-feedback",
        dest="c2l_action_feedback",
        action="store_true",
        default=env_flag("VLA_C2L_ACTION_FEEDBACK"),
        help=(
            "Fold the action the environment actually executed -- the client-side "
            "ensembler's blend, re-normalized and sent back with the next request -- into "
            "the C2L action history, instead of this step's own unensembled prediction. "
            "Off by default, which keeps the original behaviour. Also reads "
            "VLA_C2L_ACTION_FEEDBACK so a launcher can set it without editing the server "
            "command line."
        ),
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--output_prior_action", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="Seed Python, NumPy, and torch RNGs")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic torch/CUDA settings in addition to seeding RNGs",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main(build_argparser().parse_args())
