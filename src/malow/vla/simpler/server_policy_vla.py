#!/usr/bin/env python
"""Websocket policy server for Hippocampus VLA checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


from malow.vla.simpler.tools.websocket_policy_server import WebsocketPolicyServer
from malow.vla.modeling import VLAModelWrapper
from malow.vla.constants import VLA_ADAPTER_DIR, VLA_CONFIG_NAME, VLA_TRAINABLE_WEIGHTS_NAME


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


def _read_adapter_base_model(adapter_dir: Path) -> str | None:
    adapter_config = adapter_dir / "adapter_config.json"
    if not adapter_config.is_file():
        return None

    with open(adapter_config, "r", encoding="utf-8") as f:
        data = json.load(f)
    value = data.get("base_model_name_or_path")
    return str(value) if value else None


def _load_processor(checkpoint_dir: Path, model_name_or_path: str):
    for path in (checkpoint_dir, Path(model_name_or_path)):
        if path.exists():
            try:
                processor = AutoProcessor.from_pretrained(str(path), trust_remote_code=True)
                if hasattr(processor, "tokenizer"):
                    processor.tokenizer.padding_side = "left"
                    processor.tokenizer.init_kwargs["padding_side"] = "left"
                return processor
            except Exception:
                logging.info("Processor is not loadable from %s; trying next source.", path)

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.init_kwargs["padding_side"] = "left"
    return processor


def load_hippo_vla_checkpoint(
    checkpoint_path: str,
    model_name_or_path: str | None = None,
    use_bf16: bool = True,
) -> VLAModelWrapper:
    checkpoint_dir = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    config_path = checkpoint_dir / VLA_CONFIG_NAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing {VLA_CONFIG_NAME} in {checkpoint_dir}")

    adapter_dir = checkpoint_dir / VLA_ADAPTER_DIR
    if model_name_or_path is None:
        model_name_or_path = _read_adapter_base_model(adapter_dir)
    if model_name_or_path is None:
        model_name_or_path = "Qwen/Qwen3.5-2B"
        logging.warning("Could not infer base model from adapter config; using %s", model_name_or_path)

    dtype = torch.bfloat16 if use_bf16 else torch.float32
    logging.info("Loading processor from checkpoint/base model.")
    processor = _load_processor(checkpoint_dir, model_name_or_path)

    logging.info("Loading base VLM: %s", model_name_or_path)
    qwen_model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if adapter_dir.is_dir():
        logging.info("Loading VLM LoRA adapter: %s", adapter_dir)
        qwen_model = PeftModel.from_pretrained(qwen_model, str(adapter_dir), is_trainable=False)
    else:
        raise FileNotFoundError(f"Missing required {VLA_ADAPTER_DIR} in {checkpoint_dir}")

    vla_config = OmegaConf.load(config_path)
    model = VLAModelWrapper(qwen_model=qwen_model, processor=processor, vla_config=vla_config)

    trainable_path = checkpoint_dir / VLA_TRAINABLE_WEIGHTS_NAME
    if trainable_path.is_file():
        logging.info("Loading VLA trainable weights: %s", trainable_path)
        state_dict = torch.load(trainable_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        serious_unexpected = [name for name in unexpected if not name.startswith("qwen_model.")]
        if serious_unexpected:
            raise RuntimeError(f"Unexpected VLA checkpoint keys: {serious_unexpected[:10]}")
        missing_vla = [name for name in missing if not name.startswith("qwen_model.")]
        if missing_vla:
            raise RuntimeError(f"Missing action checkpoint weights: {missing_vla[:10]}")
    else:
        raise FileNotFoundError(f"Missing required {VLA_TRAINABLE_WEIGHTS_NAME} in {checkpoint_dir}")

    model.to(dtype=dtype)
    return model


def main(args: argparse.Namespace) -> None:
    configure_reproducibility(args.seed, args.deterministic)

    model = load_hippo_vla_checkpoint(
        checkpoint_path=args.ckpt_path,
        model_name_or_path=args.model_name_or_path,
        use_bf16=args.use_bf16,
    )
    model.to(args.device).eval()

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = WebsocketPolicyServer(
        policy=model,
        host=args.host,
        port=args.port,
        metadata={"env": "simpler_env", "model_type": "hippocampus_vla"},
        output_prior_action=args.output_prior_action,
    )
    logging.info("Server running with Hippocampus VLA model...")
    server.serve_forever()


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hippocampus VLA websocket policy server")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to a Hippocampus VLA checkpoint directory")
    parser.add_argument(
        "--model_name_or_path",
        "--model-name-or-path",
        dest="model_name_or_path",
        type=str,
        default=None,
        help="Optional base VLM path/name override",
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
