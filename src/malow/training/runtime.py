"""Isolated C2L worker; invoke with torchrun through the training launcher."""
from __future__ import annotations

import json
import sys


def run_training(config: dict) -> None:
    """Train a resolved LlamaFactory C2L configuration, without other stages."""
    if sys.version_info < (3, 11):
        raise RuntimeError("MaLoW training requires Python 3.11 or newer.")
    config = dict(config)
    if config.get("stage", "c2l") != "c2l":
        raise ValueError("The MaLoW training runtime supports stage=c2l only.")
    if config.get("deepspeed") or config.get("fsdp"):
        raise ValueError("The portable training runtime supports single-GPU and torchrun DDP only.")
    if config.get("push_to_hub"):
        raise ValueError("Publish checkpoints explicitly after reviewing the training output.")
    config.setdefault("stage", "c2l")
    from .checkpointing import training_config_context
    from ._vendor.llamafactory.hparams import get_train_args
    from ._vendor.llamafactory.train.c2l.workflow import run_c2l
    from ._vendor.llamafactory.train.callbacks import LogCallback

    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(config)
    # Parsing may resolve an automatic resume; keep the effective path in provenance.
    config["resume_from_checkpoint"] = training_args.resume_from_checkpoint
    with training_config_context(config):
        run_c2l(model_args, data_args, training_args, finetuning_args, generating_args,
                callbacks=[LogCallback()])


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="Resolved training configuration JSON")
    args = parser.parse_args()
    with open(args.config) as handle:
        run_training(json.load(handle))


if __name__ == "__main__":
    main()
