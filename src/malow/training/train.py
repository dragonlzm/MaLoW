"""Launch one of the six portable, model-verified MaLoW training recipes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .recipes import PRESETS, SOURCE_COMMIT, resolve_recipe


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=PRESETS)
    parser.add_argument("--annotations-root", type=Path)
    parser.add_argument("--replay-manifest", type=Path, help="Frozen reference schedule; replaces annotation sampling.")
    parser.add_argument("--videos-root", required=True, type=Path)
    parser.add_argument("--base-model", help="Matching local snapshot or Hugging Face model ID.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--resume", type=Path, help="A checkpoint-N directory from this training runtime.")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print config and command without downloads or writes.")
    parser.add_argument("--max-steps", type=int, default=2800)
    parser.add_argument("--save-steps", type=int, default=2800)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--gradient-accumulation-steps", type=int,
                        help="Default: 16 for streaming, 64 / num-processes for replay.")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not args.annotations_root and not args.replay_manifest:
        parser.error("--annotations-root is required unless --replay-manifest is supplied")
    if args.dataloader_num_workers < 0:
        parser.error("--dataloader-num-workers must be nonnegative")
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 64 // max(1, args.num_processes) if args.replay_manifest else 16
    for name in ("num_processes", "max_steps", "save_steps", "logging_steps", "gradient_accumulation_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def build_config(args):
    from .data import resolve_training_sources

    profiles, roots = ([], []) if args.replay_manifest else resolve_training_sources(args.annotations_root, args.videos_root)
    # The upstream data parser uses commas as list separators.
    if any("," in str(p) for p in (*profiles, *roots)):
        raise ValueError("Annotation and video paths cannot contain commas.")
    config = resolve_recipe(args.model)
    config.update(dataset=",".join(map(str, profiles)), video_roots=",".join(map(str, roots)),
                  media_dir=str(args.videos_root.expanduser().resolve()),
                  output_dir=str(args.output_dir.expanduser().resolve()),
                  max_steps=args.max_steps, save_steps=args.save_steps, logging_steps=args.logging_steps,
                  gradient_accumulation_steps=args.gradient_accumulation_steps,
                  dataloader_num_workers=args.dataloader_num_workers,
                  dataloader_persistent_workers=args.dataloader_num_workers > 0,
                  resume_from_checkpoint=str(args.resume.expanduser().resolve()) if args.resume else None)
    if args.replay_manifest:
        from .replay import load_manifest, validate_layout, validate_training_contract

        manifest = load_manifest(args.replay_manifest)
        validate_training_contract(manifest, config["c2l_loss_type"], config["base_with_last_2seconds"])
        validate_layout(args.num_processes, 1, args.gradient_accumulation_steps, args.max_steps, manifest)
        config.update(dataset=None, video_roots=None, streaming=False, mix_strategy="concat",
                      interleave_probs=None, c2l_streaming_repeat=False, disable_shuffling=True,
                      replay_manifest=str(args.replay_manifest.expanduser().resolve()),
                      replay_fingerprint=manifest["fingerprint"],
                      accelerator_config={"dispatch_batches": False, "split_batches": False,
                                          "use_seedable_sampler": False})
    if args.base_model:
        config["model_name_or_path"] = str(Path(args.base_model).expanduser().resolve()) if Path(args.base_model).expanduser().is_dir() else args.base_model
    return config


def build_command(args):
    config_path = args.output_dir.expanduser().resolve() / "resolved_config.json"
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc_per_node={args.num_processes}", "--module", "malow.training.worker",
            "--config", str(config_path)]


def launch_manifest(args):
    manifest = {"model": args.model, "source_commit": SOURCE_COMMIT,
            "num_processes": args.num_processes,
            "effective_batch_size": args.num_processes * args.gradient_accumulation_steps,
            "recipe": "model_verified_cross_video_mc_2to1", "reference_save_steps": 2800}
    if args.replay_manifest:
        from .replay import load_manifest

        manifest["replay_fingerprint"] = load_manifest(args.replay_manifest)["fingerprint"]
    return manifest


def validate_resume(args):
    if not args.resume:
        return
    checkpoint = args.resume.expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Missing resume checkpoint: {checkpoint}")
    manifest_path = checkpoint / "launch_manifest.json"
    if not manifest_path.exists():
        manifest_path = checkpoint.parent / "launch_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Resume requires launch_manifest.json in {checkpoint} or its parent")
    old = json.loads(manifest_path.read_text())
    new = launch_manifest(args)
    for name in ("model", "source_commit", "num_processes", "effective_batch_size", "recipe", "replay_fingerprint"):
        if old.get(name) != new.get(name):
            raise ValueError(f"Resume {name} mismatch: saved={old.get(name)!r}, requested={new.get(name)!r}")
    if args.replay_manifest:
        state = json.loads((checkpoint / "trainer_state.json").read_text())
        step = state.get("global_step")
        if type(step) is not int or not 0 <= step < args.max_steps:
            raise ValueError("Replay resume requires a checkpoint before the requested final step.")


def validate_resume_data(args, report):
    """Relocating data is supported; changing training annotation bytes is not."""
    if not args.resume:
        return
    checkpoint = args.resume.expanduser().resolve()
    path = checkpoint / "data_validation.json"
    if not path.is_file():
        path = checkpoint.parent / "data_validation.json"
    if not path.is_file():
        raise ValueError("Resume requires the original data_validation.json report.")
    saved = json.loads(path.read_text())
    if not saved.get("annotation_sha256") or saved["annotation_sha256"] != report.get("annotation_sha256"):
        raise ValueError("Resume annotations differ from the original training dataset.")


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    command = build_command(args)
    manifest = launch_manifest(args)
    if args.dry_run:
        print(json.dumps({"launch": manifest, "config": config, "offline": args.offline}, indent=2))
        print(shlex.join(command))
        return 0
    validate_resume(args)
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError(f"Output directory is not empty: {output}. Choose a new output or use --resume.")
    from .data import validate_training_data

    if args.replay_manifest:
        from .replay import validate

        report = validate(args.replay_manifest, videos_root=args.videos_root)
    else:
        report = validate_training_data(args.annotations_root, args.videos_root, check_videos=True)
    validate_resume_data(args, report)
    env = os.environ.copy()
    env.setdefault("FORCE_QWENVL_VIDEO_READER", "decord")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("NCCL_DEBUG", "WARN")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.offline:
        env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("resolved_config.json", config), ("launch_manifest.json", manifest),
                        ("data_validation.json", report)):
        (output / name).write_text(json.dumps(value, indent=2) + "\n")
    (output / "recipe_provenance.json").write_bytes(
        Path(__file__).with_name("recipe_provenance.json").read_bytes())
    if args.replay_manifest:
        (output / "replay_manifest.json").write_bytes(args.replay_manifest.read_bytes())
    return subprocess.run(command, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
