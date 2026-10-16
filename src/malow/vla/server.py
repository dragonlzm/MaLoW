"""Start a single baseline or MaLoW VLA policy server."""
import argparse
import logging
import os


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=("c2l", "baseline"), default="c2l")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--stage1-checkpoint", help="Required released baseline checkpoint for MaLoW VLA")
    p.add_argument("--base-model", default="Qwen/Qwen3.5-2B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=10100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.kind == "c2l" and not args.stage1_checkpoint:
        p.error("--stage1-checkpoint is required for C2L")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(level=logging.INFO, force=True)
    if args.kind == "c2l":
        from .simpler.server_policy_c2l_vla import main as serve
    else:
        from .simpler.server_policy_vla import main as serve
    serve(argparse.Namespace(ckpt_path=args.checkpoint, stage1_ckpt_path=args.stage1_checkpoint,
        model_name_or_path=args.base_model, device=args.device, host=args.host,
        port=args.port, seed=args.seed, deterministic=True, use_bf16=True,
        output_prior_action=False, c2l_step_seconds=0.2, c2l_action_feedback=True))


if __name__ == "__main__":
    main()
