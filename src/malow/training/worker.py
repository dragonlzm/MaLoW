"""Internal torchrun worker; use malow.training.train for launch and preflight."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    from .runtime import run_training
    run_training(json.loads(args.config.read_text()))


if __name__ == "__main__":
    main()
