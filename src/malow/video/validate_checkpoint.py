"""Check a video checkpoint's configuration and tensor layout on CPU."""
from pathlib import Path
import argparse


def validate(directory):
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from malow.checkpoints import load_checkpoint_metadata
    from malow.video.ctx_to_lora.modeling.hypernet import HyperLoRA, HyperLoRA_valid_slots
    path = Path(directory)
    metadata = load_checkpoint_metadata(path, runtime="video")
    config = metadata["hypernet_config"]
    cls = HyperLoRA_valid_slots if getattr(config, "use_valid_slots", False) else HyperLoRA
    with init_empty_weights():
        model = cls(config)
    expected = model.state_dict()
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as weights:
        keys = set(weights.keys())
        if keys != set(expected):
            raise ValueError(f"Tensor keys mismatch: missing={sorted(set(expected)-keys)}, extra={sorted(keys-set(expected))}")
        for key, tensor in expected.items():
            actual = weights.get_slice(key).get_shape()
            if actual != list(tensor.shape):
                raise ValueError(f"Tensor shape mismatch for {key}: {actual} != {list(tensor.shape)}")
    return {"base_model": metadata["base_model_name_or_path"], "tensor_count": len(expected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Released checkpoint directory")
    args = parser.parse_args()
    result = validate(args.checkpoint)
    print(f"Validated {result['tensor_count']} tensor shapes for {result['base_model']}")


if __name__ == "__main__":
    main()
