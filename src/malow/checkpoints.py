"""Versioned, non-pickle inference metadata for released MaLoW checkpoints.

Only the explicitly registered configuration types can be reconstructed. Large
model tensors remain in their original files; small metadata tensors use a
separate safetensors file.
"""
from __future__ import annotations

import importlib
import json
from enum import Enum
from pathlib import Path
from typing import Any

FORMAT = "malow-checkpoint-metadata"
FORMAT_VERSION = 1
METADATA_FILENAME = "c2l_metadata.json"
TENSORS_FILENAME = "c2l_metadata.safetensors"

_CONFIG_TYPES = {
    "HypernetConfig": "modeling.hypernet.HypernetConfig",
    "AggregatorConfig": "modeling.aggregator.AggregatorConfig",
    "CtxEncoderArguments": "configs.CtxEncoderArguments",
}
_ENUM_TYPES = {"AGGREGATOR_TYPE": "modeling.aggregator.AGGREGATOR_TYPE"}
_PEFT_TYPES = {
    "LoraConfig": "peft.tuners.lora.config.LoraConfig",
    "LoraRuntimeConfig": "peft.tuners.lora.config.LoraRuntimeConfig",
    "PeftType": "peft.utils.peft_types.PeftType",
    "TaskType": "peft.utils.peft_types.TaskType",
}
_ALLOWED_TYPES = set(_CONFIG_TYPES) | set(_ENUM_TYPES) | set(_PEFT_TYPES)


def _resolve_type(name: str, runtime: str):
    if runtime not in {"video", "vla"}:
        raise ValueError("runtime must be 'video' or 'vla'")
    if name in _CONFIG_TYPES:
        qualified = f"malow.{runtime}.ctx_to_lora.{_CONFIG_TYPES[name]}"
    elif name in _ENUM_TYPES:
        qualified = f"malow.{runtime}.ctx_to_lora.{_ENUM_TYPES[name]}"
    elif name in _PEFT_TYPES:
        qualified = _PEFT_TYPES[name]
    else:
        raise ValueError(f"Unsupported checkpoint configuration type: {name!r}")
    module, _, attribute = qualified.rpartition(".")
    return getattr(importlib.import_module(module), attribute)


def _encode(value: Any, tensors: dict[str, Any]) -> Any:
    import torch

    # Enum must precede str/int because the original enum types subclass str.
    if isinstance(value, Enum):
        name = type(value).__name__
        if name not in _ALLOWED_TYPES:
            raise TypeError(f"Unsupported enum: {name}")
        return {"$type": "enum", "class": name, "value": _encode(value.value, tensors)}
    if isinstance(value, torch.Tensor):
        key = f"tensor_{len(tensors):04d}"
        tensors[key] = value.detach().cpu().contiguous().clone()
        return {"$type": "tensor", "key": key}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {"$type": "dict", "items": [[_encode(k, tensors), _encode(v, tensors)] for k, v in value.items()]}
    if isinstance(value, (tuple, set, frozenset)):
        name = type(value).__name__
        values = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return {"$type": name, "items": [_encode(v, tensors) for v in values]}
    if isinstance(value, list):
        return [_encode(v, tensors) for v in value]
    name = getattr(type(value), "_metadata_type", type(value).__name__)
    if name not in _ALLOWED_TYPES:
        raise TypeError(f"Unsupported configuration: {name}")
    # Legacy-import inert proxies retain enum constructor arguments separately.
    if name in _ENUM_TYPES:
        return {"$type": "enum", "class": name, "value": _encode(value._constructor_args[0], tensors)}
    attributes = {k: v for k, v in vars(value).items() if k != "_constructor_args"}
    return {"$type": "config", "class": name, "attributes": {k: _encode(v, tensors) for k, v in attributes.items()}}


def save_checkpoint_metadata(metadata: dict[str, Any], directory: str | Path) -> Path:
    """Write a dictionary of already-loaded, allowlisted inference settings."""
    from safetensors.torch import save_file

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Any] = {}
    payload = {k: _encode(v, tensors) for k, v in metadata.items()}
    document = {"format": FORMAT, "version": FORMAT_VERSION, "metadata": payload}
    if tensors:
        save_file(tensors, str(directory / TENSORS_FILENAME))
        document["tensor_file"] = TENSORS_FILENAME
    (directory / METADATA_FILENAME).write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return directory / METADATA_FILENAME


def _decode(value: Any, tensors: dict[str, Any], runtime: str) -> Any:
    if isinstance(value, list):
        return [_decode(v, tensors, runtime) for v in value]
    if not isinstance(value, dict):
        return value
    kind = value.get("$type")
    if kind == "tensor":
        return tensors[value["key"]]
    if kind == "dict":
        return {_decode(k, tensors, runtime): _decode(v, tensors, runtime) for k, v in value["items"]}
    if kind in {"tuple", "set", "frozenset"}:
        constructor = {"tuple": tuple, "set": set, "frozenset": frozenset}[kind]
        return constructor(_decode(v, tensors, runtime) for v in value["items"])
    if kind in {"config", "enum"}:
        cls = _resolve_type(value["class"], runtime)
        if kind == "enum":
            return cls(_decode(value["value"], tensors, runtime))
        attributes = {k: _decode(v, tensors, runtime) for k, v in value["attributes"].items()}
        # Loading mirrors a dataclass pickle's state restoration: checkpoint
        # attributes remain exact, including compatibility fields from PEFT.
        instance = cls.__new__(cls)
        vars(instance).update(attributes)
        return instance
    raise ValueError(f"Unsupported metadata node: {kind!r}")


def load_checkpoint_metadata(path: str | Path, runtime: str = "video") -> dict[str, Any]:
    """Load public JSON metadata into the requested runtime's config classes.

    ``path`` may be a checkpoint directory or its ``c2l_metadata.json`` file.
    Legacy pickle metadata is deliberately unsupported by the public loader.
    """
    from safetensors.torch import load_file

    if runtime not in {"video", "vla"}:
        raise ValueError("runtime must be 'video' or 'vla'")
    path = Path(path)
    if path.is_dir():
        path = path / METADATA_FILENAME
    document = json.loads(path.read_text())
    if document.get("format") != FORMAT or document.get("version") != FORMAT_VERSION:
        raise ValueError("Unsupported MaLoW checkpoint metadata format/version")
    tensor_file = document.get("tensor_file")
    if tensor_file is not None and tensor_file != TENSORS_FILENAME:
        raise ValueError("Metadata tensor_file must be c2l_metadata.safetensors")
    tensors = load_file(str(path.parent / tensor_file), device="cpu") if tensor_file else {}
    metadata = document.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be an object")
    return {k: _decode(v, tensors, runtime) for k, v in metadata.items()}
