"""Public-format training checkpoints with tensor-only in-place resume.

Training optimizer/RNG files remain Hugging Face Trainer's responsibility. Only
checkpoints produced by this training runtime are resumable here.
"""
from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Any

_RUN_CONFIG = contextvars.ContextVar("malow_training_config", default=None)
CONFIG_FILENAME = "training_config.json"
_PORTABLE_FIELDS = {
    "output_dir", "logging_dir", "run_name", "resume_from_checkpoint",
    "model_name_or_path", "cache_dir", "media_dir", "video_roots",
    "dataset_dir", "overwrite_output_dir", "save_steps", "save_total_limit",
    "logging_steps", "report_to", "disable_tqdm", "local_rank", "max_steps",
}


@contextlib.contextmanager
def training_config_context(config: dict[str, Any]):
    token = _RUN_CONFIG.set(dict(config))
    try:
        yield
    finally:
        _RUN_CONFIG.reset(token)


def _plain(value: Any) -> Any:
    """Compare equivalent public/training config classes without class identity."""
    import torch

    if isinstance(value, torch.Tensor):
        return {"tensor": value.detach().cpu().tolist(), "dtype": str(value.dtype)}
    if isinstance(value, Enum):
        return _plain(value.value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda p: str(p[0]))}
    if isinstance(value, (set, frozenset)):
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return _plain(vars(value))


def _unwrapped(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _model_config_hash(model) -> str:
    config = _unwrapped(model).base_model.config.to_dict()

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()
                    if k not in {"_name_or_path", "transformers_version", "_commit_hash"}}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return hashlib.sha256(json.dumps(clean(config), sort_keys=True, default=str).encode()).hexdigest()


def _safe_config(config):
    return {k: v for k, v in config.items()
            if not any(word in k.lower() for word in ("token", "password", "secret", "api_key"))}


def _recipe_config(config):
    result = {k: v for k, v in _safe_config(config).items() if k not in _PORTABLE_FIELDS}
    if config.get("replay_manifest"):
        # Content is pinned by replay_fingerprint; workers only fetch indexed positions.
        for key in ("replay_manifest", "dataloader_num_workers", "dataloader_persistent_workers",
                    "dataloader_prefetch_factor"):
            result.pop(key, None)
    # Profile locations may move; their ordered stream directory names may not.
    for key in ("dataset", "eval_dataset"):
        value = result.get(key)
        if value:
            values = value.split(",") if isinstance(value, str) else value
            result[key] = [Path(item).name for item in values]
    return _plain(result)


def save_training_checkpoint_metadata(metadata, directory, model):
    from malow.checkpoints import save_checkpoint_metadata

    directory = Path(directory)
    save_checkpoint_metadata(metadata, directory)
    config = _RUN_CONFIG.get()
    if config is not None:
        document = {"format": "malow-training-config", "version": 1,
                    "config": _safe_config(config), "model_config_sha256": _model_config_hash(model)}
        (directory / CONFIG_FILENAME).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
        output_root = Path(config.get("output_dir", directory))
        for name in ("launch_manifest.json", "recipe_provenance.json", "data_validation.json", "replay_manifest.json"):
            source = output_root / name
            if source.is_file() and source.resolve() != (directory / name).resolve():
                shutil.copyfile(source, directory / name)


def validate_resume_checkpoint(directory, model, config=None):
    """Reject changed architecture/recipe before optimizer state can be loaded."""
    import torch
    from malow.checkpoints import load_checkpoint_metadata

    directory = Path(directory)
    model = _unwrapped(model)
    saved = load_checkpoint_metadata(directory, runtime="video")
    current = {k: v for k, v in model.state_dict().items() if not isinstance(v, torch.Tensor)}
    # A matching full base config allows relocation of the same model snapshot.
    document_path = directory / CONFIG_FILENAME
    config = _RUN_CONFIG.get() if config is None else config
    if config is not None:
        if not document_path.is_file():
            raise ValueError("Resume requires training_config.json from a MaLoW training checkpoint.")
        document = json.loads(document_path.read_text())
        if document.get("format") != "malow-training-config" or document.get("version") != 1:
            raise ValueError("Unsupported training checkpoint configuration format.")
        if document["model_config_sha256"] != _model_config_hash(model):
            raise ValueError("Resume base-model configuration differs from the saved checkpoint.")
        old_recipe, new_recipe = _recipe_config(document["config"]), _recipe_config(config)
        if old_recipe != new_recipe:
            changed = sorted(k for k in old_recipe.keys() | new_recipe.keys()
                             if old_recipe.get(k) != new_recipe.get(k))
            raise ValueError(f"Resume training recipe differs: {', '.join(changed)}")
    elif saved.get("base_model_name_or_path") != current.get("base_model_name_or_path"):
        raise ValueError("Resume base-model identity differs from the saved checkpoint.")

    for values in (saved, current):
        values.pop("base_model_name_or_path", None)
    saved, current = _plain(saved), _plain(current)
    for values in (saved, current):
        values.get("ctx_encoder_args", {}).pop("ctx_encoder_model_name_or_path", None)
        values.get("hypernet_config", {}).get("lora_config", {}).pop("base_model_name_or_path", None)
    if saved != current:
        changed = sorted(k for k in saved.keys() | current.keys() if saved.get(k) != current.get(k))
        raise ValueError(f"Resume checkpoint architecture/settings differ: {', '.join(changed)}")


def load_training_weights(directory, model, config=None):
    """Restore weights without reconstructing modules or replacing Parameters."""
    from safetensors.torch import load_file

    validate_resume_checkpoint(directory, model, config)
    tensors = load_file(str(Path(directory) / "model.safetensors"), device="cpu")
    # Metadata must NOT be passed to this method: upstream would rebuild hypernet.
    return _unwrapped(model).load_state_dict(tensors, strict=True)
