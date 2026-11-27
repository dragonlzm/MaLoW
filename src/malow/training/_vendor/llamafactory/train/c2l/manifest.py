import dataclasses
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


C2L_MANIFEST_NAME = "c2l_training_manifest.yaml"
REDACTED_VALUE = "<REDACTED>"
UNAVAILABLE_VALUE = "unavailable"
_SENSITIVE_NAME_PARTS = {
    "secret",
    "password",
    "passwd",
    "api_key",
    "access_key",
    "private_key",
    "credential",
    "authorization",
    "auth",
}
_SENSITIVE_NAME_PHRASES = {"api_key", "access_key", "private_key"}
_TOKEN_CREDENTIAL_QUALIFIERS = {
    "access",
    "api",
    "auth",
    "bearer",
    "credential",
    "github",
    "gitlab",
    "hf",
    "hub",
    "private",
    "refresh",
    "secret",
    "session",
    "slack",
    "wandb",
}
_SOFTWARE_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "datasets": "datasets",
    "peft": "peft",
    "wandb": "wandb",
    "llamafactory": "llamafactory",
}


def _is_sensitive_name(name: Any, *, redact_token_suffix: bool = False) -> bool:
    normalized = str(name).lower().replace("-", "_")
    name_parts = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    token_credential = normalized == "token" or (
        normalized.endswith("_token")
        and (redact_token_suffix or bool(name_parts.intersection(_TOKEN_CREDENTIAL_QUALIFIERS)))
    )
    return (
        token_credential
        or bool(name_parts.intersection(_SENSITIVE_NAME_PARTS))
        or any(phrase in normalized for phrase in _SENSITIVE_NAME_PHRASES)
    )


def _to_yaml_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return _to_yaml_safe(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _to_yaml_safe(dataclasses.asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_yaml_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_yaml_safe(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _sanitize_mapping(value: Any, *, redact_token_suffix: bool) -> Any:
    value = _to_yaml_safe(value)
    if isinstance(value, Mapping):
        return {
            str(key): (
                REDACTED_VALUE
                if _is_sensitive_name(key, redact_token_suffix=redact_token_suffix)
                else _sanitize_mapping(item, redact_token_suffix=redact_token_suffix)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_mapping(item, redact_token_suffix=redact_token_suffix) for item in value]
    return value


def sanitize_mapping(value: Any) -> Any:
    return _sanitize_mapping(value, redact_token_suffix=False)


def _sanitize_environment(value: Mapping[str, str]) -> dict[str, Any]:
    return _sanitize_mapping(value, redact_token_suffix=True)


def sanitize_command(argv: Sequence[str]) -> list[str]:
    sanitized: list[str] = []
    redact_next = False
    for raw_arg in argv:
        arg = str(raw_arg)
        if redact_next:
            sanitized.append(REDACTED_VALUE)
            redact_next = False
            continue

        if "=" in arg:
            key, _value = arg.split("=", 1)
            sanitized.append(f"{key}={REDACTED_VALUE}" if _is_sensitive_name(key) else arg)
            continue

        sanitized.append(arg)
        if arg.startswith("-") and _is_sensitive_name(arg):
            redact_next = True

    return sanitized


def _run_git_command(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=os.getcwd(),
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_git_provenance() -> dict[str, Any]:
    repository_root = _run_git_command("rev-parse", "--show-toplevel")
    commit = _run_git_command("rev-parse", "HEAD")
    branch = _run_git_command("symbolic-ref", "--short", "-q", "HEAD")
    status = _run_git_command("status", "--porcelain")

    if repository_root is None and commit is None:
        return {
            "repository_root": UNAVAILABLE_VALUE,
            "commit": UNAVAILABLE_VALUE,
            "branch": UNAVAILABLE_VALUE,
            "detached_head": UNAVAILABLE_VALUE,
            "dirty": UNAVAILABLE_VALUE,
        }

    detached_head = branch is None
    return {
        "repository_root": repository_root or UNAVAILABLE_VALUE,
        "commit": commit or UNAVAILABLE_VALUE,
        "branch": branch or "detached",
        "detached_head": detached_head,
        "dirty": status is not None and bool(status),
    }


def collect_software_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for manifest_name, package_name in _SOFTWARE_PACKAGES.items():
        try:
            versions[manifest_name] = importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            versions[manifest_name] = UNAVAILABLE_VALUE
    return versions


def collect_runtime_provenance(training_args: Any) -> dict[str, Any]:
    environment = dict(os.environ)
    return {
        "command": sanitize_command(sys.argv),
        "working_directory": os.getcwd(),
        "hostname": socket.gethostname(),
        "process": {
            "pid": os.getpid(),
            "world_size": getattr(training_args, "world_size", UNAVAILABLE_VALUE),
            "process_index": getattr(training_args, "process_index", UNAVAILABLE_VALUE),
            "local_process_index": getattr(training_args, "local_process_index", UNAVAILABLE_VALUE),
        },
        "slurm": _sanitize_environment(
            {key: value for key, value in environment.items() if key.startswith("SLURM_")}
        ),
        "environment": _sanitize_environment(environment),
        "git": collect_git_provenance(),
        "software": collect_software_versions(),
    }


def _arguments_to_dict(arguments: Any) -> dict[str, Any]:
    if hasattr(arguments, "to_dict"):
        return arguments.to_dict()
    if dataclasses.is_dataclass(arguments) and not isinstance(arguments, type):
        return dataclasses.asdict(arguments)
    if isinstance(arguments, Mapping):
        return dict(arguments)
    raise TypeError(f"Cannot serialize argument object of type {type(arguments).__name__}.")


def build_c2l_training_manifest(
    *,
    model_args: Any,
    data_args: Any,
    training_args: Any,
    finetuning_args: Any,
    generating_args: Any,
    save_kind: str,
    global_step: int,
    target_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_type": "c2l_training",
        "saved_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "save_context": {
            "kind": save_kind,
            "global_step": int(global_step),
            "output_dir": str(Path(target_dir).resolve()),
        },
        "arguments": sanitize_mapping(
            {
                "model": _arguments_to_dict(model_args),
                "data": _arguments_to_dict(data_args),
                "training": _arguments_to_dict(training_args),
                "finetuning": _arguments_to_dict(finetuning_args),
                "generation": _arguments_to_dict(generating_args),
            }
        ),
        "runtime": collect_runtime_provenance(training_args),
    }


def write_c2l_training_manifest(
    target_dir: str | os.PathLike[str], manifest: Mapping[str, Any]
) -> str:
    target_directory = Path(target_dir)
    target_directory.mkdir(parents=True, exist_ok=True)
    output_path = target_directory / C2L_MANIFEST_NAME
    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target_directory,
            prefix=f".{C2L_MANIFEST_NAME}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            yaml.safe_dump(_to_yaml_safe(manifest), temporary_file, sort_keys=False, allow_unicode=True)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise

    return str(output_path)


class C2LTrainingManifestCallback(TrainerCallback):
    def __init__(self, *, model_args: Any, data_args: Any, finetuning_args: Any, generating_args: Any) -> None:
        self.model_args = model_args
        self.data_args = data_args
        self.finetuning_args = finetuning_args
        self.generating_args = generating_args

    def _write(self, args: Any, state: Any, save_kind: str, target_dir: str | os.PathLike[str]) -> None:
        if not state.is_world_process_zero:
            return

        manifest = build_c2l_training_manifest(
            model_args=self.model_args,
            data_args=self.data_args,
            training_args=args,
            finetuning_args=self.finetuning_args,
            generating_args=self.generating_args,
            save_kind=save_kind,
            global_step=state.global_step,
            target_dir=target_dir,
        )
        write_c2l_training_manifest(target_dir, manifest)

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write(args, state, "run_start", args.output_dir)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        checkpoint_dir = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        self._write(args, state, "checkpoint", checkpoint_dir)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        self._write(args, state, "run_end", args.output_dir)
