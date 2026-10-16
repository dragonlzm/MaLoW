import ast
import gc
import hashlib
import logging
import os
import random
import string
import time
from collections.abc import Iterable
from contextlib import contextmanager
from enum import Enum

import torch
import yaml
from peft import PeftConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import get_peft_model_state_dict

TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])


logger = logging.getLogger()


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3


def get_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "language_model"):
        return get_layers(model.language_model)
    if hasattr(model, "base_model"):
        return get_layers(model.base_model)
    if hasattr(model, "model"):
        return get_layers(model.model)
    raise AttributeError(f"Cannot find decoder layers on {model.__class__.__name__}.")


def get_num_layers(model):
    return len(get_layers(model))


def get_base_model(model):
    if hasattr(model, "language_model"):
        return get_base_model(model.language_model)
    if hasattr(model, "base_model"):
        return get_base_model(model.base_model)
    if hasattr(model, "model"):
        return get_base_model(model.model)
    return model




















def _iter_lora_target_modules(peft_config: PeftConfig) -> list[str]:
    target_modules = peft_config.target_modules
    if target_modules is None:
        return []
    if isinstance(target_modules, str):
        return [target_modules]
    return list(target_modules)


def get_lora_target_key(module_name: str, peft_config: PeftConfig) -> str | None:
    if not check_target_module_exists(peft_config, module_name):
        return None

    leaf_name = module_name.split(".")[-1]
    for target_key in sorted(_iter_lora_target_modules(peft_config), key=len, reverse=True):
        if (
            module_name == target_key
            or module_name.endswith(f".{target_key}")
            or leaf_name == target_key
        ):
            return target_key
    return leaf_name


def safe_lora_target_key(target_key: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in target_key)


def get_lora_target_modules(model: torch.nn.Module, peft_config: PeftConfig) -> list[dict[str, str]]:
    return [
        {"name": name, "module": module, "target_key": target_key}
        for name, module in model.named_modules()
        if (target_key := get_lora_target_key(name, peft_config)) is not None
        and isinstance(module, (BaseTunerLayer, torch.nn.Linear))
    ]


def get_peft_in_out_features(
    model: torch.nn.Module,
    peft_config: PeftConfig | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    if peft_config is None:
        return None, None
    in_features = dict()
    out_features = dict()
    if isinstance(model, PeftModel):
        modules = get_lora_target_modules(model, peft_config)
    else:
        modules = []
        for layer in get_layers(model):
            modules.extend(get_lora_target_modules(layer, peft_config))

    for module_info in modules:
        module_name = module_info["name"]
        module = module_info["module"]
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        if isinstance(module, BaseTunerLayer):
            assert isinstance(module.base_layer, torch.nn.Linear), (
                "all modules should be a leave module that is Linear layer"
            )
            in_features_value = module.in_features
            out_features_value = module.out_features
        else:
            assert isinstance(module, torch.nn.Linear), "all modules should be Linear layers"
            in_features_value = module.in_features
            out_features_value = module.out_features

        name = module_info.get("target_key") or module_name.split(".")[-1]

        if name not in in_features:
            in_features[name] = in_features_value
            out_features[name] = out_features_value
        else:
            # assumes each module has the same input and output features
            assert in_features[name] == in_features_value
            assert out_features[name] == out_features_value

    return in_features, out_features


def get_lora_target_slots(
    model: torch.nn.Module,
    peft_config: PeftConfig,
    layer_indices: Iterable[int] | None = None,
) -> list[dict[str, int | str]]:
    if layer_indices is None:
        layer_indices = range(len(get_layers(model)))

    slots = []
    seen = set()
    layers = get_layers(model)
    for layer_idx in layer_indices:
        layer_idx = int(layer_idx)
        layer = layers[layer_idx]
        for module_info in get_lora_target_modules(layer, peft_config):
            module_name = module_info["name"]
            target_key = module_info["target_key"]
            module = module_info["module"]
            if isinstance(module, BaseTunerLayer):
                assert isinstance(module.base_layer, torch.nn.Linear), (
                    "all modules should be a leave module that is Linear layer"
                )
                in_features_value = module.in_features
                out_features_value = module.out_features
            else:
                assert isinstance(module, torch.nn.Linear), "all modules should be Linear layers"
                in_features_value = module.in_features
                out_features_value = module.out_features

            slot_key = f"layer_{layer_idx}__{safe_lora_target_key(target_key)}"
            if slot_key in seen:
                continue
            seen.add(slot_key)
            slots.append(
                {
                    "slot_key": slot_key,
                    "layer_idx": layer_idx,
                    "module_path": module_name,
                    "target_key": target_key,
                    "d_in": int(in_features_value),
                    "d_out": int(out_features_value),
                }
            )

    return slots












