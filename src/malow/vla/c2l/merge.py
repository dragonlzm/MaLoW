"""Checking that the stage-1 VLA adapter really merged into the frozen base.

This matters more than it looks. C2L wraps the base in `ModulatedPretrainedModel`, whose
`_init_model` calls `base_model.disable_adapter_layers()` when that method exists. On a
*merged* model there is no such method and the call is a no-op, which is what we want. On
an *attached* model it switches the whole stage-1 finetune off, silently — the run would
train a hypernet to modulate an un-finetuned backbone and nothing would report an error.
So the merge is worth asserting rather than assuming.

The obvious assertion is wrong, and was: `hasattr(model, "peft_config")` is **true after a
successful merge**. `BaseTuner.__init__` ends with

    # Copy the peft_config in the injected model.
    self.model.peft_config = self.peft_config

so the attribute is written onto the *base* model, and `merge_and_unload()` returns that
same object with the attribute still on it. Measured on peft 0.18.1: after
`merge_and_unload()` the attribute is present, zero `BaseTunerLayer` modules remain, and
the target module is a plain `nn.Linear`.

What actually distinguishes the two states is whether live tuner layers are left, which is
also exactly what `disable_adapter_layers()` would act on. That is what this checks.
"""

from __future__ import annotations

from typing import Any

from torch import nn


def residual_adapter_layers(model: nn.Module, limit: int = 5) -> list[str]:
    """Names of PEFT tuner layers still wrapping modules of ``model``.

    Empty after a successful merge. `peft_config` left over on the model is *not* evidence
    of an unmerged adapter — see the module docstring.
    """
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:  # peft is a hard dependency of this stage, but do not mask its absence
        raise

    names = []
    for name, module in model.named_modules():
        if isinstance(module, BaseTunerLayer):
            names.append(name)
            if len(names) >= limit:
                break
    return names


def assert_adapter_merged(model: nn.Module, source: Any) -> None:
    """Raise unless the adapter was merged away rather than attached."""
    residual = residual_adapter_layers(model)
    if residual:
        raise RuntimeError(
            f"The stage-1 VLA adapter from {source} was attached rather than merged: "
            f"{len(residual)}+ PEFT tuner layers remain (e.g. {residual}). "
            "`ModulatedPretrainedModel` would then call `disable_adapter_layers()` and train "
            "against the un-finetuned backbone with no error. Check that "
            "`finetuning_type: lora` is set and that DeepSpeed ZeRO-3 is off — ZeRO-3 forces "
            "`is_mergeable = False` in `model/adapter.py`."
        )
