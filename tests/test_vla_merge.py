"""Inference regression tests derived from Hippocampus master e998ec8."""
import pytest
import torch
from torch import nn

from malow.vla.c2l.merge import assert_adapter_merged, residual_adapter_layers


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def _attached() -> nn.Module:
    """The base model with a live adapter injected into it."""
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(_Tiny(), LoraConfig(target_modules=["q_proj"], r=2))
    return peft_model.base_model.model


def _merged() -> nn.Module:
    """The same, after merging.

    Built from its own `_Tiny()` rather than by merging the one `_attached()` returned:
    `merge_and_unload()` mutates in place and hands back the *same* object, so sharing one
    construction would leave both names pointing at the merged model and the "attached"
    half of every assertion below would be vacuous.
    """
    from peft import LoraConfig, get_peft_model

    peft_model = get_peft_model(_Tiny(), LoraConfig(target_modules=["q_proj"], r=2))
    return peft_model.merge_and_unload()


def test_peft_config_survives_a_successful_merge():
    """The regression itself: `hasattr(model, "peft_config")` cannot distinguish the states."""
    attached, merged = _attached(), _merged()
    assert hasattr(attached, "peft_config")
    assert hasattr(merged, "peft_config"), (
        "peft stopped copying peft_config onto the base model; the comment in merge.py "
        "explaining why that attribute is not evidence needs revisiting."
    )


def test_residual_layers_distinguish_merged_from_attached():
    attached, merged = _attached(), _merged()
    assert residual_adapter_layers(attached), "an attached adapter must leave tuner layers"
    assert residual_adapter_layers(merged) == [], "a merged adapter must leave none"
    assert type(merged.q_proj) is nn.Linear


def test_assert_passes_on_merged_and_raises_on_attached():
    attached, merged = _attached(), _merged()
    assert_adapter_merged(merged, "test-checkpoint")
    with pytest.raises(RuntimeError, match="attached rather than merged"):
        assert_adapter_merged(attached, "test-checkpoint")


def test_a_plain_model_passes():
    """A base model that never met peft is trivially merged."""
    model = _Tiny()
    assert not hasattr(model, "peft_config")
    assert_adapter_merged(model, "test-checkpoint")


def test_merged_model_has_no_disable_adapter_layers():
    """The reason the merge matters: the no-op is what keeps the finetune alive."""
    attached, merged = _attached(), _merged()
    assert not hasattr(merged, "disable_adapter_layers")
    # ... and the merged weights are not the pristine ones.
    assert not torch.equal(merged.q_proj.weight, _Tiny().q_proj.weight)
