"""The interactive Context-to-LoRA path must reproduce the offline one.

Interactive evaluation answers a video's questions as its footage arrives, so the temporal
module sees the blocks in segments instead of all at once. Both of its branches are exact
prefix functions, and these tests pin that down:

* the Gated DeltaNet, continued through its ``(conv_state, recurrent_state)`` cache;
* the ``gated_residual`` skip connection, continued through a running ``(sum, count)``.

The carry is algebraically exact, so in fp32 the two paths agree to round-off; the chunk
boundaries only change the order of the summations. On GPU the module runs under bf16
autocast with fused kernels, where the agreement is looser -- the runtime
``--interactive-verify-sessions`` check is the one that measures that.

**These tests are only meaningful if dropping the state would actually change the answer.**
Under the module's default random init the recurrent state decays by roughly half per block,
so after a dozen blocks a *broken* implementation would still look almost correct and an
equivalence assertion would pass vacuously. Every equivalence test below therefore runs in a
retention regime (``strong_memory=True``: ``g ~ 0``, so nothing is forgotten) and is paired
with a negative control that feeds the same segment with **no** history and asserts the
result is visibly wrong.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from malow.video.ctx_to_lora.modeling.temporal_aggregator import TemporalGatedDeltaNet


LATENT = 128
NUM_BASE_LAYERS = 3

# fp32 round-off from reassociating the chunked scan at a different boundary. Each
# equivalence test reports the margin over this that a state-dropping bug would produce.
EXACT = {"rtol": 1e-4, "atol": 1e-4}


def build_module(num_hidden_layers: int = 2, strong_memory: bool = True) -> TemporalGatedDeltaNet:
    torch.manual_seed(0)
    module = TemporalGatedDeltaNet(
        LATENT, NUM_BASE_LAYERS, num_hidden_layers=num_hidden_layers
    ).eval()
    with torch.no_grad():
        # The default init is close to the identity, which would hide a broken state carry.
        for parameter in module.parameters():
            parameter.normal_(0, 0.3)
        if strong_memory:
            # g = -exp(A_log) * softplus(a + dt_bias). A tiny A drives g to 0, so the
            # recurrent state is never forgotten and dropping it cannot go unnoticed.
            for block in module.blocks:
                block.mixer.A_log.fill_(math.log(1e-4))
                block.mixer.dt_bias.fill_(-4.0)
                block.mixer.conv1d.weight.normal_(0, 0.5)
    return module


def run(module, blocks, cache=None):
    with torch.no_grad():
        return module(blocks, torch.ones(blocks.shape[:2], dtype=torch.long), cache_params=cache)


def stream(module: TemporalGatedDeltaNet, blocks: torch.Tensor, splits: list[int]) -> torch.Tensor:
    """Feed `blocks` segment by segment through one shared cache, as `interactive_update` does."""
    cache = module.new_cache()
    bounds = [0, *splits, blocks.shape[1]]
    return torch.cat(
        [run(module, blocks[:, start:end], cache) for start, end in zip(bounds[:-1], bounds[1:])],
        dim=1,
    )


def assert_state_actually_matters(module, blocks, split):
    """The negative control: the same segment with no history must be visibly wrong.

    Without this, an equivalence assertion proves nothing -- a module that silently discards
    the carried state would satisfy it too.
    """
    offline_tail = run(module, blocks)[:, split:]
    cold = run(module, blocks[:, split:])
    error = (cold - offline_tail).abs().max()
    span = (offline_tail - blocks[:, split:]).abs().max()
    assert error > 100 * EXACT["atol"], (
        f"the negative control is too weak to detect a dropped state: {error:.3e}"
    )
    assert error > 0.1 * span, "dropping the recurrent state barely moves the output here"
    return float(error)


@pytest.mark.parametrize(
    "splits",
    [[12], [12, 25], [1, 2, 3, 5, 8, 13, 21, 34], list(range(1, 37))],
    ids=["one-split", "two-splits", "uneven-splits", "block-at-a-time"],
)
def test_streamed_deltanet_matches_a_single_pass(splits):
    module = build_module()
    blocks = torch.randn(3, 37, LATENT)
    offline = run(module, blocks)

    streamed = stream(module, blocks, splits)

    # The last token is what `_aggregate_temporal_lora_embeddings` selects.
    torch.testing.assert_close(streamed[:, -1], offline[:, -1], **EXACT)
    torch.testing.assert_close(streamed, offline, **EXACT)


@pytest.mark.parametrize("segment_blocks", [1, 2, 3, 5, 18], ids=lambda n: f"{n}-block-segment")
def test_a_multi_block_segment_consumes_the_carried_state(segment_blocks):
    """A segment longer than one block must still continue the cached prefix.

    Multi-token continuation and single-token decode are different branches of
    `Qwen3_5GatedDeltaNet.forward`, so segment length is the axis a state-carry bug would
    hide behind. This asserts both that the warm run is right and that the cold run is wrong.
    """
    module = build_module()
    split = 6
    blocks = torch.randn(2, split + segment_blocks, LATENT)
    offline_tail = run(module, blocks)[:, split:]

    cache = module.new_cache()
    run(module, blocks[:, :split], cache)
    warm = run(module, blocks[:, split:], cache)
    cold = run(module, blocks[:, split:])

    torch.testing.assert_close(warm, offline_tail, **EXACT)
    assert not torch.equal(warm, cold)
    cold_error = assert_state_actually_matters(module, blocks, split)
    warm_error = float((warm - offline_tail).abs().max())
    # The whole point: the warm run is orders of magnitude closer than the cold one.
    assert cold_error > 1e3 * max(warm_error, 1e-12)


def test_streaming_is_exact_for_a_deeper_stack():
    """V1.4.9 runs four DeltaNet layers, so each layer keeps its own cached state."""
    module = build_module(num_hidden_layers=4)
    blocks = torch.randn(2, 20, LATENT)
    offline = run(module, blocks)

    assert_state_actually_matters(module, blocks, 3)
    streamed = stream(module, blocks, [3, 11])

    torch.testing.assert_close(streamed[:, -1], offline[:, -1], **EXACT)


def test_cache_covers_the_causal_conv_left_context():
    """A split shorter than the conv kernel still has to see the frames before it."""
    module = build_module()
    blocks = torch.randn(1, 8, LATENT)
    offline = run(module, blocks)

    # conv_kernel_dim is 4, so splitting after 1, 2 and 3 blocks exercises a partially
    # filled conv state as well as a full one.
    streamed = stream(module, blocks, [1, 2, 3])

    torch.testing.assert_close(streamed[:, -1], offline[:, -1], **EXACT)


def test_a_fresh_cache_does_not_change_a_single_pass():
    module = build_module()
    blocks = torch.randn(2, 9, LATENT)
    mask = torch.ones(2, 9, dtype=torch.long)
    with torch.no_grad():
        uncached = module(blocks, mask)
        cached = module(blocks, mask, cache_params=module.new_cache())

    torch.testing.assert_close(cached, uncached, rtol=0, atol=0)  # identical code path


def test_streaming_also_holds_at_the_default_initialization():
    """The retention regime is a test device; the carry itself is not conditional on it."""
    module = build_module(strong_memory=False)
    blocks = torch.randn(2, 24, LATENT)
    offline = run(module, blocks)

    streamed = stream(module, blocks, [6, 7])

    torch.testing.assert_close(streamed[:, -1], offline[:, -1], **EXACT)



@pytest.mark.parametrize("mode", ["gated_residual", "only"])
@pytest.mark.parametrize("weighting", [None, "raw_pooled_cosine_gate"])
@pytest.mark.parametrize("per_layer", [True, False])
def test_interactive_loras_match_original_offline_aggregation(mode, weighting, per_layer):
    from malow.video.ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
    from malow.video.ctx_to_lora.modeling.temporal_weighting import RawPooledCosineGate

    class Host(ModulatedPretrainedModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.temporal_aggregator = mode
            self.temporal_query_weighting = weighting
            self.image_context_encoder = self.block_lora = self.temporal_sum_at_query = True
            self.frame_per_lora = 2
            self.user_defined_scaling = 1.3
            self._logged_temporal_gate_retention = True
            self.hypernet = SimpleNamespace(
                temporal_aggregator=build_module(),
                temporal_query_weighting=RawPooledCosineGate(threshold=0.3),
                config=SimpleNamespace(latent_size=LATENT, aggregator_config=SimpleNamespace(
                    query_per_layer=per_layer, query_per_module=False, query_per_rank=False)),
                generate_weights_from_lora_embeddings=lambda x: ({"module": {"A": x, "B": x * 0.3}}, None),
            )
        def _interactive_block_embeddings(self, inputs):
            return inputs

    host = Host()
    embeddings = torch.randn(9, NUM_BASE_LAYERS if per_layer else 1, 1, 1, LATENT).expand(-1, NUM_BASE_LAYERS, 2, 2, -1)
    pooled = torch.randn(9, 8)
    # Cross-segment near-duplicate: gate must compare to the previous segment's final block.
    pooled[3] = pooled[2]
    owners = torch.zeros(9, dtype=torch.long)
    weights = host._raw_temporal_query_weights(pooled, owners, 1)
    with torch.no_grad():
        reference = host._aggregate_temporal_lora_embeddings(embeddings, owners, 1, weights) * host.user_defined_scaling
    state = host.new_interactive_state()
    for start, end in [(0, 3), (3, 4), (4, 9)]:
        host.interactive_update((embeddings[start:end], pooled[start:end]), state)
    actual, norms = host.interactive_generate_weights(state)
    torch.testing.assert_close(actual["module"]["A"], reference, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual["module"]["B"], reference * 0.3, rtol=1e-4, atol=1e-4)
    assert norms is None and state.block_count == 9 and state.segment_count == 3
    # Materializing weights must not alter memory or accumulate user scaling.
    repeated, _ = host.interactive_generate_weights(state)
    torch.testing.assert_close(actual["module"]["A"], repeated["module"]["A"], rtol=0, atol=0)
    cold = host.new_interactive_state()
    host.interactive_update((embeddings[4:], pooled[4:]), cold)
    dropped, _ = host.interactive_generate_weights(cold)
    assert (dropped["module"]["A"] - reference).abs().max() > 0.01
    assert cold.deltanet_cache is not state.deltanet_cache


def test_generate_accepts_precomputed_loras_without_encoding(monkeypatch):
    from malow.video.ctx_to_lora.modeling import hypernet
    calls = []
    supplied = {"module": {"A": torch.ones(1), "B": torch.ones(1)}}
    host = SimpleNamespace(
        use_base_input_as_ctx=False, image_context_encoder=True, device="cpu",
        use_sequence_packing=False,
        _default_image_n_ctx_chunks=lambda x: torch.tensor([1]),
        combine_lora=lambda x, *a, **kw: x,
        hypernet=SimpleNamespace(config=SimpleNamespace(use_bias=False), layer_indices=[0]),
        hypernet_config=SimpleNamespace(use_valid_slots=False),
        base_model=SimpleNamespace(generate=lambda **kw: kw),
    )
    monkeypatch.setattr(hypernet, "apply_lora_to_layers", lambda base, layers, weights, *a: calls.append(weights))
    result = hypernet.ModulatedPretrainedModel.generate(host, generated_loras=supplied, input_ids=torch.tensor([[1]]))
    assert calls == [supplied]
    assert "generated_loras" not in result
    with pytest.raises(ValueError, match="either"):
        hypernet.ModulatedPretrainedModel.generate(host, generated_loras=supplied, ctx_ids={})


def test_single_block_fallback_preserves_carry(monkeypatch):
    from malow.video.ctx_to_lora.modeling import temporal_aggregator
    monkeypatch.setattr(temporal_aggregator, "multitoken_continuation_supported", lambda *a: False)
    module = build_module()
    blocks = torch.randn(2, 9, LATENT)
    reference = run(module, blocks)
    cache = module.new_cache()
    run(module, blocks[:, :3], cache)
    calls = []
    original = module._forward_blocks
    def record(sequence, mask, cache):
        calls.append(sequence.shape[1])
        return original(sequence, mask, cache)
    monkeypatch.setattr(module, "_forward_blocks", record)
    actual = run(module, blocks[:, 3:], cache)
    assert calls == [1] * 6
    torch.testing.assert_close(actual, reference[:, 3:], **EXACT)
