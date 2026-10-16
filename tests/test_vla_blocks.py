"""Temporal state, padding and reset regression tests from master e998ec8."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from malow.vla.ctx_to_lora.modeling.temporal_aggregator import TemporalGatedDeltaNet


from malow.vla.c2l import blocks  # noqa: E402


MAX_STEP = 24

# Bridge (5 Hz), RT-1/fractal (3 Hz), and a 1 Hz control. The first two are the rates a
# `bridge_rt_1` batch actually mixes; the third keeps the arithmetic readable when a
# failure has to be diagnosed by eye.
STEP_SECONDS_CASES = (0.2, 1.0 / 3.0, 1.0)
BRIDGE_STEP_SECONDS = 0.2


# --- 1. agreement with the offline collator ----------------------------------------



def test_labels_are_on_the_datasets_own_clock():
    """The correction this stage exists for: a step is a control period, not a second.

    Bridge steps are 0.2s and RT-1 steps 1/3s, so the same block index names different
    times in the two halves of one `bridge_rt_1` batch.
    """
    assert blocks.context_block_labels(4, 0.2) == ["0s-0.4s", "0.4s-0.8s"]
    assert blocks.context_block_labels(4, 1.0 / 3.0) == ["0s-0.666667s", "0.666667s-1.33333s"]
    # The old one-second-per-step policy, for contrast.
    assert blocks.context_block_labels(4, 1.0) == ["0s-2s", "2s-4s"]


def test_step_seconds_must_be_positive():
    for bad in (0.0, -0.2):
        with pytest.raises(ValueError):
            blocks.context_block_labels(4, bad)


def test_step_zero_has_no_context():
    assert blocks.context_frame_steps(0) == []
    assert blocks.context_block_labels(0, BRIDGE_STEP_SECONDS) == []
    assert blocks.context_action_steps(0) == []
    assert blocks.block_count(0) == 0


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (1, [0, 0]),
        (2, [0, 1]),
        (3, [0, 1, 2, 2]),
        (4, [0, 1, 2, 3]),
        (5, [0, 1, 2, 3, 4, 4]),
    ],
)
def test_history_is_padded_with_the_last_frame(step, expected):
    """`[0, t-1s]`, padded to an even length by repeating the last frame."""
    assert blocks.context_frame_steps(step) == expected


def test_every_block_has_exactly_two_frames():
    for step in range(1, MAX_STEP):
        frames = blocks.context_frame_steps(step)
        assert len(frames) % blocks.FRAMES_PER_BLOCK == 0
        assert len(frames) // blocks.FRAMES_PER_BLOCK == blocks.block_count(step)
        assert len(blocks.context_block_labels(step, BRIDGE_STEP_SECONDS)) == blocks.block_count(step)


# --- 2. committed blocks are final --------------------------------------------------

@pytest.mark.parametrize("step_seconds", STEP_SECONDS_CASES)
def test_committed_blocks_never_change(step_seconds):
    """A block with two real frames keeps its frames, label, and actions for the episode.

    This is the property the streaming state depends on: a recurrent state cannot un-fold
    a block, so anything folded permanently must never need revising.
    """
    for step in range(1, MAX_STEP):
        frames = blocks.context_frame_steps(step)
        labels = blocks.context_block_labels(step, step_seconds)
        actions = blocks.context_block_action_steps(step)
        for block_index in range(blocks.committed_block_count(step)):
            start = block_index * blocks.FRAMES_PER_BLOCK
            assert frames[start : start + blocks.FRAMES_PER_BLOCK] == (
                blocks.committed_block_frame_steps(block_index)
            ), f"step {step}, block {block_index}"
            assert labels[block_index] == blocks.committed_block_label(block_index, step_seconds)
            assert actions[block_index] == blocks.committed_block_action_steps(block_index)


def test_only_the_last_block_of_an_odd_history_is_provisional():
    for step in range(1, MAX_STEP):
        committed = blocks.committed_block_count(step)
        if blocks.has_provisional_block(step):
            assert step % 2 == 1
            assert blocks.block_count(step) == committed + 1
            assert blocks.context_frame_steps(step)[-blocks.FRAMES_PER_BLOCK :] == (
                blocks.provisional_block_frame_steps(step)
            )
            assert blocks.context_block_labels(step, BRIDGE_STEP_SECONDS)[-1] == (
                blocks.provisional_block_label(step, BRIDGE_STEP_SECONDS)
            )
            assert blocks.context_block_action_steps(step)[-1] == (
                blocks.provisional_block_action_steps(step)
            )
        else:
            assert blocks.block_count(step) == committed


def test_a_provisional_block_is_replaced_not_extended():
    """At step 3 the tail is `[x2,x2]`/"0.4s-0.6s"; at step 4 it is `[x2,x3]`/"0.4s-0.8s".

    Same block position, different content and label -- which is exactly why it must be
    folded into a clone and thrown away rather than into the session state. The actions
    move with the frames: `[a1, a1]` becomes `[a1, a2]`.
    """
    assert blocks.provisional_block_frame_steps(3) == [2, 2]
    assert blocks.provisional_block_label(3, BRIDGE_STEP_SECONDS) == "0.4s-0.6s"
    assert blocks.provisional_block_action_steps(3) == [1, 1]
    assert blocks.committed_block_frame_steps(1) == [2, 3]
    assert blocks.committed_block_label(1, BRIDGE_STEP_SECONDS) == "0.4s-0.8s"
    assert blocks.committed_block_action_steps(1) == [1, 2]


# --- 2b. every frame carries the action that produced it ----------------------------

def test_each_frame_pairs_with_the_previous_steps_action():
    for step in range(1, MAX_STEP):
        frame_steps = blocks.context_frame_steps(step)
        action_steps = blocks.context_action_steps(step)
        assert action_steps == [frame - 1 for frame in frame_steps], f"step {step}"


def test_the_empty_action_is_exactly_frame_zeros_slot():
    """Only a frame-0 slot is empty -- including step 1, whose padded history is [x0, x0].

    That duplicate is not a special case to strip: the pair `(frame, action)` is repeated
    whole, so a repeated frame 0 correctly repeats the empty action beside it.
    """
    for step in range(1, MAX_STEP):
        frame_steps = blocks.context_frame_steps(step)
        action_steps = blocks.context_action_steps(step)
        empty = [i for i, action in enumerate(action_steps) if action == blocks.EMPTY_ACTION_STEP]
        assert empty == [i for i, frame in enumerate(frame_steps) if frame == 0], f"step {step}"

    assert blocks.context_action_steps(1) == [-1, -1]
    assert blocks.context_action_steps(2) == [-1, 0]


def test_padding_repeats_the_whole_frame_action_pair():
    """The padded frame brings its own action along, so the pair stays consistent."""
    assert blocks.context_frame_steps(3) == [0, 1, 2, 2]
    assert blocks.context_action_steps(3) == [-1, 0, 1, 1]


def test_block_action_steps_line_up_with_block_labels():
    for step in range(1, MAX_STEP):
        per_block = blocks.context_block_action_steps(step)
        assert len(per_block) == blocks.block_count(step)
        assert all(len(entry) == blocks.FRAMES_PER_BLOCK for entry in per_block)
        assert [action for entry in per_block for action in entry] == blocks.context_action_steps(step)


# --- 3. the streaming session reproduces the training stream ------------------------


class _RecordingProcessor:
    """Minimal stand-in for the Qwen processor: records what the session asked to encode."""

    class _Tokenizer:
        def __call__(self, labels, add_special_tokens=False):
            return {"input_ids": [[len(label)] for label in labels]}

        def pad(self, features, padding=True, return_tensors="pt"):
            return {
                "input_ids": torch.tensor([f["input_ids"] for f in features]),
                "attention_mask": torch.ones(len(features), 1, dtype=torch.long),
            }

    def __init__(self):
        self.tokenizer = self._Tokenizer()
        self.video_processor = type("_VP", (), {"temporal_patch_size": 2})()
        self.calls: list[list] = []
        self.metadata: list = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return ["<video>"]

    def __call__(self, text=None, videos=None, **kwargs):
        self.calls.append(list(videos[0]))
        self.metadata.append(kwargs.get("video_metadata"))
        grid = torch.tensor([[len(video) // 2, 2, 2] for video in videos], dtype=torch.long)
        return {"video_grid_thw": grid, "pixel_values_videos": torch.zeros(1)}


class _RecordingModel:
    """Stand-in for `ModulatedPretrainedModel`'s interactive API."""

    def __init__(self):
        self.interleave_timestamp = True

    def new_interactive_state(self):
        return {"folds": []}

    def interactive_update(self, encoder_inputs, state):
        state["folds"].append(encoder_inputs)
        return state

    def interactive_generate_weights(self, state):
        return {"folds": list(state["folds"])}, None


def _session_stream(step_count: int):
    """Block stream the session presents at each step, as (frame steps, label) pairs."""
    from malow.vla.c2l.session import C2LVLASession

    processor = _RecordingProcessor()
    session = C2LVLASession(
        model=_RecordingModel(), processor=processor, device=None,
        step_seconds=BRIDGE_STEP_SECONDS,
    )

    per_step = []
    for step in range(step_count):
        # A frame whose pixels identify its env step, so a mis-ordered or stale frame in
        # the block stream is visible rather than merely plausible.
        frame = torch.full((2, 2, 3), step, dtype=torch.uint8).numpy()
        processor.calls.clear()
        lora = session.step(frame)
        if lora is None:
            per_step.append([])
            continue
        stream = []
        for fold in lora["folds"]:
            stream.append(int(fold["timestamp_input_ids"][0, 0]))
        per_step.append(stream)
    return per_step, session


def test_session_frames_and_labels_match_the_training_stream():
    """The end-to-end claim: same episode, same step -> byte-identical block stream."""
    from malow.vla.c2l.session import C2LVLASession

    processor = _RecordingProcessor()
    session = C2LVLASession(
        model=_RecordingModel(), processor=processor, device=None,
        step_seconds=BRIDGE_STEP_SECONDS,
    )

    committed_encodings: list[tuple[tuple[int, ...], str]] = []
    for step in range(MAX_STEP):
        frame = torch.full((2, 2, 3), step, dtype=torch.uint8).numpy()
        before = len(committed_encodings)
        processor.calls.clear()
        provisional = None

        # Re-derive what the session encoded from the frames handed to the processor and
        # the labels it attached; both come from the session, not from `blocks`.
        lora = session.step(frame)
        encoded = list(processor.calls)

        newly_committed = blocks.committed_block_count(step) - before
        for offset in range(newly_committed):
            frames = encoded[offset]
            committed_encodings.append(
                (
                    tuple(int(f.getpixel((0, 0))[0]) for f in frames),
                    blocks.committed_block_label(before + offset, BRIDGE_STEP_SECONDS),
                )
            )
        if blocks.has_provisional_block(step) and step > 0:
            frames = encoded[-1]
            provisional = (
                tuple(int(f.getpixel((0, 0))[0]) for f in frames),
                blocks.provisional_block_label(step, BRIDGE_STEP_SECONDS),
            )

        if step == 0:
            assert lora is None
            continue

        streamed = [entry for entry in committed_encodings]
        if provisional is not None:
            streamed.append(provisional)

        expected_frames = blocks.context_frame_steps(step)
        expected = [
            (
                tuple(expected_frames[i * 2 : i * 2 + 2]),
                blocks.context_block_labels(step, BRIDGE_STEP_SECONDS)[i],
            )
            for i in range(blocks.block_count(step))
        ]
        assert streamed == expected, f"block stream diverged at env step {step}"


def test_session_prunes_frames_to_the_two_most_recent():
    """Streaming is O(1) in episode length, not just in compute."""
    from malow.vla.c2l.session import C2LVLASession

    session = C2LVLASession(
        model=_RecordingModel(), processor=_RecordingProcessor(), device=None,
        step_seconds=BRIDGE_STEP_SECONDS,
    )
    for step in range(MAX_STEP):
        session.step(torch.full((2, 2, 3), step, dtype=torch.uint8).numpy())
        assert len(session._frames) <= blocks.FRAMES_PER_BLOCK


def test_reset_clears_the_session():
    """Without this, episode n+1 begins remembering episode n."""
    from malow.vla.c2l.session import C2LVLASession

    session = C2LVLASession(
        model=_RecordingModel(), processor=_RecordingProcessor(), device=None,
        step_seconds=BRIDGE_STEP_SECONDS,
    )
    for step in range(6):
        session.step(torch.full((2, 2, 3), step, dtype=torch.uint8).numpy())
    # The last observation is env step 5, whose history [0, 4] holds two whole blocks.
    assert session.committed_blocks == blocks.committed_block_count(5) == 2
    session.reset()
    assert session.committed_blocks == 0
    assert session.next_step == 0
    assert session._frames == {}
    assert session._actions == {}
    assert session.step(torch.zeros((2, 2, 3), dtype=torch.uint8).numpy()) is None


# --- 3c. the streaming action history ------------------------------------------------

ACTION_DIM = 7


def _action_for(step: int):
    """A distinct action per env step, so a stale or mis-indexed one is visible."""
    return np.full(ACTION_DIM, float(step) + 0.5, dtype=np.float32)


def _run_session_with_actions(step_count: int):
    from malow.vla.c2l.session import C2LVLASession

    processor = _RecordingProcessor()
    session = C2LVLASession(
        model=_RecordingModel(),
        processor=processor,
        device=None,
        interleave_action_history=True,
        action_dim=ACTION_DIM,
        step_seconds=BRIDGE_STEP_SECONDS,
    )

    folds_per_step = []
    for step in range(step_count):
        frame = torch.full((2, 2, 3), step, dtype=torch.uint8).numpy()
        lora = session.step(frame)
        folds_per_step.append([] if lora is None else list(lora["folds"]))
        # The policy records its own prediction for this step before the next one.
        session.record_action(_action_for(step))
    return session, folds_per_step


def test_the_session_folds_the_action_the_block_policy_asks_for():
    """Streaming actions must equal what training would build for the same step."""
    session, folds_per_step = _run_session_with_actions(MAX_STEP)

    for step in range(1, MAX_STEP):
        expected_blocks = blocks.context_block_action_steps(step)
        streamed = folds_per_step[step]
        assert len(streamed) == len(expected_blocks), f"step {step}"
        for block_idx, (fold, action_steps) in enumerate(zip(streamed, expected_blocks)):
            values = fold["action_history_values"][0]
            is_empty = fold["action_history_is_empty"][0]
            for position, action_step in enumerate(action_steps):
                if action_step == blocks.EMPTY_ACTION_STEP:
                    assert bool(is_empty[position]), f"step {step}, block {block_idx}"
                    continue
                assert not bool(is_empty[position]), f"step {step}, block {block_idx}"
                np.testing.assert_allclose(
                    values[position].numpy(),
                    _action_for(action_step),
                    err_msg=f"step {step}, block {block_idx}, slot {position}",
                )


def test_the_session_prunes_the_action_buffer():
    """Streaming stays O(1) in episode length with the action buffer too.

    The bound is four, not two: an even step ``t`` folds the block whose actions are
    ``[a_{t-3}, a_{t-2}]``, so pruning can only drop below ``t-3``, and the policy then
    records ``a_t`` on top of the three that survived.
    """
    session, _ = _run_session_with_actions(MAX_STEP)
    assert len(session._actions) <= 4


def test_a_missing_prediction_is_an_error_not_a_zero():
    """Forgetting `record_action` must fail loudly rather than fold a silent zero."""
    from malow.vla.c2l.session import C2LVLASession

    session = C2LVLASession(
        model=_RecordingModel(),
        processor=_RecordingProcessor(),
        device=None,
        interleave_action_history=True,
        action_dim=ACTION_DIM,
        step_seconds=BRIDGE_STEP_SECONDS,
    )
    session.step(torch.zeros((2, 2, 3), dtype=torch.uint8).numpy())  # step 0, no LoRA
    session.record_action(_action_for(0))
    session.step(torch.ones((2, 2, 3), dtype=torch.uint8).numpy())  # step 1
    # Step 1's prediction is never recorded. Step 2 gets away with it -- block 0 covers
    # frames [0, 1] and so wants `[a_-1, a_0]` -- but step 3's provisional block covers
    # `[x2, x2]` and wants `[a_1, a_1]`, which was never supplied.
    session.step(torch.full((2, 2, 3), 2, dtype=torch.uint8).numpy())
    session.record_action(_action_for(2))
    with pytest.raises(RuntimeError, match="missing its own prediction"):
        session.step(torch.full((2, 2, 3), 3, dtype=torch.uint8).numpy())


def test_a_wrong_width_prediction_is_rejected():
    from malow.vla.c2l.session import C2LVLASession

    session = C2LVLASession(
        model=_RecordingModel(),
        processor=_RecordingProcessor(),
        device=None,
        interleave_action_history=True,
        action_dim=ACTION_DIM,
        step_seconds=BRIDGE_STEP_SECONDS,
    )
    session.step(torch.zeros((2, 2, 3), dtype=torch.uint8).numpy())
    with pytest.raises(ValueError, match="7-dim actions"):
        session.record_action(np.zeros(3, dtype=np.float32))


def test_action_interleaving_needs_the_action_width():
    from malow.vla.c2l.session import C2LVLASession

    with pytest.raises(ValueError, match="action_dim"):
        C2LVLASession(
            model=_RecordingModel(),
            processor=_RecordingProcessor(),
            device=None,
            interleave_action_history=True,
            step_seconds=BRIDGE_STEP_SECONDS,
        )


def test_recording_actions_is_a_no_op_when_the_stream_is_off():
    """A checkpoint without the stream must not accumulate state it never reads."""
    from malow.vla.c2l.session import C2LVLASession

    session = C2LVLASession(
        model=_RecordingModel(),
        processor=_RecordingProcessor(),
        device=None,
        step_seconds=BRIDGE_STEP_SECONDS,
    )
    session.step(torch.zeros((2, 2, 3), dtype=torch.uint8).numpy())
    session.record_action(_action_for(0))
    assert session._actions == {}


# --- 3b. the committed/provisional split is numerically exact ------------------------

LATENT = 128
NUM_BASE_LAYERS = 3
EXACT = {"rtol": 1e-4, "atol": 1e-4}


def _build_module() -> TemporalGatedDeltaNet:
    torch.manual_seed(0)
    module = TemporalGatedDeltaNet(LATENT, NUM_BASE_LAYERS, num_hidden_layers=2).eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0, 0.3)
        # g = -exp(A_log) * softplus(a + dt_bias). A tiny A drives g to 0, so nothing is
        # forgotten and a dropped carry cannot hide behind decay.
        for block in module.blocks:
            block.mixer.A_log.fill_(math.log(1e-4))
            block.mixer.dt_bias.fill_(-4.0)
            block.mixer.conv1d.weight.normal_(0.0, 0.5)
    return module


def _run(module, sequence, cache=None):
    # `torch.inference_mode`, not `no_grad`: `interactive_update` is decorated with it, so
    # the cached conv/recurrent tensors are *inference* tensors. That distinction is what
    # the clone below has to survive, and `no_grad` would not reproduce it.
    with torch.inference_mode():
        return module(sequence, torch.ones(sequence.shape[:2], dtype=torch.long), cache_params=cache)


def test_streaming_with_a_provisional_clone_matches_a_single_pass():
    """Fold committed blocks once; fold the padded block into a clone that is discarded.

    The comparison target at every step is a single pass over the *whole* padded block
    stream for that step -- exactly what training computes. Running it across a whole
    episode also proves the clone does not corrupt the session state: if the clone shared
    the cache, an odd step's provisional fold would poison the committed chain and the next
    even step would disagree.
    """
    from malow.vla.c2l.session import clone_interactive_state

    module = _build_module()
    torch.manual_seed(1)
    # Block j of the committed stream covers frames [2j, 2j+1]. The padded block that
    # temporarily occupies position j covers [2j, 2j] and is therefore a different input.
    committed_embeddings = torch.randn(1, MAX_STEP, LATENT)
    provisional_embeddings = torch.randn(1, MAX_STEP, LATENT)

    cache = module.new_cache()
    committed = 0
    last_committed_output = None
    checked_odd = checked_even = 0

    for step in range(1, MAX_STEP):
        target = blocks.committed_block_count(step)
        while committed < target:
            last_committed_output = _run(
                module, committed_embeddings[:, committed : committed + 1], cache
            )
            committed += 1

        if blocks.has_provisional_block(step):
            working_cache = clone_interactive_state(cache)
            provisional = provisional_embeddings[:, target : target + 1]
            streamed = _run(module, provisional, working_cache)
            offline_sequence = torch.cat([committed_embeddings[:, :target], provisional], dim=1)
            checked_odd += 1
        else:
            streamed = last_committed_output
            offline_sequence = committed_embeddings[:, :target]
            checked_even += 1

        offline = _run(module, offline_sequence)[:, -1:]
        torch.testing.assert_close(
            streamed, offline, msg=lambda text, step=step: f"env step {step}: {text}", **EXACT
        )

    assert checked_odd and checked_even, "the loop must exercise both padded and whole tails"


def test_dropping_the_carry_would_be_visible():
    """Negative control: without it the equivalence test above could pass vacuously."""
    module = _build_module()
    torch.manual_seed(1)
    block_embeddings = torch.randn(1, 8, LATENT)

    offline_tail = _run(module, block_embeddings)[:, -1:]
    cold = _run(module, block_embeddings[:, -1:])
    error = (cold - offline_tail).abs().max()
    assert error > 100 * EXACT["atol"], (
        f"the control is too weak to detect a dropped state: {error:.3e}"
    )
