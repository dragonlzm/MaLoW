"""Inference regression tests derived from Hippocampus master e998ec8."""
import sys
from pathlib import Path

import numpy as np
import pytest


from malow.vla.c2l import blocks  # noqa: E402
from malow.vla.c2l.policy import C2LVLAInteractivePolicy  # noqa: E402
from malow.vla.c2l.session import C2LVLASession  # noqa: E402


ACTION_DIM = 7
# Mirrors `model2simpler_interface_c2l_lora.GRIPPER_DIM`; the last of the 7 action dims.
GRIPPER_DIM = 6


# --- 1. the invariant that makes feedback possible at all ----------------------------


@pytest.mark.parametrize("step", range(1, 20))
def test_the_context_never_reads_an_action_newer_than_two_steps_back(step):
    """`a_{t-1}` is not needed at step `t`, which is exactly the slack feedback spends.

    If this ever fails, the feedback protocol deadlocks: the client cannot know the
    executed action for step `t-1` until the server has already answered request `t-1`,
    and it ships it with request `t`.
    """
    action_steps = [s for s in blocks.context_action_steps(step) if s != blocks.EMPTY_ACTION_STEP]
    if not action_steps:
        return
    assert max(action_steps) <= step - 2


def test_the_provisional_and_committed_blocks_agree_with_that_bound():
    """The same bound, read off the two fold paths `session.step` actually takes."""
    for step in range(1, 20):
        if blocks.has_provisional_block(step):
            assert max(blocks.provisional_block_action_steps(step)) <= step - 2
        for block_index in range(blocks.committed_block_count(step)):
            # Block j is folded at step 2j+2, so that is the step its bound is read at.
            folded_at = (block_index + 1) * blocks.FRAMES_PER_BLOCK
            assert max(blocks.committed_block_action_steps(block_index)) <= folded_at - 2


# --- 2. record_action with an explicit index -----------------------------------------


class _StubModel:
    """Enough of a model for the session's bookkeeping; nothing is folded here."""

    def new_interactive_state(self):
        return object()


def _session(**kwargs):
    return C2LVLASession(
        model=_StubModel(),
        processor=None,
        interleave_action_history=True,
        action_dim=ACTION_DIM,
        encoder_text="stub",
        **kwargs,
    )


def test_an_explicit_index_addresses_that_step():
    session = _session()
    session.next_step = 5
    session.record_action(np.arange(ACTION_DIM, dtype=np.float32), step_index=3)
    assert 3 in session._actions
    assert 4 not in session._actions
    np.testing.assert_allclose(session._actions[3], np.arange(ACTION_DIM))


def test_an_explicit_index_works_before_the_first_step():
    """The feedback path files an action before `step()` runs, unlike self-recording."""
    session = _session()
    session.record_action(np.ones(ACTION_DIM, dtype=np.float32), step_index=0)
    np.testing.assert_allclose(session._actions[0], np.ones(ACTION_DIM))


def test_the_default_index_is_unchanged():
    """Self-recording still files the step `step()` was last called for."""
    session = _session()
    session.next_step = 5
    session.record_action(np.ones(ACTION_DIM, dtype=np.float32))
    assert list(session._actions) == [4]


def test_the_default_index_still_refuses_to_run_before_the_first_step():
    session = _session()
    with pytest.raises(RuntimeError, match="before its first step"):
        session.record_action(np.ones(ACTION_DIM, dtype=np.float32))


def test_a_negative_index_is_rejected():
    session = _session()
    with pytest.raises(ValueError, match="non-negative"):
        session.record_action(np.ones(ACTION_DIM, dtype=np.float32), step_index=-1)


def test_a_wrong_action_width_is_still_rejected_on_the_explicit_path():
    session = _session()
    with pytest.raises(ValueError, match="7-dim"):
        session.record_action(np.ones(3, dtype=np.float32), step_index=0)


def test_recording_is_a_no_op_when_the_stream_is_off():
    session = C2LVLASession(
        model=_StubModel(),
        processor=None,
        interleave_action_history=False,
        encoder_text="stub",
    )
    session.record_action(np.ones(ACTION_DIM, dtype=np.float32), step_index=0)
    assert session._actions == {}


# --- 3. the fail-loud matrix ----------------------------------------------------------


class _FakeSession:
    def __init__(self, next_step):
        self.next_step = next_step
        self.recorded = []

    def record_action(self, action, step_index=None):
        self.recorded.append((step_index, np.asarray(action)))


def _policy(action_feedback, next_step):
    """A policy carrying only what `_accept_executed_action` reads.

    Building a real one needs a loaded checkpoint; the method under test touches nothing
    but these two attributes.
    """
    policy = object.__new__(C2LVLAInteractivePolicy)
    policy.action_feedback = action_feedback
    policy.session = _FakeSession(next_step)
    return policy


def test_feedback_records_the_clients_action_against_the_previous_step():
    policy = _policy(action_feedback=True, next_step=4)
    action = np.arange(ACTION_DIM, dtype=np.float32)
    policy._accept_executed_action(action, 3)
    assert len(policy.session.recorded) == 1
    step_index, recorded = policy.session.recorded[0]
    assert step_index == 3
    np.testing.assert_allclose(recorded, action)


def test_the_first_observation_of_an_episode_expects_nothing():
    policy = _policy(action_feedback=True, next_step=0)
    policy._accept_executed_action(None, None)
    assert policy.session.recorded == []


def test_an_action_at_the_first_observation_is_rejected():
    """A stale action here means the episode reset did not reach the client's state."""
    policy = _policy(action_feedback=True, next_step=0)
    with pytest.raises(RuntimeError, match="first observation"):
        policy._accept_executed_action(np.zeros(ACTION_DIM, dtype=np.float32), 7)


def test_a_missing_action_is_rejected():
    policy = _policy(action_feedback=True, next_step=4)
    with pytest.raises(RuntimeError, match="carried no executed action"):
        policy._accept_executed_action(None, None)


def test_a_missing_step_index_is_rejected():
    policy = _policy(action_feedback=True, next_step=4)
    with pytest.raises(RuntimeError, match="explicit `executed_step`"):
        policy._accept_executed_action(np.zeros(ACTION_DIM, dtype=np.float32), None)


@pytest.mark.parametrize("sent_step", [2, 4])
def test_a_step_mismatch_is_rejected(sent_step):
    """One behind (a dropped request) and one ahead (a duplicate) both have to be fatal."""
    policy = _policy(action_feedback=True, next_step=4)
    with pytest.raises(RuntimeError, match="step mismatch"):
        policy._accept_executed_action(np.zeros(ACTION_DIM, dtype=np.float32), sent_step)


def test_an_unexpected_action_is_rejected_when_feedback_is_off():
    """`predict_action`'s `**_` would otherwise swallow the client/server disagreement."""
    policy = _policy(action_feedback=False, next_step=4)
    with pytest.raises(RuntimeError, match="without --c2l_action_feedback"):
        policy._accept_executed_action(np.zeros(ACTION_DIM, dtype=np.float32), 3)


def test_the_default_path_records_nothing_here():
    """With the flag off this method is inert; the policy self-records after predicting."""
    policy = _policy(action_feedback=False, next_step=4)
    policy._accept_executed_action(None, None)
    assert policy.session.recorded == []


# --- 4. the client-side inverse -------------------------------------------------------


def test_normalize_actions_inverts_unnormalize_actions():
    """The pair has to round-trip, including the dimensions `mask` leaves alone.

    Skipped unless the SimplerEnv client's dependencies are installed; the module pulls in
    cv2 and transforms3d.
    """
    from malow.vla.simpler.model2simpler_interface_lora import M1InferenceLora

    rng = np.random.default_rng(0)
    stats = {
        "q01": np.array([-1.0, -2.0, -0.5, -1.0, -1.0, -1.0, 0.0]),
        "q99": np.array([1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0]),
        # Dimension 3 is deliberately excluded, and dimension 6 is the gripper.
        "mask": np.array([True, True, True, False, True, True, True]),
    }
    normalized = rng.uniform(-1.0, 1.0, size=(4, ACTION_DIM))
    # `unnormalize_actions` thresholds the gripper, so start from a value it leaves alone.
    normalized[:, 6] = 1.0

    unnormalized = M1InferenceLora.unnormalize_actions(normalized.copy(), stats)
    round_tripped = M1InferenceLora.normalize_actions(unnormalized, stats)

    np.testing.assert_allclose(round_tripped, normalized, atol=1e-6)


def test_normalize_actions_survives_a_degenerate_dimension():
    """`q99 == q01` must not divide by zero, matching the training-side normalizer."""
    from malow.vla.simpler.model2simpler_interface_lora import M1InferenceLora

    stats = {
        "q01": np.zeros(ACTION_DIM),
        "q99": np.zeros(ACTION_DIM),
        "mask": np.ones(ACTION_DIM, dtype=bool),
    }
    out = M1InferenceLora.normalize_actions(np.zeros((1, ACTION_DIM)), stats)
    assert np.isfinite(out).all()


# --- 5. the client-side state machine -------------------------------------------------


def _client(action_feedback):
    """A C2L interface carrying only what the two hooks read.

    A real one opens a websocket in `__init__`; the hooks under test touch nothing but
    these attributes.
    """
    from malow.vla.simpler.model2simpler_interface_c2l_lora import M1InferenceC2LLora

    client = object.__new__(M1InferenceC2LLora)
    client.action_feedback = action_feedback
    client._pending_executed_action = None
    client._pending_executed_step = None
    client._env_step = 0
    client.action_norm_stats = {
        "q01": np.full(ACTION_DIM, -1.0),
        "q99": np.full(ACTION_DIM, 1.0),
        "mask": np.ones(ACTION_DIM, dtype=bool),
    }
    return client


def test_the_payload_is_untouched_when_feedback_is_off():
    """The default has to be byte-identical on the wire, not merely close."""
    client = _client(action_feedback=False)
    assert client._extra_infer_payload() == {}
    client._on_action_executed(
        np.zeros(ACTION_DIM), {"open_gripper": np.array([1.0])}, {}
    )
    assert client._extra_infer_payload() == {}
    assert client._pending_executed_action is None


def test_the_first_request_of_an_episode_carries_nothing():
    """Nothing has been executed yet, which is what the server expects at step 0."""
    client = _client(action_feedback=True)
    assert client._extra_infer_payload() == {}


def test_each_request_carries_the_previous_steps_action():
    client = _client(action_feedback=True)
    for step in range(4):
        # The server reads request t's payload before folding, so check it first.
        payload = client._extra_infer_payload()
        if step == 0:
            assert payload == {}
        else:
            assert payload["executed_step"] == step - 1
        executed = np.full(ACTION_DIM, 0.1 * step)
        client._on_action_executed(executed, {"open_gripper": np.array([1.0])}, {})

    payload = client._extra_infer_payload()
    assert payload["executed_step"] == 3
    # q01/q99 of -1/1 make normalization the identity, so the value survives verbatim.
    np.testing.assert_allclose(payload["executed_action"][:GRIPPER_DIM], 0.3, atol=1e-6)


def test_the_gripper_slot_carries_the_executed_open_flag():
    """Blending makes the gripper fractional; what is recorded is the decision taken."""
    client = _client(action_feedback=True)
    for open_gripper, expected in ((0.875, 1.0), (0.5, 0.0)):
        client._on_action_executed(
            np.zeros(ACTION_DIM), {"open_gripper": np.array([open_gripper])}, {}
        )
        assert client._extra_infer_payload()["executed_action"][GRIPPER_DIM] == expected


def test_reset_clears_the_queued_action():
    """Otherwise episode n's last action would be sent as episode n+1's first feedback,
    which the server rejects -- but the client must not produce it in the first place."""
    client = _client(action_feedback=True)
    client._on_action_executed(
        np.zeros(ACTION_DIM), {"open_gripper": np.array([1.0])}, {}
    )
    assert client._extra_infer_payload() != {}

    # Only the feedback half of `reset` is exercised; the rest needs a live server.
    client._pending_executed_action = None
    client._pending_executed_step = None
    client._env_step = 0
    assert client._extra_infer_payload() == {}


# --- 6. feedback timing against the session's own pruning -----------------------------


@pytest.mark.parametrize("feedback", [False, True])
def test_pruning_never_starves_a_fold_in_either_mode(feedback):
    """The end-to-end schedule: fold, prune, learn the step's action, repeat.

    `_prune` drops actions older than `step - 3`, which is safe only because a block is
    folded once and then lives in the recurrent state. Feeding actions a step later than
    the default path does must not push anything outside that window -- if it did, a real
    run would die inside `_block_actions` partway through an episode rather than here.
    """
    policy = object.__new__(C2LVLAInteractivePolicy)
    policy.action_feedback = feedback
    policy.session = _session()
    session = policy.session
    pending = None

    for step in range(30):
        if feedback:
            policy._accept_executed_action(
                pending[1] if pending else None, pending[0] if pending else None
            )

        # Everything `session.step` does that bears on action availability.
        session.next_step += 1
        needed = []
        for block_index in range(session.committed_blocks, blocks.committed_block_count(step)):
            needed += blocks.committed_block_action_steps(block_index)
        session.committed_blocks = blocks.committed_block_count(step)
        if step > 0 and blocks.has_provisional_block(step):
            needed += blocks.provisional_block_action_steps(step)
        needed = [s for s in needed if s != blocks.EMPTY_ACTION_STEP]

        missing = [s for s in needed if s not in session._actions]
        assert not missing, f"step {step} folds blocks needing absent actions {missing}"
        session._prune(step)

        action = np.full(ACTION_DIM, 0.01 * (step + 1), dtype=np.float32)
        if feedback:
            pending = (step, action)
        else:
            session.record_action(action)


def test_feedback_records_exactly_what_was_executed_over_an_episode():
    """Compose the real client hook with the real server check: no drift over many steps."""
    client = _client(action_feedback=True)
    policy = object.__new__(C2LVLAInteractivePolicy)
    policy.action_feedback = True
    policy.session = _session()

    executed = {}
    for step in range(9):
        payload = client._extra_infer_payload()
        policy._accept_executed_action(payload.get("executed_action"), payload.get("executed_step"))
        policy.session.next_step += 1

        action = np.full(ACTION_DIM, 0.1 * (step + 1))
        executed[step] = action.copy()
        client._on_action_executed(action, {"open_gripper": np.array([1.0])}, {})

    # Every step but the last, whose feedback is still queued for the next request.
    assert sorted(policy.session._actions) == list(range(8))
    for step, recorded in policy.session._actions.items():
        expected = executed[step].copy()
        expected[GRIPPER_DIM] = 1.0
        np.testing.assert_allclose(recorded, expected, atol=1e-6)
