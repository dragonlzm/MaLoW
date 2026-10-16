"""End-to-end wiring of OVO-Bench interactive Video C2L evaluation.

The model and the video decoder are stubbed; what is under test is the driver: that a
source video's questions become one session in video-clock order, that each turn advances
the stream by exactly its own footage, that a question arriving before any context falls
back to the offline path, and that the result rows keep the offline schema.
"""

from __future__ import annotations

import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from malow.video import interactive as c2l_interactive
from malow.video import ovo as runner

class FakeState:
    def __init__(self) -> None:
        self.segments: list[int] = []


class FakeModel:
    """Records how many blocks each interactive_update was given."""

    frame_per_lora = 2
    interleave_timestamp = False
    device = "cpu"

    def __init__(self) -> None:
        self.states: list[FakeState] = []
        self.generated_with: list[str] = []

    def supports_interactive_c2l(self):
        return True, ""

    def new_interactive_state(self):
        state = FakeState()
        self.states.append(state)
        return state

    def interactive_update(self, encoder_inputs, state):
        state.segments.append(len(encoder_inputs["frames"]))
        return state

    def interactive_generate_weights(self, state):
        return {"lora": {"A": sum(state.segments)}}, None

    def generate(self, generated_loras=None, **kwargs):
        self.generated_with.append(generated_loras)
        return [[0]]


class FakeProcessor:
    tokenizer = Namespace(pad_token_id=0, eos_token_id=1)

    def apply_chat_template(self, messages, **kwargs):
        return ["<video>" for _ in messages]


def make_args(**overrides):
    values = {
        "interactive_session_key": "video",
        "interactive_sample_fps": 1.0,
        "base_tail_seconds": 2.0,
        "interactive_max_encoder_frames": None,
        "interactive_verify_sessions": 0,
        "base_input_video": True,
        "base_with_last_2seconds": True,
        "base_input_frame_number": 2,
        "num_frames": 766,
        "max_frames": 766,
        "video_fps": 1.0,
        "no_video": False,
        "enable_thinking": "false",
    }
    values.update(overrides)
    return Namespace(**values)


def make_units(times: dict[int, float], video: str = "vidA") -> list[runner.EvalUnit]:
    return [
        runner.EvalUnit(
            task_name="ovo_backward",
            doc={"id": doc_id, "video": video, "realtime": seconds, "task": "EPM", "gt": 0, "options": ["a", "b"]},
            prompt=f"question {doc_id}",
            video_path=f"/videos/{doc_id}.mp4",
            round_idx=None,
        )
        for doc_id, seconds in times.items()
    ]


class InteractiveSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = {1: 30.0, 2: 10.0, 3: 3.0}
        self.units = make_units(self.times)
        self.durations = {f"/videos/{doc_id}.mp4": seconds for doc_id, seconds in self.times.items()}
        patches = [
            mock.patch.object(c2l_interactive, "video_duration_seconds", lambda path: None),
            mock.patch.object(runner, "video_duration_seconds", lambda path: self.durations[path]),
            mock.patch.object(runner, "build_video_only_messages", lambda path, args: [{"role": "user"}]),
            mock.patch.object(runner, "annotate_video_temporal_patch_size", lambda processor, inputs: inputs),
            mock.patch.object(runner, "prepare_c2l_base_inputs", lambda *a, **k: {"base": True}),
            mock.patch.object(runner, "generate_with_loras", lambda model, proc, base, loras, args: f"lora={loras}"),
            mock.patch.object(runner, "generate_c2l_batch", lambda model, proc, units, args: ["offline-fallback"]),
            mock.patch.object(c2l_interactive, "load_video_frames_at_times", lambda path, times: list(times)),
            mock.patch.object(
                c2l_interactive,
                "processor_predecoded_video_inputs",
                lambda processor, text, videos, args, metadata_fps=None, timestamps=None: FakeBatch(videos[0]),
            ),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def run_sessions(self, args=None):
        args = args or make_args()
        model, processor = FakeModel(), FakeProcessor()
        sessions = runner.build_interactive_sessions(self.units, args)
        rows, tracked = runner.run_interactive_sessions(model, processor, sessions, args, rank=0)
        return model, rows, tracked

    def test_one_session_covers_the_whole_video_in_time_order(self):
        _model, rows, tracked = self.run_sessions()
        self.assertEqual(len(tracked), 1)
        self.assertEqual([row["id"] for row in rows], [3, 2, 1])

    def test_each_turn_streams_only_its_own_new_footage(self):
        model, rows, _tracked = self.run_sessions()
        # Encoder ends at t-2s on a 1 FPS grid with 2 frames per block:
        # t=3 -> 0 blocks (fallback), t=10 -> 4 blocks, t=30 -> 14 blocks.
        self.assertEqual([row["interactive"]["blocks_consumed"] for row in rows], [0, 4, 14])
        self.assertEqual([row["interactive"]["new_blocks"] for row in rows], [0, 4, 10])
        self.assertEqual(model.states[0].segments, [8, 20])  # frames, not blocks

    def test_a_question_before_the_first_block_uses_the_offline_path(self):
        _model, rows, tracked = self.run_sessions()
        self.assertTrue(rows[0]["interactive"]["offline_fallback"])
        self.assertEqual(rows[0]["prediction"], "offline-fallback")
        self.assertFalse(rows[1]["interactive"]["offline_fallback"])
        self.assertEqual(tracked[0].fallback_turns, 1)

    def test_result_rows_keep_the_offline_schema(self):
        _model, rows, _tracked = self.run_sessions()
        for row in rows:
            for key in ("task_name", "id", "task", "round_idx", "video_path", "prompt", "prediction", "doc"):
                self.assertIn(key, row)
            self.assertIn("pred_choice", row)

    def test_questions_in_the_same_block_do_not_re_encode(self):
        self.times = {1: 10.0, 2: 11.0}
        self.units = make_units(self.times)
        self.durations = {f"/videos/{k}.mp4": v for k, v in self.times.items()}
        model, rows, _tracked = self.run_sessions()
        self.assertEqual([row["interactive"]["new_blocks"] for row in rows], [4, 0])
        self.assertEqual(model.states[0].segments, [8])

    def test_separate_videos_get_separate_sessions(self):
        self.units = make_units({1: 20.0}) + make_units({4: 20.0}, video="vidB")
        self.durations = {"/videos/1.mp4": 20.0, "/videos/4.mp4": 20.0}
        _model, rows, tracked = self.run_sessions()
        self.assertEqual(len(tracked), 2)
        self.assertEqual({row["interactive"]["session_index"] for row in rows}, {0, 1})

    def test_fixed_gap_uses_same_questions_and_frames(self):
        reference, reference_rows, _ = self.run_sessions()
        model, rows, tracked = self.run_sessions(make_args(eval_mode="fixed-gap", rollout_step_seconds=2))
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in reference_rows])
        self.assertEqual(sum(model.states[0].segments), sum(reference.states[0].segments))
        self.assertEqual(model.states[0].segments, [2] * 14)
        self.assertIsNone(tracked[0].state)

    def test_last8s_question_and_two_second_rollout(self):
        for mode in ("interactive", "fixed-gap"):
            with self.subTest(mode=mode):
                args = make_args(eval_mode=mode, base_tail_seconds=8.0, base_input_frame_number=8,
                                 rollout_step_seconds=2.0 if mode == "fixed-gap" else None)
                model, rows, tracked = self.run_sessions(args)
                self.assertEqual([r["interactive"]["blocks_consumed"] for r in rows], [0, 1, 11])
                self.assertEqual([r["interactive"]["new_blocks"] for r in rows], [0, 1, 10])
                self.assertTrue(rows[0]["interactive"]["offline_fallback"])
                self.assertEqual(model.states[0].segments, [2] * 11 if mode == "fixed-gap" else [2, 20])
                self.assertEqual(tracked[0].config.tail_seconds, 8.0)

    def test_sharding_never_splits_sessions(self):
        units = self.units + make_units({4: 30}, "vidB")
        self.durations["/videos/4.mp4"] = 30
        sessions = runner.build_interactive_sessions(units, make_args())
        shards = [runner.shard_sessions(sessions, rank, 2) for rank in range(2)]
        self.assertEqual([s for shard in shards for s in shard], sessions)
        self.assertEqual(len(shards[0][0]), 3)
        self.assertEqual(len(shards[1][0]), 1)


class FakeBatch(dict):
    def __init__(self, frames):
        super().__init__(frames=frames)

    def to(self, device):
        return self



