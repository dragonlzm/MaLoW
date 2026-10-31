"""End-to-end wiring of StreamingBench interactive Video C2L evaluation.

The legacy driver shares state within each annotation entry; opt-in video grouping
also joins repeated video paths across entries. Questions advance in video-clock
order, and proactive polling advances one `--proactive-step` at a time. The model and the
video decoder are stubbed; the control flow is real.
"""

from __future__ import annotations

import sys
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from malow.video import interactive as c2l_interactive
from malow.video import streaming as runner

class FakeState:
    def __init__(self) -> None:
        self.segments: list[int] = []


class FakeModel:
    frame_per_lora = 2
    interleave_timestamp = False
    device = "cpu"

    def __init__(self, yes_at: int | None = None) -> None:
        self.states: list[FakeState] = []
        self.prompts: list[str] = []
        self.yes_at = yes_at
        self.calls = 0

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


class FakeBatch(dict):
    def __init__(self, frames):
        super().__init__(frames=frames)

    def to(self, device):
        return self


def make_args(**overrides):
    values = {
        "interactive": True,
        "interactive_sample_fps": 1.0,
        "base_tail_seconds": 2.0,
        "interactive_max_encoder_frames": None,
        "interactive_verify_sessions": 0,
        "base_input_video": True,
        "base_with_last_2seconds": True,
        "base_input_frame_number": 2,
        "context_time": -1,
        "num_frames": 766,
        "max_frames": 766,
        "video_fps": 1.0,
        "enable_thinking": "false",
        "system_prompt": "",
        "no_video": False,
        "progress_ranks": "none",
        "proactive_step": 2,
        "proactive_extra_seconds": 4,
        "max_new_tokens": 16,
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "c2l_checkpoint": "/ckpt",
        "use_streamingbench_frame_sampling": False,
    }
    values.update(overrides)
    return Namespace(**values)


class InteractiveStreamingBenchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.written: list[dict] = []
        patches = [
            mock.patch.object(c2l_interactive, "video_duration_seconds", lambda path: None),
            mock.patch.object(runner, "absolutize_video_path", lambda path, annotation_file: path),
            mock.patch.object(runner, "validate_video_path", lambda path: None),
            mock.patch.object(runner, "done_ids", lambda path: set()),
            mock.patch.object(runner, "append_rows", lambda path, rows: self.written.extend(rows)),
            mock.patch.object(runner, "build_c2l_encoder_messages", lambda *a, **k: [{"role": "user"}]),
            mock.patch.object(runner, "apply_chat_template", lambda processor, messages, args: ["<video>"]),
            mock.patch.object(runner, "annotate_video_temporal_patch_size", lambda processor, inputs: inputs),
            mock.patch.object(runner, "prepare_c2l_base_inputs_single", lambda *a, **k: {"base": True}),
            mock.patch.object(runner, "generate_c2l_batch", lambda *a, **k: ["offline-fallback"]),
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

    def test_one_session_per_entry_advances_only_by_new_footage(self):
        entry = {
            "video_path": "/videos/a.mp4",
            "questions": [
                {"time_stamp": "00:00:10", "question": "q1", "options": ["A. x", "B. y"], "answer": "A"},
                {"time_stamp": "00:00:30", "question": "q2", "options": ["A. x", "B. y"], "answer": "B"},
            ],
        }
        model = FakeModel()
        with mock.patch.object(runner, "generate_with_loras_single", lambda *a, **k: "A"):
            tracked = runner.evaluate_mc_task_interactive(
                "real", [(0, entry)], Path("/tmp/ann.json"), model, object(), make_args(),
                Path("/tmp/rank.jsonl"), 0, 1,
            )

        self.assertEqual(len(tracked), 1)
        # t=10 -> 4 blocks (8 frames), t=30 -> 14 blocks (10 more, 20 frames).
        self.assertEqual(model.states[0].segments, [8, 20])
        self.assertEqual([row["interactive"]["blocks_consumed"] for row in self.written], [4, 14])
        self.assertEqual([row["interactive"]["new_blocks"] for row in self.written], [4, 10])

    def test_last8s_question_and_two_second_rollout(self):
        entry = {"video_path": "/videos/a.mp4", "questions": [
            {"time_stamp": time, "question": "q", "options": ["A. x"], "answer": "A"}
            for time in ("00:00:08", "00:00:10", "00:00:30")
        ]}
        for mode in ("interactive", "fixed-gap"):
            with self.subTest(mode=mode):
                self.written.clear()
                model = FakeModel()
                args = make_args(eval_mode=mode, base_tail_seconds=8.0, base_input_frame_number=8,
                                 rollout_step_seconds=2.0 if mode == "fixed-gap" else None)
                with mock.patch.object(runner, "generate_with_loras_single", lambda *a, **k: "A"):
                    tracked = runner.evaluate_mc_task_interactive(
                        "real", [(0, entry)], Path("/tmp/ann.json"), model, object(), args,
                        Path("/tmp/rank.jsonl"), 0, 1,
                    )
                self.assertEqual([r["interactive"]["blocks_consumed"] for r in self.written], [0, 1, 11])
                self.assertTrue(self.written[0]["interactive"]["offline_fallback"])
                self.assertEqual(model.states[0].segments, [2] * 11 if mode == "fixed-gap" else [2, 20])
                self.assertEqual(tracked[0].config.tail_seconds, 8.0)

    def test_questions_out_of_annotation_order_are_streamed_causally(self):
        entry = {
            "video_path": "/videos/a.mp4",
            "questions": [
                {"time_stamp": "00:00:30", "question": "late", "options": ["A. x"], "answer": "A"},
                {"time_stamp": "00:00:10", "question": "early", "options": ["A. x"], "answer": "A"},
            ],
        }
        model = FakeModel()
        with mock.patch.object(runner, "generate_with_loras_single", lambda *a, **k: "A"):
            runner.evaluate_mc_task_interactive(
                "real", [(0, entry)], Path("/tmp/ann.json"), model, object(), make_args(),
                Path("/tmp/rank.jsonl"), 0, 1,
            )

        self.assertEqual([row["question"] for row in self.written], ["early", "late"])
        self.assertEqual([row["interactive"]["turn_index"] for row in self.written], [0, 1])
        self.assertEqual(model.states[0].segments, [8, 20])

    def test_sqa_context_still_follows_the_annotation_order(self):
        entry = {
            "video_path": "/videos/a.mp4",
            "questions": [
                {"time_stamp": "00:00:10", "question": "first", "options": ["A. x"], "answer": "A"},
                {"time_stamp": "00:00:20", "question": "second", "options": ["A. y"], "answer": "A"},
            ],
        }
        model = FakeModel()
        with mock.patch.object(runner, "generate_with_loras_single", lambda *a, **k: "A"):
            runner.evaluate_mc_task_interactive(
                "sqa", [(0, entry)], Path("/tmp/ann.json"), model, object(), make_args(),
                Path("/tmp/rank.jsonl"), 0, 1,
            )

        prompts = [row["prompt"] for row in self.written]
        self.assertNotIn("At timestamp", prompts[0])
        self.assertIn("At timestamp 00:00:10", prompts[1])
        self.assertIn("Question: first", prompts[1])

    def test_an_early_question_falls_back_to_the_offline_path(self):
        entry = {
            "video_path": "/videos/a.mp4",
            "questions": [{"time_stamp": "00:00:03", "question": "q", "options": ["A. x"], "answer": "A"}],
        }
        model = FakeModel()
        with mock.patch.object(runner, "generate_with_loras_single", lambda *a, **k: "A"):
            tracked = runner.evaluate_mc_task_interactive(
                "real", [(0, entry)], Path("/tmp/ann.json"), model, object(), make_args(),
                Path("/tmp/rank.jsonl"), 0, 1,
            )

        self.assertEqual(model.states[0].segments, [])
        self.assertTrue(self.written[0]["interactive"]["offline_fallback"])
        self.assertEqual(tracked[0].fallback_turns, 1)

    def test_proactive_polling_advances_one_step_at_a_time(self):
        entry = {
            "video_path": "/videos/a.mp4",
            "questions": [
                {
                    "time_stamp": "00:00:20",
                    "ground_truth_time_stamp": "00:00:30",
                    "ground_truth_output": "now",
                    "question": "q",
                }
            ],
        }
        model = FakeModel()
        answers = iter(["no", "no", "no", "no", "yes", "now"])
        with (
            mock.patch.object(runner, "generate_with_loras_single", lambda *a, **k: next(answers)),
            mock.patch.object(runner, "can_decode_video_time_range", lambda *a, **k: True),
        ):
            tracked = runner.evaluate_proactive_task_interactive(
                [(0, entry)], Path("/tmp/ann.json"), model, object(), make_args(),
                Path("/tmp/rank.jsonl"), 0, 1,
            )

        row = self.written[0]
        self.assertEqual(row["answered_time"], 32)
        self.assertEqual(row["prediction"], "now")
        self.assertEqual(row["trigger_hit"], 1)
        self.assertEqual(row["output_hit"], 1)
        # The window starts at the question, so a poll at t sees floor((t-22)/2) blocks:
        # t=22 has none and falls back, then 24..32 add exactly one block (2 frames) each.
        self.assertEqual(model.states[0].segments, [2, 2, 2, 2, 2])
        self.assertEqual(tracked[0].fallback_turns, 1)
        # The final answer is asked at the same moment, so it must not advance the stream.
        self.assertEqual(len([step for step in row["dialog_history"] if step["role"] == "assistant"]), 7)

    def test_proactive_sessions_do_not_share_state_across_questions(self):
        entry = {
            "video_path": "/videos/a.mp4",
            "questions": [
                {"time_stamp": "00:00:20", "ground_truth_time_stamp": "00:00:22",
                 "ground_truth_output": "a", "question": "q1"},
                {"time_stamp": "00:01:00", "ground_truth_time_stamp": "00:01:02",
                 "ground_truth_output": "b", "question": "q2"},
            ],
        }
        model = FakeModel()
        with (
            mock.patch.object(runner, "generate_with_loras_single", lambda *a, **k: "no"),
            mock.patch.object(runner, "can_decode_video_time_range", lambda *a, **k: True),
        ):
            tracked = runner.evaluate_proactive_task_interactive(
                [(0, entry)], Path("/tmp/ann.json"), model, object(), make_args(),
                Path("/tmp/rank.jsonl"), 0, 1,
            )

        self.assertEqual(len(tracked), 2)
        self.assertEqual(len(model.states), 2)
        # Each question restarts the window at its own timestamp: polls at +2/+4/+6 seconds
        # give 0 (fallback), 1 and 2 blocks, so two one-block segments per session.
        for state in model.states:
            self.assertEqual(state.segments, [2, 2])

    def test_resume_replays_completed_questions_on_original_tick_grid(self):
        questions = [
            {"time_stamp": "00:00:10.5", "question": "done", "options": ["A. x"], "answer": "A"},
            {"time_stamp": "00:00:20.5", "question": "pending", "options": ["A. y"], "answer": "A"},
        ]
        entry = {"video_path": "/videos/a.mp4", "questions": questions}
        qid = runner.question_id("sqa", 0, 0, questions[0])
        model = FakeModel()
        with mock.patch.object(runner, "done_ids", lambda _: {qid}), mock.patch.object(runner, "generate_with_loras_single", lambda *a: "A"):
            tracked = runner.evaluate_mc_task_interactive(
                "sqa", [(0, entry)], Path("/tmp/ann.json"), model, object(),
                make_args(eval_mode="fixed-gap", rollout_step_seconds=3), Path("/tmp/rank.jsonl"), 0, 1)
        self.assertEqual(len(self.written), 1)
        self.assertEqual(self.written[0]["question"], "pending")
        self.assertIn("Question: done", self.written[0]["prompt"])
        self.assertEqual(sum(model.states[0].segments), 18)
        self.assertEqual(tracked[0].consumed_blocks, 9)
        self.assertIsNone(tracked[0].state)

    def test_proactive_fixed_gap_keeps_one_second_polling(self):
        entry = {"video_path": "/videos/a.mp4", "questions": [{
            "time_stamp": "00:00:20", "ground_truth_time_stamp": "00:00:26",
            "ground_truth_output": "now", "question": "q"}]}
        model = FakeModel()
        with mock.patch.object(runner, "generate_with_loras_single", lambda *a: "no"), mock.patch.object(runner, "can_decode_video_time_range", lambda *a, **kw: True):
            runner.evaluate_proactive_task_interactive(
                [(0, entry)], Path("/tmp/ann.json"), model, object(),
                make_args(eval_mode="fixed-gap", rollout_step_seconds=2, proactive_step=1), Path("/tmp/rank.jsonl"), 0, 1)
        times = [r["time"] for r in self.written[0]["dialog_history"] if r["role"] == "user"]
        self.assertEqual(times, list(range(21, 31)))
        self.assertEqual(model.states[0].segments, [2, 2, 2, 2])

    def test_unsorted_sqa_prompts_equal_existing_evaluator(self):
        entry = {"video_path": "/videos/a.mp4", "questions": [
            {"time_stamp": "00:00:20", "question": "first", "options": ["A. x"], "answer": "A"},
            {"time_stamp": "00:00:10", "question": "second", "options": ["A. y"], "answer": "A"}]}
        model = FakeModel()
        with mock.patch.object(runner, "generate_batch", lambda *a, **kw: ["A"]):
            runner.evaluate_mc_task("sqa", [(0, entry)], Path("/tmp/ann.json"), model, object(),
                                    make_args(batch_size=1), Path("/tmp/rank.jsonl"), 0, 1)
        reference = {r["question_id"]: r["prompt"] for r in self.written}
        self.written.clear()
        with mock.patch.object(runner, "generate_with_loras_single", lambda *a: "A"):
            runner.evaluate_mc_task_interactive("sqa", [(0, entry)], Path("/tmp/ann.json"), model, object(),
                                                make_args(), Path("/tmp/rank.jsonl"), 0, 1)
        self.assertEqual(reference, {r["question_id"]: r["prompt"] for r in self.written})



