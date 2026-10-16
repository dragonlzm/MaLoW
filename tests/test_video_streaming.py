# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
#!/usr/bin/env python
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from unittest import mock


from malow.video import streaming as runner


def make_args(**overrides):
    values = {
        "video_min_pixels": 784,
        "video_max_pixels": 151200,
        "video_total_pixels": 0,
        "video_fps": 2.0,
        "max_frames": 32,
        "num_frames": None,
        "use_streamingbench_frame_sampling": False,
        "last_2second_1fps": False,
        "last_seconds_1fps": None,
    }
    values.update(overrides)
    return type("Args", (), values)()


class StreamingBenchEvaluatorTest(unittest.TestCase):
    def test_c2l_vision_encoder_only_cli_is_opt_in(self) -> None:
        required = [
            "prog",
            "--model-id",
            "qwen35",
            "--c2l-checkpoint",
            "checkpoint",
            "--output-dir",
            "/tmp/output",
        ]
        with mock.patch.object(sys, "argv", required):
            self.assertFalse(runner.parse_args().vision_encoder_only)
        with mock.patch.object(sys, "argv", [*required, "--vision-encoder-only"]):
            self.assertTrue(runner.parse_args().vision_encoder_only)

    def test_low_fps_window_is_padded_backward_for_qwen_reader(self) -> None:
        original_probe = runner.probe_video_frame_info
        try:
            fps = 0.7645848499989565
            runner.probe_video_frame_info = lambda _path: (768, fps)

            content = runner.make_video_content(
                "prepared.mp4",
                203,
                204,
                make_args(use_streamingbench_frame_sampling=True),
            )

            start_frame = math.ceil(content["video_start"] * fps)
            end_frame = math.floor(content["video_end"] * fps)
            self.assertLess(start_frame, end_frame)
            self.assertLess(content["video_start"], 203)
            self.assertEqual(content["video_end"], 204)
            self.assertEqual(content["fps"], 1.0)
            self.assertEqual(content["max_frames"], runner.STREAMINGBENCH_MAX_FRAMES)
            self.assertTrue(runner.can_decode_video_time_range("prepared.mp4", 203, 204, 2))
        finally:
            runner.probe_video_frame_info = original_probe

    def test_impossible_initial_low_fps_window_is_not_decodable(self) -> None:
        original_probe = runner.probe_video_frame_info
        try:
            runner.probe_video_frame_info = lambda _path: (768, 0.7200900100011113)

            self.assertFalse(runner.can_decode_video_time_range("prepared.mp4", 0, 1, 2))
            self.assertTrue(runner.can_decode_video_time_range("prepared.mp4", 0, 2, 2))
        finally:
            runner.probe_video_frame_info = original_probe

    def test_high_fps_window_is_left_unchanged(self) -> None:
        original_probe = runner.probe_video_frame_info
        try:
            runner.probe_video_frame_info = lambda _path: (9000, 30.0)

            content = runner.make_video_content("regular.mp4", 203, 204, make_args())

            self.assertEqual(content["video_start"], 203)
            self.assertEqual(content["video_end"], 204)
        finally:
            runner.probe_video_frame_info = original_probe

    def test_last_2second_1fps_uses_tail_window(self) -> None:
        original_probe = runner.probe_video_frame_info
        try:
            runner.probe_video_frame_info = lambda _path: (9000, 30.0)

            content = runner.make_video_content("regular.mp4", 10, 20, make_args(last_2second_1fps=True))

            self.assertEqual(content["video_start"], 18.0)
            self.assertEqual(content["video_end"], 20.0)
            self.assertEqual(content["fps"], 1.0)
            self.assertEqual(content["max_frames"], 2)
            self.assertNotIn("nframes", content)
        finally:
            runner.probe_video_frame_info = original_probe

    def test_last_2second_1fps_keeps_short_window_at_1fps(self) -> None:
        original_probe = runner.probe_video_frame_info
        try:
            runner.probe_video_frame_info = lambda _path: (9000, 30.0)

            content = runner.make_video_content("regular.mp4", 10, 11, make_args(last_2second_1fps=True))

            self.assertEqual(content["video_start"], 10.0)
            self.assertEqual(content["video_end"], 11.0)
            self.assertEqual(content["fps"], 1.0)
            self.assertEqual(content["max_frames"], 2)
        finally:
            runner.probe_video_frame_info = original_probe

    def test_last_seconds_1fps_uses_tail_window(self) -> None:
        original_probe = runner.probe_video_frame_info
        try:
            runner.probe_video_frame_info = lambda _path: (60000, 30.0)

            content = runner.make_video_content("regular.mp4", 100, 1000, make_args(last_seconds_1fps=600))

            self.assertEqual(content["video_start"], 400.0)
            self.assertEqual(content["video_end"], 1000.0)
            self.assertEqual(content["fps"], 1.0)
            self.assertEqual(content["max_frames"], 600)
        finally:
            runner.probe_video_frame_info = original_probe

    def test_last_seconds_1fps_keeps_short_window(self) -> None:
        original_probe = runner.probe_video_frame_info
        try:
            runner.probe_video_frame_info = lambda _path: (9000, 30.0)

            content = runner.make_video_content("regular.mp4", 100, 200, make_args(last_seconds_1fps=600))

            self.assertEqual(content["video_start"], 100.0)
            self.assertEqual(content["video_end"], 200.0)
            self.assertEqual(content["fps"], 1.0)
            self.assertEqual(content["max_frames"], 600)
        finally:
            runner.probe_video_frame_info = original_probe

    def test_base_last2_c2l_preflight_does_not_require_encoder_cap(self) -> None:
        base_last2_args = make_args(
            num_frames=766,
            c2l_checkpoint="checkpoint-2200",
            base_input_video=True,
            base_input_frame_number=2,
            base_with_last_2seconds=True,
        )
        split_video_args = make_args(
            num_frames=766,
            c2l_checkpoint="checkpoint-2200",
            base_input_video=True,
            base_input_frame_number=2,
            base_with_last_2seconds=False,
        )

        self.assertEqual(runner.min_decodable_source_frames(base_last2_args), 2)
        self.assertEqual(runner.min_decodable_source_frames(split_video_args), 766)

    def test_weighted_summary_uses_standard_split_sizes(self) -> None:
        weighted = runner.build_weighted_summary(
            {
                "real": {"accuracy": 0.83},
                "omni": {"accuracy": 0.576},
                "proactive": {"accuracy": 0.096},
                "sqa": {"accuracy": 0.568},
            }
        )

        self.assertEqual(
            weighted,
            {
                "weights": {"real": 2500, "omni": 1500, "proactive": 250, "sqa": 250},
                "total_questions": 4500,
                "accuracy": 0.69,
                "accuracy_percent": 69.0,
            },
        )

    def test_weighted_summary_requires_all_standard_splits(self) -> None:
        weighted = runner.build_weighted_summary(
            {
                "real": {"accuracy": 0.83},
                "omni": {"accuracy": 0.576},
                "sqa": {"accuracy": 0.568},
            }
        )

        self.assertIsNone(weighted)


if __name__ == "__main__":
    unittest.main()
