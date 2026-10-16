# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
import sys
from unittest import mock

from malow.video import ovo as runner
from malow.video.ovo import (
    EvalUnit,
    build_baseline_messages,
    build_prompt,
    extract_count,
    extract_option_letter,
    extract_yes_no,
    score_backward_realtime,
    score_forward,
)


def test_distributed_file_barriers_use_cpu_backend_with_gpu_workers(monkeypatch):
    import torch

    monkeypatch.setenv('WORLD_SIZE', '8')
    monkeypatch.setenv('RANK', '3')
    monkeypatch.setenv('LOCAL_RANK', '3')
    with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
         mock.patch.object(torch.cuda, 'set_device') as set_device, \
         mock.patch.object(torch.distributed, 'is_initialized', return_value=False), \
         mock.patch.object(torch.distributed, 'init_process_group') as init_group:
        assert runner.init_distributed() == (3, 3, 8)
    set_device.assert_called_once_with(3)
    init_group.assert_called_once_with(backend='gloo')


def test_backward_prompt_matches_lmms_shape():
    doc = {
        "id": 1,
        "task": "EPM",
        "question": "What happened first?",
        "options": ["open", "close", "jump"],
    }

    prompt = build_prompt(doc)

    assert "Question: What happened first?" in prompt
    assert "A. open; B. close; C. jump;" in prompt
    assert "Respond only with the letter corresponding to your chosen option" in prompt


def test_c2l_vision_encoder_only_cli_is_opt_in():
    with mock.patch.object(sys, "argv", ["prog", "--c2l-checkpoint", "checkpoint"]):
        assert runner.parse_args().vision_encoder_only is False
    with mock.patch.object(
        sys,
        "argv",
        ["prog", "--c2l-checkpoint", "checkpoint", "--vision-encoder-only"],
    ):
        assert runner.parse_args().vision_encoder_only is True


def test_forward_prompt_uses_round_specific_step():
    doc = {
        "id": 2,
        "task": "SSR",
        "test_info": [{"step": "wash hands"}, {"step": "dry hands"}],
    }

    prompt = build_prompt(doc, index=1)

    assert "dry hands" in prompt
    assert "Answer only with “Yes” or “No”." in prompt


def test_backward_realtime_scoring_uses_extracted_letter():
    rows = [
        {"id": 1, "task": "EPM", "question": "q", "prediction": "A. open", "ground_truth": "A"},
        {"id": 2, "task": "EPM", "question": "q", "prediction": "A red door is visible.", "ground_truth": "A"},
    ]

    results, scores = score_backward_realtime(rows)

    assert scores["EPM"] == 50.0
    assert scores["average"] == 50.0
    assert results[0]["pred_choice"] == "A"
    assert results[1]["pred_choice"] is None


def test_option_count_and_yes_no_extraction():
    assert extract_option_letter("The answer is c.", ["open", "close", "jump", "sit"]) == "C"
    assert extract_option_letter("They close the door.", ["open", "close the door", "jump", "sit"]) == "B"
    assert extract_option_letter("A person closes the door.") is None
    assert extract_count("The answer is 3.") == "3"
    assert extract_count("A. 2") == "2"
    assert extract_yes_no("Answer: No.") == "No"
    assert extract_yes_no("Y") == "Yes"


def test_forward_scoring_groups_round_predictions():
    rows = [
        {
            "id": 1,
            "task_name": "ovo_forward",
            "task": "REC",
            "round_idx": 0,
            "prediction": "2",
            "doc": {"id": 1, "task": "REC", "test_info": [{"count": 2}]},
        },
        {
            "id": 2,
            "task_name": "ovo_forward",
            "task": "SSR",
            "round_idx": 0,
            "prediction": "Yes",
            "doc": {"id": 2, "task": "SSR", "test_info": [{"type": 1}]},
        },
    ]

    _results, scores = score_forward(rows)

    assert scores["REC"] == 100.0
    assert scores["SSR"] == 100.0
    assert scores["average"] == 100.0


def test_no_video_baseline_message_is_text_only():
    args = type("Args", (), {"no_video": True, "system_prompt": "You are a helpful assistant."})()
    unit = EvalUnit("ovo_backward", {"id": 1, "task": "EPM"}, "Question?\nA. yes", None, None)

    messages = build_baseline_messages(unit, args)

    assert messages == [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "text", "text": "Question?\nA. yes"}]},
    ]


def test_base_generation_drops_processor_video_metadata():
    import torch
    from transformers.feature_extraction_utils import BatchFeature

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 2

    class FakeProcessor:
        tokenizer = FakeTokenizer()

        @staticmethod
        def batch_decode(*_args, **_kwargs):
            return ["A"]

    class MetadataRejectingModel:
        device = "cpu"

        def __init__(self):
            self.generate_keys = set()

        def generate(self, **kwargs):
            self.generate_keys = set(kwargs)
            if "video_metadata" in kwargs:
                raise ValueError("video_metadata is not a model input")
            return torch.tensor([[11, 12, 13]])

    inputs = BatchFeature(
        data={
            "input_ids": torch.tensor([[11, 12]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "pixel_values_videos": torch.ones((1, 1)),
            "video_metadata": [{"fps": 1.0, "frames_indices": [0, 1]}],
        }
    )
    args = type(
        "Args",
        (),
        {
            "max_new_tokens": 1,
            "temperature": 0,
            "top_p": None,
            "top_k": None,
            "num_beams": 1,
            "enable_thinking": "false",
        },
    )()
    model = MetadataRejectingModel()

    outputs = runner.generate_preprocessed_baseline_batch(model, FakeProcessor(), inputs, args)

    assert outputs == ["A"]
    assert "video_metadata" not in model.generate_keys
    assert "pixel_values_videos" in model.generate_keys


def test_c2l_base_video_message_uses_tail_frame_count():
    args = type(
        "Args",
        (),
        {
            "system_prompt": "",
            "video_min_pixels": 1024,
            "video_max_pixels": 262144,
            "video_total_pixels": 0,
            "video_fps": 2.0,
            "num_frames": 32,
            "max_frames": 32,
            "base_input_frame_number": 4,
        },
    )()
    unit = EvalUnit("ovo_backward", {"id": 1, "task": "EPM"}, "Question?\nA. yes", "/tmp/1.mp4", None)

    original_capped_nframes = runner.capped_nframes
    original_duration = runner.video_duration_seconds
    try:
        runner.capped_nframes = lambda _video_path, requested_frames, *_bounds: requested_frames
        runner.video_duration_seconds = lambda _video_path: 10.0
        messages = runner.build_video_text_messages(unit, args)
    finally:
        runner.capped_nframes = original_capped_nframes
        runner.video_duration_seconds = original_duration

    content = messages[0]["content"]
    assert content[0]["type"] == "video"
    assert content[0]["nframes"] == 4
    assert content[0]["video_start"] == 8.0
    assert content[0]["video_end"] == 10.0
    assert content[1] == {"type": "text", "text": "Question?\nA. yes"}


def test_last_2second_1fps_video_content_uses_tail_window():
    args = type(
        "Args",
        (),
        {
            "video_min_pixels": 1024,
            "video_max_pixels": 262144,
            "video_total_pixels": 0,
            "video_fps": None,
            "num_frames": 64,
            "max_frames": 64,
            "last_2second_1fps": True,
        },
    )()

    original_duration = runner.video_duration_seconds
    try:
        runner.video_duration_seconds = lambda _video_path: 10.0
        content = runner.video_content("/tmp/1.mp4", args)
    finally:
        runner.video_duration_seconds = original_duration

    assert content["video_start"] == 8.0
    assert content["video_end"] == 10.0
    assert content["fps"] == 1.0
    assert content["max_frames"] == 2
    assert "nframes" not in content


def test_last_2second_1fps_video_content_keeps_short_window():
    args = type(
        "Args",
        (),
        {
            "video_min_pixels": 1024,
            "video_max_pixels": 262144,
            "video_total_pixels": 0,
            "video_fps": None,
            "num_frames": 64,
            "max_frames": 64,
            "last_2second_1fps": True,
        },
    )()

    original_duration = runner.video_duration_seconds
    try:
        runner.video_duration_seconds = lambda _video_path: 1.5
        content = runner.video_content("/tmp/1.mp4", args)
    finally:
        runner.video_duration_seconds = original_duration

    assert content["video_start"] == 0.0
    assert content["video_end"] == 1.5
    assert content["fps"] == 1.0
    assert content["max_frames"] == 2


def test_run_batches_uses_prefetch_pipeline():
    calls = []

    def fake_preprocess(_processor, units, _args):
        calls.append(("preprocess", [unit.prompt for unit in units]))
        return [unit.prompt for unit in units]

    def fake_generate(_model, _processor, inputs, _args):
        calls.append(("generate", list(inputs)))
        return [f"pred-{item}" for item in inputs]

    args = type("Args", (), {"batch_size": 1, "c2l_checkpoint": None, "no_video": False})()
    units = [
        EvalUnit("ovo_backward", {"id": 1, "task": "EPM", "gt": 0}, "p1", "/tmp/1.mp4", None),
        EvalUnit("ovo_backward", {"id": 2, "task": "EPM", "gt": 1}, "p2", "/tmp/2.mp4", None),
    ]

    original_preprocess = runner.preprocess_baseline_batch
    original_generate = runner.generate_preprocessed_baseline_batch
    try:
        runner.preprocess_baseline_batch = fake_preprocess
        runner.generate_preprocessed_baseline_batch = fake_generate
        rows = runner.run_batches(model=None, processor=None, units=units, args=args, rank=0)
    finally:
        runner.preprocess_baseline_batch = original_preprocess
        runner.generate_preprocessed_baseline_batch = original_generate

    assert [row["prediction"] for row in rows] == ["pred-p1", "pred-p2"]
    assert calls[0] == ("preprocess", ["p1"])
    assert sorted(calls) == sorted(
        [
            ("preprocess", ["p1"]),
            ("preprocess", ["p2"]),
            ("generate", ["p1"]),
            ("generate", ["p2"]),
        ]
    )


def test_run_batches_c2l_prefetch_passes_model_to_preprocessor():
    calls = []
    model = object()
    processor = object()

    def fake_prepare(received_model, received_processor, units, _args):
        assert received_model is model
        assert received_processor is processor
        calls.append(("preprocess", [unit.prompt for unit in units]))
        return [unit.prompt for unit in units]

    def fake_generate(received_model, received_processor, inputs, _args):
        assert received_model is model
        assert received_processor is processor
        calls.append(("generate", list(inputs)))
        return [f"pred-{item}" for item in inputs]

    args = type("Args", (), {"batch_size": 1, "c2l_checkpoint": "/tmp/checkpoint", "no_video": False})()
    units = [EvalUnit("ovo_backward", {"id": 1, "task": "EPM", "gt": 0}, "p1", "/tmp/1.mp4", None)]

    original_prepare = runner.prepare_c2l_inputs
    original_generate = runner.generate_preprocessed_c2l_batch
    try:
        runner.prepare_c2l_inputs = fake_prepare
        runner.generate_preprocessed_c2l_batch = fake_generate
        rows = runner.run_batches(model=model, processor=processor, units=units, args=args, rank=0)
    finally:
        runner.prepare_c2l_inputs = original_prepare
        runner.generate_preprocessed_c2l_batch = original_generate

    assert [row["prediction"] for row in rows] == ["pred-p1"]
    assert calls == [("preprocess", ["p1"]), ("generate", ["p1"])]
