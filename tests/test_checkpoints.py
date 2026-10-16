"""Metadata tests use tiny tensors and stand-in config modules, without models."""
import json
import tempfile
import types
import unittest
from enum import Enum
from pathlib import Path
from unittest.mock import patch

import torch
from malow.checkpoints import load_checkpoint_metadata, save_checkpoint_metadata


class HypernetConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class AGGREGATOR_TYPE(str, Enum):
    ENCODER_AGGREGATOR = "encoder_aggregator"


class CheckpointMetadataTests(unittest.TestCase):
    def test_nested_types_and_tensors_round_trip_both_runtimes(self):
        source = {"hypernet_config": HypernetConfig(layer_indices=torch.tensor([0, 3]), feature_sizes=({("layer", 0): 4}, {}), targets={"q", "v"}, aggregator_type=AGGREGATOR_TYPE.ENCODER_AGGREGATOR), "enabled": True}
        modules = {}
        for runtime in ("video", "vla"):
            modules[f"malow.{runtime}.ctx_to_lora.modeling.hypernet"] = types.SimpleNamespace(HypernetConfig=HypernetConfig)
            modules[f"malow.{runtime}.ctx_to_lora.modeling.aggregator"] = types.SimpleNamespace(AGGREGATOR_TYPE=AGGREGATOR_TYPE)
        with tempfile.TemporaryDirectory() as folder, patch.dict("sys.modules", modules):
            save_checkpoint_metadata(source, folder)
            for runtime in ("video", "vla"):
                loaded = load_checkpoint_metadata(folder, runtime)
                config = loaded["hypernet_config"]
                self.assertIsInstance(config, HypernetConfig)
                self.assertTrue(torch.equal(config.layer_indices, source["hypernet_config"].layer_indices))
                self.assertEqual(config.feature_sizes, ({("layer", 0): 4}, {}))
                self.assertEqual(config.targets, {"q", "v"})
                self.assertIs(config.aggregator_type, AGGREGATOR_TYPE.ENCODER_AGGREGATOR)

    def test_rejects_unknown_type_version_and_tensor_path(self):
        with tempfile.TemporaryDirectory() as folder:
            path = save_checkpoint_metadata({"enabled": True}, folder)
            original = json.loads(path.read_text())
            for mutation in ({"version": 999}, {"tensor_file": "../private.safetensors"}, {"metadata": {"bad": {"$type": "config", "class": "os.system", "attributes": {}}}}):
                path.write_text(json.dumps(dict(original, **mutation)))
                with self.assertRaises(ValueError):
                    load_checkpoint_metadata(path)

    def test_peft_config_round_trip(self):
        from peft import LoraConfig
        from peft.utils.peft_types import TaskType
        with tempfile.TemporaryDirectory() as folder:
            original = LoraConfig(r=8, lora_alpha=16, target_modules={"q_proj", "v_proj"}, task_type=TaskType.CAUSAL_LM)
            save_checkpoint_metadata({"lora": original}, folder)
            restored = load_checkpoint_metadata(folder)["lora"]
            self.assertIsInstance(restored, LoraConfig)
            self.assertEqual(original.to_dict(), restored.to_dict())


if __name__ == "__main__":
    unittest.main()
