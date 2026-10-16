"""Exercise Qwen3-VL's real backbone, vision outputs, and portable tower loader."""
from types import SimpleNamespace

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, Qwen3VLConfig, Qwen3VLTextConfig, Qwen3VLVisionConfig

from malow.video.ctx_to_lora.modeling.ctx_encoder import VisionEmbedder
from malow.video.ctx_to_lora.utils import get_layers, get_lora_target_slots
from malow.video.vision_encoder import load_c2l_vision_encoder


def tiny_config():
    text = Qwen3VLTextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
                            head_dim=8, rope_parameters={"rope_type": "default", "rope_theta": 10000,
                                                         "mrope_section": [1, 1, 2]})
    vision = Qwen3VLVisionConfig(depth=1, hidden_size=16, intermediate_size=32, num_heads=2,
                                patch_size=2, spatial_merge_size=2, temporal_patch_size=2,
                                out_hidden_size=16, num_position_embeddings=16,
                                deepstack_visual_indexes=[0])
    return Qwen3VLConfig(text_config=text.to_dict(), vision_config=vision.to_dict(),
                         image_token_id=29, video_token_id=30, vision_start_token_id=31)


def test_qwen3vl_backbone_and_vision_tower_roundtrip(tmp_path):
    torch.manual_seed(71)
    base = AutoModelForImageTextToText.from_config(tiny_config()).eval()
    assert base.__class__.__name__ == "Qwen3VLForConditionalGeneration"
    assert len(get_layers(base)) == 2
    lora = LoraConfig(r=2, target_modules=["down_proj", "up_proj", "gate_proj", "self_attn.v_proj"])
    slots = get_lora_target_slots(base, lora)
    assert len(slots) == 8
    assert {s["target_key"] for s in slots} == set(lora.target_modules)
    with torch.inference_mode():
        output = base(input_ids=torch.tensor([[1, 2, 3]]))
    assert output.logits.shape == (1, 3, 32)
    base.save_pretrained(tmp_path)
    loaded = load_c2l_vision_encoder(tmp_path, base.config, dtype=torch.float32)
    assert loaded.model.__class__.__name__ == "Qwen3VLVisionModel"
    assert loaded.feature_config.hidden_size == 16
    assert not loaded.model.training and not any(p.requires_grad for p in loaded.model.parameters())
    for name, value in base.model.visual.state_dict().items():
        torch.testing.assert_close(value, loaded.model.state_dict()[name], rtol=0, atol=0)
    # Eight frames form four temporal tubelets, each with one merged spatial token.
    grid = torch.tensor([[4, 2, 2]])
    pixels = torch.randn(16, 3 * 2 * 2 * 2)
    embedder = VisionEmbedder(loaded.model, SimpleNamespace())
    features, mask = embedder(pixel_values_videos=pixels, video_grid_thw=grid)
    with torch.inference_mode():
        expected = base.model.visual(pixels, grid_thw=grid).pooler_output
    assert features.shape == (1, 4, 16)
    assert mask.tolist() == [[1, 1, 1, 1]]
    torch.testing.assert_close(features[0], expected, rtol=0, atol=0)
