"""Apply streaming memory adapters to the VLA's current observation."""
from __future__ import annotations
from typing import Any
import torch
from malow.vla.modeling import VLAModelWrapper


class C2LVLAModelWrapper(VLAModelWrapper):
    def __init__(self, modulated_model: Any, processor: Any, vla_config: Any) -> None:
        super().__init__(qwen_model=modulated_model, processor=processor, vla_config=vla_config)

    @property
    def modulated_model(self) -> Any:
        return self.qwen_model

    def _forward_qwen(self, model_inputs: dict[str, Any],
                      generated_loras: dict[str, dict[str, torch.Tensor]] | None = None,
                      unmodulated: bool = False,
                      n_ctx_chunks: torch.Tensor | None = None,
                      n_queries: torch.Tensor | None = None, **_: Any) -> Any:
        if (generated_loras is not None) + bool(unmodulated) != 1:
            raise ValueError("Specify generated_loras or unmodulated, exclusively")
        batch_size = int(model_inputs["input_ids"].shape[0]) if "input_ids" in model_inputs else int(model_inputs["inputs_embeds"].shape[0])
        if unmodulated:
            self.qwen_model.reset()
            try:
                return self.qwen_model.base_model(**model_inputs, output_attentions=False,
                    output_hidden_states=True, return_dict=True)
            finally:
                self.qwen_model.patch_lora_forward()
        self.qwen_model.apply_generated_loras(generated_loras, n_ctx_chunks=n_ctx_chunks,
            n_queries=n_queries, position_ids=model_inputs.get("position_ids"), batch_size=batch_size)
        return self.qwen_model.base_model(**model_inputs, output_attentions=False,
            output_hidden_states=True, return_dict=True)
