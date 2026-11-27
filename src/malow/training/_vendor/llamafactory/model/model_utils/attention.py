# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import TYPE_CHECKING, Any

from ...extras import logging
from ...extras.constants import AttentionFunction
from ...extras.packages import is_torch_version_greater_than


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def _patch_flash_attention_optional_s_aux() -> None:
    r"""Allow HF's flash attention wrapper to receive ``s_aux=None``.

    Transformers 5.6.0 routes Qwen3-VL/Qwen3.5 vision attention through the generic
    flash attention interface, but these models do not pass the optional
    learnable-attention-sink tensor. The upstream wrapper unconditionally calls
    ``s_aux.to(...)``, so patch the registered interface with the same behavior
    except for guarding that optional value.
    """
    try:
        from transformers.integrations import flash_attention as flash_attention_module
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return

    original_forward = flash_attention_module.flash_attention_forward
    if getattr(original_forward, "_llamafactory_accepts_none_s_aux", False):
        return

    def flash_attention_forward(
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        sliding_window: int | None = None,
        softcap: float | None = None,
        is_causal: bool | None = None,
        s_aux: Any | None = None,
        **kwargs,
    ) -> tuple[Any, None]:
        if kwargs.get("output_attentions", False):
            flash_attention_module.logger.warning_once(
                "Flash Attention does not support `output_attentions=True`."
                " Please set your attention to `eager` if you want any of these features."
            )

        seq_len = query.shape[2]
        if any(dim == 0 for dim in query.shape):
            raise ValueError(
                "Tensor query has shape with a zero dimension.\n"
                "FlashAttention does not support inputs with dim=0.\n"
                "Please check your input shapes or use SDPA instead."
            )

        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        target_dtype = flash_attention_module.get_target_dtype(query, module)
        is_causal = is_causal if is_causal is not None else module.is_causal
        attn_output = flash_attention_module._flash_attention_forward(
            query,
            key,
            value,
            attention_mask,
            query_length=seq_len,
            is_causal=is_causal,
            dropout=dropout,
            softmax_scale=scaling,
            sliding_window=sliding_window,
            softcap=softcap,
            use_top_left_mask=flash_attention_module._use_top_left_mask,
            target_dtype=target_dtype,
            attn_implementation=module.config._attn_implementation,
            layer_idx=module.layer_idx if hasattr(module, "layer_idx") else None,
            s_aux=s_aux.to(query.dtype) if s_aux is not None else None,
            **kwargs,
        )

        return attn_output, None

    setattr(flash_attention_forward, "_llamafactory_accepts_none_s_aux", True)
    flash_attention_module.flash_attention_forward = flash_attention_forward
    for attn_implementation in ("flash_attention_2", "flash_attention_3", "flash_attention_4"):
        ALL_ATTENTION_FUNCTIONS[attn_implementation] = flash_attention_forward


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    from transformers.utils import is_flash_attn_2_available

    if getattr(config, "model_type", None) == "gpt_oss":
        from transformers.integrations.hub_kernels import load_and_register_kernel

        flash_attn3_kernel = "kernels-community/vllm-flash-attn3"
        load_and_register_kernel(flash_attn3_kernel)
        setattr(config, "_attn_implementation", flash_attn3_kernel)
        setattr(config, "_attn_implementation_internal", flash_attn3_kernel)
        model_args.flash_attn = AttentionFunction.FA3

        logger.info_rank0("Using FlashAttention-3 with attention sink for the gpt-oss model.")
        return

    if getattr(config, "model_type", None) == "gemma2":
        if model_args.flash_attn == AttentionFunction.AUTO or model_args.flash_attn == AttentionFunction.FA2:
            if is_flash_attn_2_available():
                if model_args.flash_attn != AttentionFunction.FA2:
                    logger.warning_rank0("Gemma 2 should use flash attention 2, change `flash_attn` to fa2.")
                    model_args.flash_attn = AttentionFunction.FA2
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, use eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
        elif model_args.flash_attn == AttentionFunction.SDPA:
            logger.warning_rank0(
                "Gemma-2 should use soft-capping attention, while the SDPA attention does not support it."
            )

    if getattr(config, "model_type", None) in ["youtu", "youtu_vl"]:
        if model_args.flash_attn in (AttentionFunction.AUTO, AttentionFunction.SDPA):
            logger.warning_rank0("Youtu-VL does not support SDPA, forcing eager attention.")
            model_args.flash_attn = AttentionFunction.DISABLED

    if model_args.flash_attn == AttentionFunction.AUTO:
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        if not is_torch_version_greater_than("2.1.1"):
            logger.warning_rank0("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        from transformers import is_torch_npu_available

        if not (is_flash_attn_2_available() or is_torch_npu_available()):
            if is_torch_version_greater_than("2.1.1"):
                logger.warning_rank0("FlashAttention-2 is not installed, falling back to torch SDPA attention.")
                model_args.flash_attn = AttentionFunction.SDPA
                requested_attn_implementation = "sdpa"
            else:
                logger.warning_rank0("FlashAttention-2 is not installed, falling back to eager attention.")
                model_args.flash_attn = AttentionFunction.DISABLED
                requested_attn_implementation = "eager"
        else:
            requested_attn_implementation = "flash_attention_2"
            if getattr(config, "model_type", None) in ["qwen3_vl", "qwen3_vl_moe", "qwen3_5", "qwen3_5_moe"]:
                _patch_flash_attention_optional_s_aux()
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        setattr(config, "attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "kimi_vl":
        setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    elif getattr(config, "model_type", None) == "youtu_vl":
        setattr(config, "attn_implementation", requested_attn_implementation)
        setattr(config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "vision_config"):
            setattr(config.vision_config, "_attn_implementation", requested_attn_implementation)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "_attn_implementation", requested_attn_implementation)
    else:
        setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    if getattr(config, "model_type", None) == "internlm2":  # special case for custom models
        attn_implementation = getattr(config, "attn_implementation", None)
    else:
        attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info_rank0("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info_rank0("Using torch SDPA for faster training and inference.")
    else:
        logger.info_rank0("Using vanilla attention implementation.")
