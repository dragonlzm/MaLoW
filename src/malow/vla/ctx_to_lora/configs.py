from dataclasses import dataclass, field
from typing import Literal

@dataclass
class HypernetArguments:
    latent_size: int = field(
        default=512,
        metadata={"help": "Latent size for HyperLoRA."},
    )
    use_light_weight_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use light-weight LoRA."},
    )
    light_weight_latent_size: int = field(
        default=128,
        metadata={"help": "Latent size for light-weight LoRA."},
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for HyperLoRA."},
    )
    extra_modules: list[str] | None = field(
        default=None,
        metadata={"help": "Extra modules to train."},
    )
    per_rank_gen: bool = field(
        default=False,
        metadata={"help": "Whether to use per-rank generation."},
    )
    use_bias: bool = field(
        default=True, metadata={"help": "Whether to include data-dependent LoRA"}
    )
    use_per_rank_bias: bool = field(
        default=False, metadata={"help": "Whether to use per-rank bias."}
    )
    per_layer_processing: bool = field(
        default=False,
        metadata={"help": "Whether to use per-layer processing (after preceiver)."},
    )
    use_token_mixing: bool = field(
        default=False,
        metadata={"help": "Whether to use token mixing block."},
    )
    num_pre_head_layers: int = field(
        default=1, metadata={"help": "# of layers before hypernet head"}
    )
    head_per_layer: bool = field(
        default=True,
        metadata={"help": "Whether the HyperLoRA output head has per-layer weights."},
    )
    head_per_module: bool = field(
        default=True,
        metadata={"help": "Whether the HyperLoRA output head has per-module weights."},
    )
    head_per_rank: bool = field(
        default=False,
        metadata={"help": "Whether the HyperLoRA output head has per-rank weights."},
    )
    use_valid_slots: bool = field(
        default=False,
        metadata={"help": "Whether HyperLoRA should emit only valid layer-target slots."},
    )
    temporal_deltanet_num_layers: int = field(
        default=2,
        metadata={"help": "Number of causal Gated DeltaNet blocks in the temporal aggregator."},
    )
    action_history_dim: int | None = field(
        default=None,
        metadata={
            "help": (
                "Width of one previous-step action vector, which turns on the hypernet's "
                "action-history MLP. Set only by `stage: c2l_vla` with "
                "`interleave_action_history`; None everywhere else, so the video C2L "
                "hypernet is built exactly as before."
            )
        },
    )

@dataclass
class CtxEncoderArguments:
    ctx_encoder_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Context encoder model name or path."},
    )
    ctx_encoder_type: Literal[
        "embed_only",
        "per_layer_activations",
        "early_exit",
        "vision_embedder",
        "vjepa2_vision_embedder",
        "multimodal_embedder",
    ] = (
        field(
            default="early_exit",
            metadata={
                "help": "Context encoder type. "
                "Options: 'embed_only', 'per_layer_activations', 'early_exit', "
                "'vision_embedder', 'vjepa2_vision_embedder', 'multimodal_embedder'."
            },
        )
    )
    # used only with `early_exit` type
    layer_idx: int | None = field(
        default=None,
        metadata={
            "help": "Layer index for context encoder. "
            "Default to L//4 where L is the number of layers of the ctx model. "
            "Only used when ctx_encoder_type==early_exit"
        },
    )
    quantize_ctx_encoder: bool = field(
        default=False, metadata={"help": "Wheter to quantize the ctx encoder."}
    )
    ctx_encoder_last_layer: int | None = field(
        default=None,
        metadata={
            "help": "Maximum number of layers for the context encoder. "
            "Only used when ctx_encoder_type==per_layer_activations"
        },
    )
    vjepa2_window_size: int = field(
        default=16,
        metadata={"help": "Number of 1-FPS frames in each trailing V-JEPA2 window."},
    )
    vjepa2_window_stride: int = field(
        default=2,
        metadata={"help": "Frame stride between trailing V-JEPA2 windows."},
    )
    vjepa2_spatial_pool_size: int = field(
        default=2,
        metadata={"help": "Spatial mean-pooling factor applied to the retained V-JEPA2 tubelet."},
    )
    vjepa2_micro_batch_size: int = field(
        default=32,
        metadata={"help": "Maximum number of V-JEPA2 windows forwarded together per rank."},
    )

@dataclass
class AggregatorArguments:
    aggregator_type: Literal["pooler", "perceiver", "encoder_aggregator"] = field(
        default="perceiver",
        metadata={
            "help": "Aggregator type for HyperLoRA. Use `perceiver` for the original "
            "Idefics2 perceiver resampler or `encoder_aggregator` for a TransformerEncoder "
            "with learned query tokens."
        },
    )

    # pooler
    pooling_type: str = field(
        default="mean",
        metadata={"help": "Pooling type for HyperLoRA."},
    )
    num_latent_factor: int = field(
        default=8,
        metadata={"help": "Number of latent factors for Perceiver."},
    )
    n_latent_queries: int = field(
        default=208,  # 26 * 8
        metadata={"help": "Number of latent queries of Perceiver."},
    )

    num_blocks: int = field(
        default=8,
        metadata={"help": "Number of blocks for Perceiver."},
    )
    num_self_attn_per_block: int = field(
        default=0,
        metadata={"help": "Number of self-attention layers per block for Perceiver."},
    )
    shared_weights: bool = field(
        default=False,
        metadata={"help": "Whether to share weights across blocks for Perceiver."},
    )
    encoder_aggregator_num_layers: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of TransformerEncoder layers for EncoderAggregator. "
                "Defaults to num_blocks when unset."
            )
        },
    )
    encoder_aggregator_d_model: int | None = field(
        default=None,
        metadata={
            "help": (
                "Internal TransformerEncoder d_model for EncoderAggregator. "
                "Defaults to HyperLoRA latent_size when unset."
            )
        },
    )
    layer_specific_queries: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether EncoderAggregator should learn a separate output-query table "
                "for each target LLM layer."
            )
        },
    )
    query_per_layer: bool = field(
        default=False,
        metadata={
            "help": "Whether EncoderAggregator output queries vary by target LLM layer."
        },
    )
    query_per_module: bool = field(
        default=True,
        metadata={"help": "Whether EncoderAggregator output queries vary by target module."},
    )
    query_per_rank: bool = field(
        default=True,
        metadata={"help": "Whether EncoderAggregator output queries vary by LoRA rank."},
    )
