# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
from dataclasses import dataclass, field
from typing import Literal



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



