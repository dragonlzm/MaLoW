import logging
from dataclasses import dataclass
from enum import Enum

import torch
from einops import rearrange, repeat
try:
    from jaxtyping import Float, Integer
except ImportError:
    class _TensorAnnotation:
        def __getitem__(self, item):
            return item[0]

    Float = Integer = _TensorAnnotation()
from torch import Tensor, nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from malow.vla.ctx_to_lora.configs import (
    AggregatorArguments,
)
from malow.vla.ctx_to_lora.modeling.idefics2 import Idefics2Perceiver, Idefics2PerceiverConfig
from malow.vla.ctx_to_lora.pooling import POOL_FN
from malow.vla.ctx_to_lora.utils import (
    get_num_layers,
)

logger = logging.getLogger()


class AGGREGATOR_TYPE(str, Enum):
    POOLER = "pooler"
    PERCEIVER = "perceiver"
    ENCODER_AGGREGATOR = "encoder_aggregator"


@dataclass
class AggregatorConfig:
    aggregator_type: AGGREGATOR_TYPE
    num_layers: int
    num_modules: int
    num_extra_modules: int
    output_size: int
    feature_size: int

    # pooler
    pooling_type: POOL_FN

    # perceiver
    num_latent_factor: int
    lora_r: int
    per_rank_gen: bool

    n_latent_queries: int
    num_blocks: int
    num_self_attn_per_block: int
    shared_weights: bool
    encoder_aggregator_num_layers: int | None
    encoder_aggregator_d_model: int | None
    layer_to_layer_ctx_encoder: bool
    layer_specific_queries: bool = False
    query_per_layer: bool = False
    query_per_module: bool = True
    query_per_rank: bool = True
    attn_implementation: str = "eager"
    valid_query_mask: list[list[bool]] | None = None


def ensure_aggregator_config_compat(config: AggregatorConfig) -> AggregatorConfig:
    """Fill fields that may be absent from older pickled checkpoint configs."""
    defaults = {
        "encoder_aggregator_num_layers": None,
        "encoder_aggregator_d_model": None,
        "layer_specific_queries": False,
        "query_per_layer": False,
        "query_per_module": True,
        "query_per_rank": True,
        "attn_implementation": "eager",
        "valid_query_mask": None,
    }
    for name, value in defaults.items():
        if getattr(config, name, None) is None:
            setattr(config, name, value)

    if isinstance(config.aggregator_type, str):
        config.aggregator_type = AGGREGATOR_TYPE(config.aggregator_type)

    return config


def get_aggregator_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    layer_to_layer_ctx_encoder: bool,
    output_size: int,
    num_modules: int,
    num_extra_modules: int,
    lora_r: int,
    per_rank_gen: bool,
    aggregator_args: AggregatorArguments,
    valid_query_mask: list[list[bool]] | None = None,
):
    return AggregatorConfig(
        feature_size=ctx_encoder_model_config.hidden_size,
        output_size=output_size,
        num_layers=get_num_layers(model),
        num_modules=num_modules,
        num_extra_modules=num_extra_modules,
        lora_r=lora_r,
        per_rank_gen=per_rank_gen,
        layer_to_layer_ctx_encoder=layer_to_layer_ctx_encoder,
        valid_query_mask=valid_query_mask,
        **vars(aggregator_args),
    )


class _LoRAOutputMixin:
    def _init_output_shape(
        self,
        num_layers: int,
        num_modules: int,
        num_extra_modules: int,
        per_rank_gen: bool,
        lora_r: int,
        layer_to_layer_ctx_encoder: bool,
        query_per_layer: bool = True,
        query_per_module: bool = True,
        query_per_rank: bool = True,
    ) -> int:
        assert num_extra_modules == 0
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.num_extra_modules = num_extra_modules
        self.per_rank_gen = per_rank_gen
        self.r = lora_r if self.per_rank_gen else 1
        # self.layer_to_layer=False means the aggregator sees one context feature sequence and 
        # produces the whole [n_layers, n_modules, r] LoRA latent grid in one output.
        # when using vision embedder / multi-modal embedder, it will duplicate the feature for each layer
        # so they will still be self.layer_to_layer=Ture
        self.layer_to_layer = layer_to_layer_ctx_encoder
        self.query_per_layer = bool(query_per_layer)
        self.query_per_module = bool(query_per_module)
        self.query_per_rank = bool(query_per_rank)
        query_layers = 1 if self.layer_to_layer else self._query_num_layers
        n_output_queries = query_layers * self._query_num_modules * self._query_num_r
        self.iterative_mode = False
        return n_output_queries

    @property
    def _query_num_layers(self) -> int:
        return self.num_layers if self.query_per_layer else 1

    @property
    def _query_num_modules(self) -> int:
        return self.num_modules if self.query_per_module else 1

    @property
    def _query_num_r(self) -> int:
        return self.r if self.query_per_rank else 1

    def enable_iterative_mode(self, x: bool):
        self.iterative_mode = x

    def _reshape_outputs(self, x: Float[Tensor, "bs n_queries feature_dim"]):
        q_modules = self._query_num_modules
        q_r = self._query_num_r

        if self.layer_to_layer and self.iterative_mode:
            lora_x = rearrange(
                x,
                "bs (q_modules q_r) d -> bs q_modules q_r d",
                q_modules=q_modules,
                q_r=q_r,
            )
            lora_x = self._expand_module_rank(lora_x)
            if not self.per_rank_gen:
                lora_x = lora_x.squeeze(2)
            return lora_x, None

        if self.layer_to_layer:
            lora_x = rearrange(
                x,
                "(num_layers bs) (q_modules q_r) d -> bs num_layers q_modules q_r d",
                num_layers=self.num_layers,
                q_modules=q_modules,
                q_r=q_r,
            )
            lora_x = self._expand_module_rank(lora_x)
        else:
            q_layers = self._query_num_layers
            lora_x = rearrange(
                x,
                "bs (q_layers q_modules q_r) d -> bs q_layers q_modules q_r d",
                q_layers=q_layers,
                q_modules=q_modules,
                q_r=q_r,
            )
            lora_x = self._expand_layer_module_rank(lora_x)

        if not self.per_rank_gen:
            lora_x = lora_x.squeeze(3)

        return lora_x, None

    def _expand_layer_module_rank(self, x: Float[Tensor, "bs _ _ _ d"]):
        if not self.query_per_layer:
            x = x.expand(-1, self.num_layers, -1, -1, -1)
        return self._expand_module_rank(x)

    def _expand_module_rank(self, x: Tensor):
        if not self.query_per_module:
            if x.dim() == 5:
                x = x.expand(-1, -1, self.num_modules, -1, -1)
            else:
                x = x.expand(-1, self.num_modules, -1, -1)
        if not self.query_per_rank:
            if x.dim() == 5:
                x = x.expand(-1, -1, -1, self.r, -1)
            else:
                x = x.expand(-1, -1, self.r, -1)
        return x


class Perceiver(_LoRAOutputMixin, nn.Module):
    """perceiver w/ bottleneck size = n_modules * n_layers"""

    def __init__(
        self,
        feature_size,
        output_size,
        num_layers,
        num_modules,
        num_extra_modules,
        per_rank_gen,
        lora_r,
        num_latent_factor,  # unused
        layer_to_layer_ctx_encoder,
        n_latent_queries,
        *args,
        **kwargs,
    ):
        super().__init__()
        n_output_queries = self._init_output_shape(
            num_layers,
            num_modules,
            num_extra_modules,
            per_rank_gen,
            lora_r,
            layer_to_layer_ctx_encoder,
        )
        attn_implementation = kwargs.get("attn_implementation", "eager")
        self.config = Idefics2PerceiverConfig(
            input_size=feature_size,
            num_blocks=kwargs["num_blocks"],
            num_self_attn_per_block=kwargs["num_self_attn_per_block"],
            shared_weights=kwargs["shared_weights"],
            n_latents=n_latent_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation=attn_implementation,
        )
        self.decoder_config = Idefics2PerceiverConfig(
            input_size=output_size,
            num_blocks=1,
            num_self_attn_per_block=0,
            shared_weights=False,
            n_latents=n_output_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation=attn_implementation,
        )
        self.perceiver = Idefics2Perceiver(self.config, self.decoder_config)

    def forward(
        self,
        ctx_features: Float[Tensor, "bs seq_len feature_dim"]
        | Float[Tensor, "bs x seq_len feature_dim"],
        ctx_attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        if self.layer_to_layer and not self.iterative_mode:
            if ctx_attn_mask is not None:
                ctx_attn_mask = repeat(
                    ctx_attn_mask,
                    "bs seq_len -> (num_layers bs) seq_len",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "bs num_layers seq_len feature_dim -> (num_layers bs) seq_len feature_dim",
                )
            if ctx_position_ids is not None:
                ctx_position_ids = repeat(
                    ctx_position_ids,
                    "1 seq_len -> 1 (num_layers seq_len)",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "1 num_layers seq_len feature_dim -> 1 (num_layers seq_len) feature_dim",
                )

        x = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)
        return self._reshape_outputs(x)


class EncoderAggregator(_LoRAOutputMixin, nn.Module):
    """Transformer encoder aggregator with learned output queries."""

    def __init__(
        self,
        feature_size,
        output_size,
        num_layers,
        num_modules,
        num_extra_modules,
        per_rank_gen,
        lora_r,
        num_latent_factor,  # unused
        layer_to_layer_ctx_encoder,
        n_latent_queries,  # unused
        *args,
        **kwargs,
    ):
        super().__init__()
        query_per_layer = bool(
            kwargs.get("query_per_layer", False)
            or kwargs.get("layer_specific_queries", False)
        )
        n_output_queries = self._init_output_shape(
            num_layers,
            num_modules,
            num_extra_modules,
            per_rank_gen,
            lora_r,
            layer_to_layer_ctx_encoder,
            query_per_layer,
            kwargs.get("query_per_module", True),
            kwargs.get("query_per_rank", True),
        )

        encoder_d_model = kwargs.get("encoder_aggregator_d_model") or output_size
        encoder_num_layers = (
            kwargs.get("encoder_aggregator_num_layers") or kwargs["num_blocks"]
        )
        self.layer_specific_queries = bool(kwargs.get("layer_specific_queries", False))

        self.ctx_proj = nn.Linear(feature_size, encoder_d_model)
        valid_query_mask = kwargs.get("valid_query_mask")
        self.sparse_valid_queries = bool(
            valid_query_mask is not None
            and self.layer_to_layer
            and self.query_per_layer
            and self.query_per_module
        )
        if self.sparse_valid_queries:
            valid_query_mask = torch.as_tensor(valid_query_mask, dtype=torch.bool)
            expected_shape = (self.num_layers, self.num_modules)
            if tuple(valid_query_mask.shape) != expected_shape:
                raise ValueError(
                    "C2L valid query mask must have shape "
                    f"{expected_shape}, got {tuple(valid_query_mask.shape)}."
                )
            valid_query_token_mask = (
                valid_query_mask.unsqueeze(-1)
                .expand(-1, -1, self._query_num_r)
                .reshape(self.num_layers, n_output_queries)
            )
            self.register_buffer(
                "valid_query_token_mask",
                valid_query_token_mask,
                persistent=False,
            )
            query_shape = (int(valid_query_token_mask.sum().item()), encoder_d_model)
        elif self.query_per_layer and self.layer_to_layer:
            query_shape = (num_layers, n_output_queries, encoder_d_model)
        else:
            query_shape = (n_output_queries, encoder_d_model)
        self.query = nn.Parameter(torch.randn(*query_shape) * encoder_d_model**-0.5)
        self.output_proj = (
            nn.Identity()
            if encoder_d_model == output_size
            else nn.Linear(encoder_d_model, output_size)
        )
        nhead = self._get_num_heads(encoder_d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=encoder_d_model,
            nhead=nhead,
            dim_feedforward=encoder_d_model * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_num_layers,
            norm=nn.LayerNorm(encoder_d_model),
            enable_nested_tensor=False,
        )
        self._warned_position_ids = False

    @staticmethod
    def _get_num_heads(hidden_size: int) -> int:
        for nhead in (8, 16, 4, 2, 1):
            if hidden_size % nhead == 0:
                return nhead
        return 1

    def _get_key_padding_mask(
        self,
        ctx_attn_mask: Integer[Tensor, "bs seq_len"] | None,
        n_queries: int,
    ) -> Tensor | None:
        if ctx_attn_mask is None:
            return None
        ctx_padding_mask = ctx_attn_mask == 0
        query_padding_mask = torch.zeros(
            (ctx_padding_mask.shape[0], n_queries),
            dtype=torch.bool,
            device=ctx_padding_mask.device,
        )
        return torch.cat([query_padding_mask, ctx_padding_mask], dim=1)

    def forward(
        self,
        ctx_features: Float[Tensor, "bs seq_len feature_dim"]
        | Float[Tensor, "bs x seq_len feature_dim"],
        ctx_attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        if self.query_per_layer and self.layer_to_layer and self.iterative_mode:
            raise ValueError("EncoderAggregator layer-specific queries do not support iterative mode.")

        if self.layer_to_layer and not self.iterative_mode:
            if ctx_features.dim() == 4:
                ctx_features = rearrange(
                    ctx_features,
                    "bs num_layers seq_len feature_dim -> (num_layers bs) seq_len feature_dim",
                )
            if ctx_attn_mask is not None:
                ctx_attn_mask = repeat(
                    ctx_attn_mask,
                    "bs seq_len -> (num_layers bs) seq_len",
                    num_layers=self.num_layers,
                )

        if ctx_position_ids is not None and not self._warned_position_ids:
            logger.warning(
                "EncoderAggregator received ctx_position_ids; "
                "packed position-id segmentation is ignored."
            )
            self._warned_position_ids = True

        ctx_features = self.ctx_proj(ctx_features)
        query_padding_mask = None
        if self.sparse_valid_queries:
            if ctx_features.shape[0] % self.num_layers != 0:
                raise ValueError(
                    "Layer-specific queries expect ctx_features batch to be a multiple "
                    f"of num_layers={self.num_layers}, got {ctx_features.shape[0]}."
                )
            bs = ctx_features.shape[0] // self.num_layers
            dense_queries = self.query.new_zeros(
                self.num_layers,
                self.valid_query_token_mask.shape[1],
                self.query.shape[-1],
            )
            dense_queries[self.valid_query_token_mask] = self.query
            queries = repeat(
                dense_queries,
                "num_layers n d -> (num_layers bs) n d",
                bs=bs,
            )
            query_padding_mask = repeat(
                ~self.valid_query_token_mask,
                "num_layers n -> (num_layers bs) n",
                bs=bs,
            )
            n_queries = dense_queries.shape[1]
        elif self.query_per_layer and self.layer_to_layer:
            if ctx_features.shape[0] % self.num_layers != 0:
                raise ValueError(
                    "Layer-specific queries expect ctx_features batch to be a multiple "
                    f"of num_layers={self.num_layers}, got {ctx_features.shape[0]}."
                )
            bs = ctx_features.shape[0] // self.num_layers
            queries = repeat(self.query, "num_layers n d -> (num_layers bs) n d", bs=bs)
            n_queries = self.query.shape[1]
        else:
            queries = repeat(self.query, "n d -> bs n d", bs=ctx_features.shape[0])
            n_queries = self.query.shape[0]
        tgt = torch.cat([queries, ctx_features], dim=1)
        key_padding_mask = self._get_key_padding_mask(ctx_attn_mask, n_queries)
        if query_padding_mask is not None:
            if key_padding_mask is None:
                context_padding_mask = torch.zeros(
                    (ctx_features.shape[0], ctx_features.shape[1]),
                    dtype=torch.bool,
                    device=ctx_features.device,
                )
            else:
                context_padding_mask = key_padding_mask[:, n_queries:]
            key_padding_mask = torch.cat(
                [query_padding_mask, context_padding_mask], dim=1
            )
        x = self.context_encoder(tgt, src_key_padding_mask=key_padding_mask)
        x = x[:, :n_queries]
        x = self.output_proj(x)
        if query_padding_mask is not None:
            x = x.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        return self._reshape_outputs(x)


AGGREGATOR_CLS = {
    AGGREGATOR_TYPE.PERCEIVER: Perceiver,
    AGGREGATOR_TYPE.ENCODER_AGGREGATOR: EncoderAggregator,
}
