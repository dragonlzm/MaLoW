# Inference code adapted from Hippocampus (e2f6b9b). See NOTICE.
import logging
from collections.abc import Iterable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from math import isfinite, sqrt
from operator import attrgetter
from typing import Any

import torch
import torch.nn.functional as F
from einops import repeat, unpack
from einops.layers.torch import EinMix as Mix
try:
    from jaxtyping import Float, Integer
except ImportError:
    class _TensorAnnotation:
        def __getitem__(self, item):
            return item[0]

    Float = Integer = _TensorAnnotation()
from peft import (
    LoraConfig,
    LoraRuntimeConfig,
    PeftConfig,
    PeftModel,
)
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import PeftType, TaskType
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertModel

from malow.video.ctx_to_lora.configs import (
    CtxEncoderArguments,
)
from malow.video.ctx_to_lora.modeling.aggregator import (
    AGGREGATOR_CLS,
    AggregatorConfig,
    ensure_aggregator_config_compat,
)
from malow.video.ctx_to_lora.modeling.ctx_encoder import CTX_ENCODER_CLS, CTX_ENCODER_TYPE
from malow.video.ctx_to_lora.modeling.lora_layer import (
    apply_lora_to_valid_slots,
    apply_lora_to_layers,
    lora_forward,
    lora_forward_packed,
)
from malow.video.ctx_to_lora.modeling.lora_merger import combine_lora
from malow.video.ctx_to_lora.modeling.interactive import InteractiveC2LMixin
from malow.video.ctx_to_lora.modeling.temporal_aggregator import TemporalGatedDeltaNet, TemporalHistoryGate
from malow.video.ctx_to_lora.modeling.temporal_weighting import (
    TEMPORAL_QUERY_WEIGHTING_MODES,
    RawPooledCosineGate,
    RawPooledCosineWeighting,
    raw_pooled_cosine_distances,
)
from malow.video.ctx_to_lora.utils import (
    get_layers,
    get_lora_target_modules,
    get_lora_target_slots,
    get_num_layers,
    get_peft_in_out_features,
    safe_lora_target_key,
)

logger = logging.getLogger()


def _ctx_encoder_type_is(ctx_encoder_type: CTX_ENCODER_TYPE | str, expected: CTX_ENCODER_TYPE) -> bool:
    return ctx_encoder_type == expected or ctx_encoder_type == expected.value


def _get_ctx_encoder_cls(ctx_encoder_type: CTX_ENCODER_TYPE | str):
    return CTX_ENCODER_CLS[CTX_ENCODER_TYPE(ctx_encoder_type)]


def _target_modules_and_valid_query_mask(
    target_slots: Iterable[Mapping[str, int | str]],
    layer_indices: Iterable[int],
) -> tuple[tuple[str, ...], list[list[bool]]]:
    """Return stable target-module order and its valid decoder-layer slots."""
    layer_indices = tuple(int(layer_idx) for layer_idx in layer_indices)
    layer_positions = {
        layer_idx: layer_pos for layer_pos, layer_idx in enumerate(layer_indices)
    }
    slots = list(target_slots)

    target_modules = []
    for slot in slots:
        layer_idx = int(slot["layer_idx"])
        if layer_idx not in layer_positions:
            raise ValueError(
                f"C2L target slot references decoder layer {layer_idx}, "
                f"which is absent from layer_indices={layer_indices}."
            )
        target_key = str(slot["target_key"])
        if target_key not in target_modules:
            target_modules.append(target_key)

    target_positions = {
        target_key: target_pos for target_pos, target_key in enumerate(target_modules)
    }
    valid_query_mask = [
        [False for _ in target_modules] for _ in layer_indices
    ]
    for slot in slots:
        layer_pos = layer_positions[int(slot["layer_idx"])]
        target_pos = target_positions[str(slot["target_key"])]
        valid_query_mask[layer_pos][target_pos] = True

    return tuple(target_modules), valid_query_mask


@dataclass
class HypernetConfig:
    latent_size: int
    use_light_weight_lora: bool
    light_weight_latent_size: int
    per_rank_gen: bool
    use_per_rank_bias: bool
    use_bias: bool
    per_layer_processing: bool
    use_token_mixing: bool
    num_pre_head_layers: int
    dropout_rate: float
    head_per_layer: bool
    head_per_module: bool
    head_per_rank: bool

    lora_config: LoraConfig
    extra_modules: list[str] | None
    base_hidden_size: int

    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig
    use_valid_slots: bool = False
    target_slots: list[dict[str, int | str]] | None = None
    temporal_aggregator: str | None = None
    temporal_deltanet_num_layers: int = 2
    temporal_query_weighting: str | None = None
    temporal_query_gate_threshold: float = 0.05


def ensure_hypernet_config_compat(config: HypernetConfig) -> HypernetConfig:
    """Fill fields that may be absent from older pickled checkpoint configs."""
    defaults = {
        "num_pre_head_layers": 4,
        "use_per_rank_bias": False,
        "use_bias": True,
        "dropout_rate": 0.0,
        "head_per_layer": True,
        "head_per_module": True,
        "head_per_rank": False,
        "use_valid_slots": False,
        "target_slots": None,
        "temporal_aggregator": None,
        "temporal_deltanet_num_layers": 2,
        "temporal_query_weighting": None,
        "temporal_query_gate_threshold": 0.05,
    }
    for name, value in defaults.items():
        if getattr(config, name, None) is None:
            setattr(config, name, value)

    ensure_aggregator_config_compat(config.aggregator_config)
    return config


def _build_temporal_query_weighting(config: HypernetConfig) -> nn.Module | None:
    mode = getattr(config, "temporal_query_weighting", None)
    if mode == "raw_pooled_cosine":
        return RawPooledCosineWeighting()
    if mode == "raw_pooled_cosine_gate":
        return RawPooledCosineGate(
            threshold=getattr(config, "temporal_query_gate_threshold", 0.05)
        )
    if mode is not None:
        raise ValueError(f"Unsupported temporal query weighting mode: {mode!r}.")
    return None


class ResMLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        layers = []
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class ResMLPBlockPerLayer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        layers = [
            nn.LayerNorm(input_size),
            Mix(
                "bs n_layers n_modules r d_in -> bs n_layers n_modules r d_hid",
                weight_shape="n_layers d_in d_hid",
                bias_shape="n_layers d_hid",
                n_layers=n_layers,
                d_in=input_size,
                d_hid=hidden_size,
            ),
            nn.SiLU(),
            Mix(
                "bs n_layers n_modules r d_hid -> bs n_layers n_modules r d_out",
                weight_shape="n_layers d_hid d_out",
                bias_shape="n_layers d_out",
                n_layers=n_layers,
                d_hid=hidden_size,
                d_out=output_size,
            ),
            nn.LayerNorm(output_size),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class HyperLoRA(nn.Module):
    def __init__(self, config: HypernetConfig):
        super().__init__()

        # aggregator output [bs, n_layers, n_modules, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers
        self.config = ensure_hypernet_config_compat(config)
        logger.debug(f"HyperLoRA config: {self.config}")
        self.iterative_mode = False
        self._init_model()

    def _init_model(self):
        self.agg_config = self.config.aggregator_config
        self.aggregator = AGGREGATOR_CLS[self.agg_config.aggregator_type](
            **vars(self.agg_config)
        )

        self.lora_config = self.config.lora_config
        self.r = self.lora_config.r

        self.target_modules = (
            tuple(sorted(self.lora_config.target_modules)) if self.lora_config else None
        )
        self.num_modules = len(self.target_modules) if self.target_modules else 0
        self.extra_modules = (
            self.config.extra_modules if self.config.extra_modules else None
        )
        self.num_extra_modules = len(self.extra_modules) if self.extra_modules else 0
        self.layer_indices = self.config.layer_indices
        self.n_layers = len(self.layer_indices)

        self.d_in, self.d_out = self.config.feature_sizes
        self.d_latent = self.config.latent_size
        self.temporal_aggregator = (
            TemporalHistoryGate(self.n_layers)
            if self.config.temporal_aggregator == "history_only"
            else TemporalGatedDeltaNet(
                self.d_latent,
                self.n_layers,
                num_hidden_layers=getattr(self.config, "temporal_deltanet_num_layers", 2),
                use_history_gate=self.config.temporal_aggregator == "gated_residual",
            )
            if self.config.temporal_aggregator is not None
            else None
        )
        self.temporal_query_weighting = _build_temporal_query_weighting(self.config)

        if self.target_modules:
            if self.config.per_layer_processing:
                layers = [
                    ResMLPBlockPerLayer(
                        self.n_layers,
                        self.d_latent,
                        self.d_latent * 4,
                        self.d_latent,
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]
            else:
                layers = [
                    ResMLPBlock(
                        input_size=self.config.latent_size,
                        hidden_size=self.config.latent_size * 4,
                        output_size=self.config.latent_size,
                        dropout_rate=getattr(self.config, "dropout_rate", 0),
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]

            self.layers = nn.Sequential(*layers)

            self.d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

            self.bias_A = nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.normal(
                            0,
                            0.2 / (self.d_in[m] * self.r) ** 0.5,
                            (self.n_layers, self.r, self.d_in[m]),
                        )
                    )
                    for m in self.target_modules
                }
            )
            self.bias_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((self.n_layers, self.r, self.d_out[m])))
                    for m in self.target_modules
                }
            )

            self.scaler_A = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.ones((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )
            self.scaler_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )

            n_modules = len(self.target_modules)
            head_axes = []
            head_axis_sizes = {}
            # import ipdb; ipdb.set_trace()
            if self.config.head_per_layer:
                head_axes.append("n_layers")
                head_axis_sizes["n_layers"] = len(self.layer_indices)
            if self.config.head_per_module:
                head_axes.append("n_modules")
                head_axis_sizes["n_modules"] = n_modules
            if self.config.head_per_rank:
                head_axes.append("r")
                head_axis_sizes["r"] = self.config.lora_config.r

            self.head = Mix(
                "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                weight_shape=" ".join([*head_axes, "d_latent", "d_lora"]),
                bias_shape=None,
                d_latent=self.config.latent_size,
                d_lora=self.d_lora,
                **head_axis_sizes,
            )

    def get_head_bias(self):
        bias_dict = dict()
        for module in self.target_modules:
            bias_A = self.bias_A[module]
            bias_B = self.bias_B[module]

            bias_dict[module] = dict(A=bias_A, B=bias_B)
        return bias_dict

    def _to_lora_dict(
        self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
    ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
        if self.target_modules is None:
            return None
        # list of [bs, n_layers, r, in_d_outim]
        # and in_d_outim might vary across modules
        loras = unpack(
            flat_loras,
            [[] for _ in range(len(self.target_modules))],
            "bs n_layers * r max_io_dim",
        )

        # dict of {module:
        #   {A: [bs, n_layers, r, d_inim],
        #    B: [bs, n_layers, r, d_outim]}}
        lora_dict = dict()
        for module, lora in zip(self.target_modules, loras):
            A, B = unpack(
                lora[..., : self.d_in[module] + self.d_out[module]],
                [[self.d_in[module]], [self.d_out[module]]],
                "bs n_layers r *",
            )

            # apparently doing A * self.scaler_A is slow due to broadcasting
            A = torch.einsum("ijkl,ijkl->ijkl", A, self.scaler_A[module])
            B = torch.einsum("ijkl,ijkl->ijkl", B, self.scaler_B[module])

            lora_dict[module] = dict(A=A, B=B)

        return lora_dict

    def _to_layernorm_dict(
        self, flat_layernorms: Float[Tensor, "bs n_layers n_modules hidden_size"]
    ) -> dict[str, Float[Tensor, "bs n_layers hidden_size"]]:
        if self.extra_modules is None:
            return None
        layernorms = unpack(
            flat_layernorms,
            [[] for _ in range(len(self.extra_modules))],
            "bs n_layers * hidden_size",
        )
        return {k: v for k, v in zip(self.extra_modules, layernorms)}

    def enable_iterative_mode(self, x: bool):
        self.iterative_mode = x
        self.aggregator.enable_iterative_mode(x)

    def aggregate_lora_embeddings(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
    ) -> Tensor:
        # [bs, n_layers, n_total_modules, r, feature_dim]
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if features.is_cuda
            else nullcontext()
        )
        with autocast_ctx:
            if self.aggregator.layer_to_layer and self.iterative_mode:
                if features.dim() != 4:
                    raise ValueError(
                        "Layer-to-layer iterative mode expects features shaped "
                        "[batch, num_layers, seq_len, hidden]."
                    )
                # iterative inference
                # features: [bs num_layers seq_len feature_dim]
                bs, n_layers = features.shape[0:2]
                lora_emb = torch.empty(
                    (bs, n_layers, self.num_modules, self.r, self.config.latent_size),
                    device=features.device,
                )
                for i in range(n_layers):
                    lora_emb[:, i], _ = self.aggregator(
                        features[:, i], attn_mask, position_ids
                    )

            else:
                # batched inference
                if self.aggregator.layer_to_layer and features.dim() == 3:
                    features = repeat(
                        features,
                        "bs seq_len feature_dim -> bs num_layers seq_len feature_dim",
                        num_layers=self.aggregator.num_layers,
                    )
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)

        return lora_emb

    def _project_lora_embeddings(self, lora_emb: Tensor) -> Tensor | None:
        # [bs, n_layers, n_modules, r, max_in_d_outim]
        flat_loras = None
        if self.target_modules:
            layer_dtype = next(self.layers.parameters()).dtype
            if lora_emb.is_floating_point() and lora_emb.dtype != layer_dtype:
                lora_emb = lora_emb.to(layer_dtype)
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            norm_lora_emb = lora_emb / norm
            flat_loras = self.head(norm_lora_emb)

        return flat_loras

    def generate_weights_from_lora_embeddings(
        self, lora_emb: Tensor
    ) -> tuple[dict[str, dict[str, Tensor]] | None, None]:
        """Project already-aggregated C2L queries into LoRA factors."""
        return self._to_lora_dict(self._project_lora_embeddings(lora_emb)), None

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
    ):
        lora_emb = self.aggregate_lora_embeddings(
            features, attn_mask, position_ids, n_ctx_chunks
        )
        return self._project_lora_embeddings(lora_emb), None

    def generate_weights(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        flat_loras, flat_layernorms = self.forward(features, attn_mask, position_ids)
        return self._to_lora_dict(flat_loras), self._to_layernorm_dict(flat_layernorms)


class HyperLoRA_valid_slots(nn.Module):
    def __init__(self, config: HypernetConfig):
        super().__init__()
        self.config = ensure_hypernet_config_compat(config)
        self.iterative_mode = False
        self._init_model()

    def _init_model(self):
        self.agg_config = self.config.aggregator_config
        self.aggregator = AGGREGATOR_CLS[self.agg_config.aggregator_type](
            **vars(self.agg_config)
        )

        self.lora_config = self.config.lora_config
        self.r = self.lora_config.r
        self.target_slots = list(self.config.target_slots or [])
        if not self.target_slots:
            raise ValueError("HyperLoRA_valid_slots requires non-empty target_slots.")

        self.layer_indices = self.config.layer_indices
        self.layer_positions = {
            int(layer_idx): layer_pos for layer_pos, layer_idx in enumerate(self.layer_indices)
        }
        self.n_layers = len(self.layer_indices)
        self.d_latent = self.config.latent_size
        self.temporal_aggregator = (
            TemporalHistoryGate(self.n_layers)
            if self.config.temporal_aggregator == "history_only"
            else TemporalGatedDeltaNet(
                self.d_latent,
                self.n_layers,
                num_hidden_layers=getattr(self.config, "temporal_deltanet_num_layers", 2),
                use_history_gate=self.config.temporal_aggregator == "gated_residual",
            )
            if self.config.temporal_aggregator is not None
            else None
        )
        self.temporal_query_weighting = _build_temporal_query_weighting(self.config)

        target_modules = []
        for slot in self.target_slots:
            target_key = str(slot["target_key"])
            if target_key not in target_modules:
                target_modules.append(target_key)
        self.target_modules = tuple(target_modules)
        self.target_to_module_idx = {
            target_key: idx for idx, target_key in enumerate(self.target_modules)
        }
        self.num_modules = len(self.target_modules)

        layers = [
            (
                ResMLPBlockPerLayer(
                    self.n_layers,
                    self.d_latent,
                    self.d_latent * 4,
                    self.d_latent,
                )
                if self.config.per_layer_processing
                else ResMLPBlock(
                    input_size=self.config.latent_size,
                    hidden_size=self.config.latent_size * 4,
                    output_size=self.config.latent_size,
                    dropout_rate=getattr(self.config, "dropout_rate", 0),
                )
            )
            for _ in range(self.config.num_pre_head_layers)
        ]
        self.layers = nn.Sequential(*layers)

        self.slots_by_target: dict[str, list[dict[str, int | str]]] = {
            target_key: [] for target_key in self.target_modules
        }
        for slot in self.target_slots:
            self.slots_by_target[str(slot["target_key"])].append(slot)

        self.d_in: dict[str, int] = {}
        self.d_out: dict[str, int] = {}
        for slot in self.target_slots:
            slot_key = str(slot["slot_key"])
            self.d_in[slot_key] = int(slot["d_in"])
            self.d_out[slot_key] = int(slot["d_out"])

        self.d_lora = max(self.d_in[s["slot_key"]] + self.d_out[s["slot_key"]] for s in self.target_slots)
        self.heads = nn.ModuleDict()
        self.bias_A = nn.ParameterDict()
        self.bias_B = nn.ParameterDict()
        self.scaler_A = nn.ParameterDict()
        self.scaler_B = nn.ParameterDict()

        for target_key, slots in self.slots_by_target.items():
            slot_d_loras = {
                int(slot["d_in"]) + int(slot["d_out"])
                for slot in slots
            }
            if len(slot_d_loras) != 1:
                raise ValueError(
                    f"Target {target_key} has heterogeneous LoRA dimensions across valid slots: "
                    f"{sorted(slot_d_loras)}."
                )
            d_lora = slot_d_loras.pop()
            head_axes = []
            head_axis_sizes = {}
            if self.config.head_per_layer:
                head_axes.append("n_slots")
                head_axis_sizes["n_slots"] = len(slots)
            if self.config.head_per_rank:
                head_axes.append("r")
                head_axis_sizes["r"] = self.r
            self.heads[safe_lora_target_key(target_key)] = Mix(
                "bs n_slots r d_latent -> bs n_slots r d_lora",
                weight_shape=" ".join([*head_axes, "d_latent", "d_lora"]),
                bias_shape=None,
                d_latent=self.config.latent_size,
                d_lora=d_lora,
                **head_axis_sizes,
            )

            for slot in slots:
                slot_key = str(slot["slot_key"])
                d_in = int(slot["d_in"])
                d_out = int(slot["d_out"])
                self.bias_A[slot_key] = nn.Parameter(
                    torch.normal(
                        0,
                        0.2 / (d_in * self.r) ** 0.5,
                        (1, self.r, d_in),
                    )
                )
                self.bias_B[slot_key] = nn.Parameter(torch.zeros((1, self.r, d_out)))
                self.scaler_A[slot_key] = nn.Parameter(torch.ones((1, 1, self.r, 1)))
                self.scaler_B[slot_key] = nn.Parameter(torch.zeros((1, 1, self.r, 1)))

    def get_head_bias(self):
        return {
            str(slot["slot_key"]): {
                "A": self.bias_A[str(slot["slot_key"])],
                "B": self.bias_B[str(slot["slot_key"])],
            }
            for slot in self.target_slots
        }

    def _gather_target_slot_embeddings(self, lora_emb: Tensor, target_key: str) -> Tensor:
        slot_embeddings = []
        target_module_idx = self.target_to_module_idx[target_key]
        for slot in self.slots_by_target[target_key]:
            layer_pos = self.layer_positions[int(slot["layer_idx"])]
            slot_embeddings.append(lora_emb[:, layer_pos, target_module_idx])
        return torch.stack(slot_embeddings, dim=1)

    def _to_lora_dict(self, lora_emb: Tensor) -> dict[str, dict[str, Tensor]]:
        """Project decoder-layer slot embeddings to LoRA A/B matrices."""

        lora_dict: dict[str, dict[str, Tensor]] = {}
        for target_key, slots in self.slots_by_target.items():
            target_emb = self._gather_target_slot_embeddings(lora_emb, target_key)
            flat_loras = self.heads[safe_lora_target_key(target_key)](target_emb)
            d_in = int(slots[0]["d_in"])
            d_out = int(slots[0]["d_out"])
            A, B = unpack(
                flat_loras[..., : d_in + d_out],
                [[d_in], [d_out]],
                "bs n_slots r *",
            )
            for slot_idx, slot in enumerate(slots):
                slot_key = str(slot["slot_key"])
                slot_A = A[:, slot_idx : slot_idx + 1]
                slot_B = B[:, slot_idx : slot_idx + 1]
                slot_A = torch.einsum("ijkl,ijkl->ijkl", slot_A, self.scaler_A[slot_key])
                slot_B = torch.einsum("ijkl,ijkl->ijkl", slot_B, self.scaler_B[slot_key])
                lora_dict[slot_key] = {"A": slot_A, "B": slot_B}
        return lora_dict

    def enable_iterative_mode(self, x: bool):
        self.iterative_mode = x
        self.aggregator.enable_iterative_mode(x)

    def aggregate_lora_embeddings(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
    ) -> Tensor:
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if features.is_cuda
            else nullcontext()
        )
        with autocast_ctx:
            if self.aggregator.layer_to_layer and self.iterative_mode:
                if features.dim() != 4:
                    raise ValueError(
                        "Layer-to-layer iterative mode expects features shaped "
                        "[batch, num_layers, seq_len, hidden]."
                    )
                bs, n_layers = features.shape[0:2]
                lora_emb = torch.empty(
                    (bs, n_layers, self.num_modules, self.r, self.config.latent_size),
                    device=features.device,
                )
                for i in range(n_layers):
                    lora_emb[:, i], _ = self.aggregator(
                        features[:, i], attn_mask, position_ids
                    )
            else:
                if self.aggregator.layer_to_layer and features.dim() == 3:
                    features = repeat(
                        features,
                        "bs seq_len feature_dim -> bs num_layers seq_len feature_dim",
                        num_layers=self.aggregator.num_layers,
                    )
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)

        return lora_emb

    def _prepare_lora_embeddings_for_head(self, lora_emb: Tensor) -> Tensor:
        layer_dtype = next(self.layers.parameters()).dtype
        if lora_emb.is_floating_point() and lora_emb.dtype != layer_dtype:
            lora_emb = lora_emb.to(layer_dtype)
        lora_emb = self.layers(lora_emb)
        norm = torch.norm(lora_emb, dim=-1, keepdim=True)
        return lora_emb / norm

    def generate_weights_from_lora_embeddings(
        self, lora_emb: Tensor
    ) -> tuple[dict[str, dict[str, Tensor]], None]:
        """Project already-aggregated C2L queries into valid-slot LoRA factors."""
        return self._to_lora_dict(self._prepare_lora_embeddings_for_head(lora_emb)), None

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
    ):
        lora_emb = self.aggregate_lora_embeddings(
            features, attn_mask, position_ids, n_ctx_chunks
        )
        return self._prepare_lora_embeddings_for_head(lora_emb)

    def generate_weights(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        lora_emb = self.forward(features, attn_mask, position_ids)
        return self._to_lora_dict(lora_emb), None


class ImagePerLayerContextEncoder(nn.Module):
    def __init__(self, encoder_model: PreTrainedModel, num_layers: int):
        super().__init__()
        self.base_model = encoder_model
        self.num_layers = num_layers

    @property
    def config(self):
        config = self.base_model.config
        return getattr(config, "text_config", config)

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in {"labels"} and v is not None}
        kwargs["output_hidden_states"] = True
        kwargs["return_dict"] = True
        outputs = self.base_model(**kwargs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if hidden_states is None and hasattr(outputs, "language_model_outputs"):
            hidden_states = getattr(outputs.language_model_outputs, "hidden_states", None)
        if hidden_states is None:
            raise ValueError("The C2L encoder model did not return hidden states.")

        if len(hidden_states) < self.num_layers:
            raise ValueError(
                f"The C2L encoder returned {len(hidden_states)} hidden states, "
                f"but the hypernet expects {self.num_layers} layers."
            )

        return torch.stack(tuple(hidden_states[: self.num_layers]), dim=1)


class ModulatedPretrainedModel(InteractiveC2LMixin, nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel | PeftModel,
        hypernet_config: HypernetConfig,
        ctx_encoder_args: CtxEncoderArguments,
        encoder_model: PreTrainedModel | None = None,
        image_context_encoder: bool = False,
        use_base_input_as_ctx: bool = False,
        # need non-packed inputs for generation
        use_sequence_packing: bool = True,
        user_defined_scaling: float = 1,
        inp_compressor=None,
        block_lora: bool = False,
        frame_per_lora: int = 4,
        normalize_add_lora: bool = False,
        interleave_timestamp: bool = False,
        micro_c2l_forward_frame: int | None = None,
        temporal_sum_at_query: bool = False,
        normalize_add_query: bool = False,
        temporal_aggregator: str | None = None,
        temporal_query_weighting: str | None = None,
        temporal_query_gate_threshold: float = 0.05,
    ):
        assert not use_base_input_as_ctx
        super().__init__()
        if frame_per_lora < 1:
            raise ValueError("frame_per_lora must be >= 1.")
        if micro_c2l_forward_frame is not None and micro_c2l_forward_frame < 1:
            raise ValueError("micro_c2l_forward_frame must be >= 1 when set.")
        self.device = getattr(base_model, "device", None)
        if self.device is None:
            self.device = next(base_model.parameters()).device
        hypernet_config = ensure_hypernet_config_compat(hypernet_config)
        if temporal_aggregator is None:
            temporal_aggregator = getattr(hypernet_config, "temporal_aggregator", None)
        if temporal_aggregator not in (None, "only", "gated_residual", "ssm_only", "history_only"):
            raise ValueError(f"Unsupported temporal aggregator mode: {temporal_aggregator!r}.")
        hypernet_config.temporal_aggregator = temporal_aggregator
        if temporal_query_weighting in ("", "none"):
            temporal_query_weighting = None
        if temporal_query_weighting is None:
            temporal_query_weighting = getattr(hypernet_config, "temporal_query_weighting", None)
        if temporal_query_weighting is not None:
            if temporal_query_weighting not in TEMPORAL_QUERY_WEIGHTING_MODES:
                raise ValueError(
                    f"Unsupported temporal query weighting mode: {temporal_query_weighting!r}."
                )
            if not temporal_sum_at_query:
                raise ValueError("Raw pooled-cosine weighting requires temporal_sum_at_query=true.")
            if temporal_aggregator not in ("gated_residual", "history_only"):
                raise ValueError(
                    "Raw pooled-cosine weighting changes the temporal history residual and requires "
                    "temporal_aggregator=gated_residual or history_only."
                )
        if not 0.0 <= float(temporal_query_gate_threshold) <= 2.0:
            raise ValueError("temporal_query_gate_threshold must be between 0 and 2.")
        hypernet_config.temporal_query_weighting = temporal_query_weighting
        hypernet_config.temporal_query_gate_threshold = float(temporal_query_gate_threshold)
        self.hypernet_config = hypernet_config
        self.peft_config = hypernet_config.lora_config
        self.ctx_encoder_args = ctx_encoder_args
        self.encoder_model = encoder_model
        self.image_context_encoder = image_context_encoder
        self.use_base_input_as_ctx = use_base_input_as_ctx
        self.use_sequence_packing = use_sequence_packing
        self.user_defined_scaling = user_defined_scaling
        self.inp_compressor = inp_compressor
        self.block_lora = bool(block_lora)
        self.frame_per_lora = int(frame_per_lora)
        self.normalize_add_lora = bool(normalize_add_lora)
        self.interleave_timestamp = bool(interleave_timestamp)
        self.micro_c2l_forward_frame = (
            int(micro_c2l_forward_frame) if micro_c2l_forward_frame is not None else None
        )
        self.temporal_sum_at_query = bool(temporal_sum_at_query)
        self.normalize_add_query = bool(normalize_add_query)
        self.temporal_aggregator = temporal_aggregator
        self.temporal_query_weighting = temporal_query_weighting
        self._logged_temporal_gate_retention = False
        self.model_accepts_loss_kwargs = True
        self.generated_loras = None

        self.register_module("base_model", base_model)
        self._init_model()
        self._bias_hyper_init()


    def patch_lora_forward(self):
        layers = get_layers(self.base_model)

        lora_forward_fn = (
            lora_forward_packed if self.use_sequence_packing else lora_forward
        )
        if getattr(self.hypernet_config, "use_valid_slots", False):
            for slot in self.hypernet.target_slots:
                module = attrgetter(str(slot["module_path"]))(layers[int(slot["layer_idx"])])
                if getattr(module, "patched_forward", False):
                    continue
                logger.debug(f"Applying LoRA forward to {slot['slot_key']}")
                module.forward_orig = module.forward
                module.patched_forward = True
                module.forward = partial(
                    lora_forward_fn,
                    self=module,
                    lora_dropout_p=self.peft_config.lora_dropout,
                    scaling=self.peft_config.lora_alpha,
                )
            return

        for layer_idx in self.hypernet.layer_indices:
            for module_info in get_lora_target_modules(layers[layer_idx], self.peft_config):
                name = module_info["name"]
                module = module_info["module"]
                if getattr(module, "patched_forward", False):
                    continue
                logger.debug(f"Applying LoRA forward to {name}")
                module.forward_orig = module.forward
                module.patched_forward = True
                module.forward = partial(
                    lora_forward_fn,
                    self=module,
                    lora_dropout_p=self.peft_config.lora_dropout,
                    scaling=self.peft_config.lora_alpha,
                )

    def _init_model(self):
        # disable adapter of the base model
        # this only works with LoRA(?)
        # we disable to avoid peft lora computation
        if hasattr(self.base_model, "disable_adapter_layers"):
            self.base_model.disable_adapter_layers()

        hypernet_cls = (
            HyperLoRA_valid_slots
            if getattr(self.hypernet_config, "use_valid_slots", False)
            else HyperLoRA
        )
        self.hypernet = (
            hypernet_cls(self.hypernet_config).to(self.device).to(torch.float32)
        )

        self.patch_lora_forward()

        if self.encoder_model is not None and self.image_context_encoder:
            self.encoder_model.requires_grad_(False)
            self.encoder_model.eval()
            if _ctx_encoder_type_is(
                self.ctx_encoder_args.ctx_encoder_type,
                CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS,
            ):
                self.ctx_encoder = ImagePerLayerContextEncoder(
                    self.encoder_model, num_layers=len(self.hypernet.layer_indices)
                )
            else:
                self.ctx_encoder = _get_ctx_encoder_cls(self.ctx_encoder_args.ctx_encoder_type)(
                    self.encoder_model, self.ctx_encoder_args
                )
        else:
            raise ValueError("Video inference requires an explicit frozen vision encoder.")

    # delegate to base_model
    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @property
    def vocab_size(self):
        return self.base_model.vocab_size

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @torch.no_grad()
    def _bias_hyper_init(self):
        if getattr(self.hypernet, "extra_modules", None):
            self.hypernet.extra_head.weight.data[:] = 0
            self.hypernet.extra_head.bias.data[:] = 0
        if self.hypernet.target_modules:
            r = self.hypernet_config.lora_config.r
            if isinstance(self.hypernet, HyperLoRA_valid_slots):
                for head in self.hypernet.heads.values():
                    d_lora = head.weight.shape[-1]
                    nn.init.normal_(
                        head.weight,
                        mean=0,
                        std=0.5
                        / sqrt(self.hypernet.config.latent_size + d_lora * r),
                    )
            else:
                nn.init.normal_(
                    self.hypernet.head.weight,
                    mean=0,
                    std=0.5
                    / sqrt(self.hypernet.config.latent_size + self.hypernet.d_lora * r),
                    # the head outputs per rank lora --> divide by r to scale down grad
                )


    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        original_state_dict = state_dict
        state_dict = state_dict.copy()
        if hasattr(original_state_dict, "_metadata"):
            state_dict._metadata = original_state_dict._metadata

        metadata_keys = {"base_model_name_or_path", "hypernet_config", "ctx_encoder_args"}
        has_metadata = any(key in state_dict for key in metadata_keys)
        if has_metadata:
            missing_keys = metadata_keys.difference(state_dict)
            if missing_keys:
                raise ValueError(
                    f"C2L checkpoint metadata is incomplete. Missing keys: {sorted(missing_keys)}"
                )

            self.base_model_name_or_path = state_dict.pop("base_model_name_or_path")
            self.hypernet_config = ensure_hypernet_config_compat(
                state_dict.pop("hypernet_config")
            )
            self.ctx_encoder_args = state_dict.pop("ctx_encoder_args")
            if self.base_model_name_or_path != self.base_model.name_or_path:
                raise ValueError(
                    f"Base model name or path mismatch. "
                    f"The base model given is: {self.base_model.name_or_path}, "
                    f"but the loaded name is: {self.base_model_name_or_path}"
                )
            self._init_model()

        normalize_add_lora_default = False if has_metadata else getattr(self, "normalize_add_lora", False)
        normalize_add_query_default = False if has_metadata else getattr(self, "normalize_add_query", False)
        temporal_aggregator_default = (
            getattr(self.hypernet_config, "temporal_aggregator", None)
            if has_metadata
            else getattr(self, "temporal_aggregator", None)
        )
        self.block_lora = bool(state_dict.pop("block_lora", getattr(self, "block_lora", False)))
        self.frame_per_lora = int(state_dict.pop("frame_per_lora", getattr(self, "frame_per_lora", 4)))
        self.normalize_add_lora = bool(
            state_dict.pop("normalize_add_lora", normalize_add_lora_default)
        )
        self.interleave_timestamp = bool(
            state_dict.pop("interleave_timestamp", getattr(self, "interleave_timestamp", False))
        )
        self.micro_c2l_forward_frame = state_dict.pop(
            "micro_c2l_forward_frame", getattr(self, "micro_c2l_forward_frame", None)
        )
        if self.micro_c2l_forward_frame is not None:
            self.micro_c2l_forward_frame = int(self.micro_c2l_forward_frame)
        self.temporal_sum_at_query = bool(
            state_dict.pop("temporal_sum_at_query", getattr(self, "temporal_sum_at_query", False))
        )
        self.normalize_add_query = bool(
            state_dict.pop("normalize_add_query", normalize_add_query_default)
        )
        self.temporal_aggregator = state_dict.pop(
            "temporal_aggregator", temporal_aggregator_default
        )
        temporal_query_weighting_default = (
            getattr(self.hypernet_config, "temporal_query_weighting", None)
            if has_metadata
            else getattr(self, "temporal_query_weighting", None)
        )
        self.temporal_query_weighting = state_dict.pop(
            "temporal_query_weighting", temporal_query_weighting_default
        )
        temporal_query_gate_threshold = float(
            state_dict.pop(
                "temporal_query_gate_threshold",
                getattr(self.hypernet_config, "temporal_query_gate_threshold", 0.05),
            )
        )
        self.hypernet_config.temporal_query_weighting = self.temporal_query_weighting
        self.hypernet_config.temporal_query_gate_threshold = temporal_query_gate_threshold
        if self.temporal_query_weighting is not None and getattr(
            self.hypernet, "temporal_query_weighting", None
        ) is None:
            raise ValueError(
                "This C2L checkpoint weights temporal queries, but the rebuilt hypernet has no "
                "temporal weighting module."
            )
        if isinstance(getattr(self.hypernet, "temporal_query_weighting", None), RawPooledCosineGate):
            self.hypernet.temporal_query_weighting.threshold = temporal_query_gate_threshold
        self.base_input_video = bool(state_dict.pop("base_input_video", getattr(self, "base_input_video", False)))
        self.base_input_frame_number = int(
            state_dict.pop("base_input_frame_number", getattr(self, "base_input_frame_number", 4))
        )
        self.base_with_last_2seconds = bool(
            state_dict.pop("base_with_last_2seconds", getattr(self, "base_with_last_2seconds", False))
        )
        self.base_tail_seconds = float(state_dict.pop("base_tail_seconds", 2.0))
        if not isfinite(self.base_tail_seconds) or self.base_tail_seconds <= 0:
            raise ValueError("base_tail_seconds must be finite and > 0.")
        self.base_last_2seconds_max_frames = int(
            state_dict.pop(
                "base_last_2seconds_max_frames",
                getattr(self, "base_last_2seconds_max_frames", 128),
            )
        )
        self.video_merge_extend = bool(state_dict.pop("video_merge_extend", getattr(self, "video_merge_extend", False)))
        self.video_merge_extend_perc = float(
            state_dict.pop("video_merge_extend_perc", getattr(self, "video_merge_extend_perc", 0.0))
        )
        if self.frame_per_lora < 1:
            raise ValueError("frame_per_lora must be >= 1.")
        if self.micro_c2l_forward_frame is not None and self.micro_c2l_forward_frame < 1:
            raise ValueError("micro_c2l_forward_frame must be >= 1 when set.")
        if self.base_input_frame_number < 1:
            raise ValueError("base_input_frame_number must be >= 1.")
        if self.base_last_2seconds_max_frames < 1:
            raise ValueError("base_last_2seconds_max_frames must be >= 1.")
        if not 0.0 <= self.video_merge_extend_perc <= 1.0:
            raise ValueError("video_merge_extend_perc must be between 0 and 1.")

        def remove_compile_prefix(sd: dict[str, Tensor]) -> dict[str, Tensor]:
            COMPILED_PREFIX = "_orig_mod."
            for k in list(sd.keys()):
                if k.startswith(COMPILED_PREFIX):
                    sd[k[len(COMPILED_PREFIX) :]] = sd.pop(k)
            return sd

        load_result = self.hypernet.load_state_dict(
            remove_compile_prefix(state_dict),
            strict=True,  # , *args, **kwargs
        )
        logger.info(f"load result: {load_result}")
        return load_result

    def _default_image_n_ctx_chunks(
        self, generated_loras: dict[str, dict[str, Tensor]]
    ) -> Integer[Tensor, "n_ctx"]:
        sampled_module = next(iter(generated_loras.values()))
        sampled_lora = sampled_module["A"]
        return torch.ones(
            sampled_lora.shape[0], dtype=torch.int32, device=sampled_lora.device
        )

    def _prepare_single_frame_video_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        """Turn Qwen's packed (f0, f1) tubelets into (f0, f0), (f1, f1).

        Processor resizing/normalization and pretrained patch weights stay intact.
        Only the C2L encoder uses these repeated pairs; base inputs are untouched.
        The resulting grid counts logical source frames, including existing padding.
        """
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

        if not _ctx_encoder_type_is(self.ctx_encoder_args.ctx_encoder_type, CTX_ENCODER_TYPE.VISION_EMBEDDER):
            raise ValueError("Single-frame C2L blocks require Qwen3.5 vision_embedder.")
        visual = self.ctx_encoder._get_visual_model()
        config = getattr(visual, "config", None)
        # Older checkpoints retain model_type="qwen3_5" on this same vision
        # config class; the serialized/logged type can differ from that value.
        if not isinstance(config, Qwen3_5VisionConfig):
            raise ValueError("Single-frame C2L blocks require the Qwen3.5 packed video patch layout.")
        patch_size = int(config.patch_size)
        channels = int(config.in_channels)
        if int(config.temporal_patch_size) != 2:
            raise ValueError("Single-frame Qwen3.5 C2L expects temporal_patch_size=2.")
        pixels = inputs["pixel_values_videos"]
        grid = inputs["video_grid_thw"]
        if pixels.ndim != 2 or pixels.shape[1] != channels * 2 * patch_size**2:
            raise ValueError("Single-frame C2L received incompatible packed video pixels.")
        if grid.ndim != 2 or grid.shape[1] != 3 or bool((grid <= 0).any()):
            raise ValueError("Single-frame C2L requires positive video_grid_thw entries.")
        if int(grid.prod(-1).sum().item()) != pixels.shape[0]:
            raise ValueError("Single-frame C2L video grid does not match packed pixels.")

        repeated_videos = []
        offset = 0
        for temporal, height, width in grid.tolist():
            spatial = height * width
            count = temporal * spatial
            # Qwen packs each row as [channel, temporal, patch_y, patch_x].
            patches = pixels[offset : offset + count].reshape(
                temporal, spatial, channels, 2, patch_size, patch_size
            )
            frames = patches.permute(0, 3, 1, 2, 4, 5).reshape(
                temporal * 2, spatial, channels, 1, patch_size, patch_size
            )
            repeated_videos.append(
                frames.expand(-1, -1, -1, 2, -1, -1).reshape(count * 2, -1)
            )
            offset += count
        prepared = dict(inputs)
        prepared["pixel_values_videos"] = torch.cat(repeated_videos, dim=0)
        prepared["video_grid_thw"] = grid.clone()
        prepared["video_grid_thw"][:, 0] *= 2
        # Logical source-frame accounting for block splitting, not a change to
        # the visual encoder's physical two-frame convolution.
        prepared["video_temporal_patch_size"] = grid.new_ones(grid.shape[0])
        prepared.pop("video_metadata", None)
        return prepared

    def _video_token_layout(self) -> tuple[int, int]:
        visual_getter = getattr(self.ctx_encoder, "_get_visual_model", None)
        visual = visual_getter() if callable(visual_getter) else None
        spatial_merge_size = int(
            getattr(visual, "spatial_merge_size", getattr(visual, "merge_size", 1)) or 1
        )
        temporal_patch_size = int(getattr(visual, "temporal_patch_size", 1) or 1)
        return max(1, spatial_merge_size), max(1, temporal_patch_size)

    def _has_nonempty_grid(self, grid: Tensor | None) -> bool:
        return grid is not None and int(grid.numel()) > 0

    def _temporal_patch_size_from_value(self, value: Any, sample_idx: int, default: int) -> int:
        if value is None:
            return default
        if isinstance(value, Tensor):
            if int(value.numel()) == 0:
                return default
            flattened = value.reshape(-1)
            if int(flattened.numel()) == 1:
                patch_size = int(flattened[0].item())
            elif sample_idx < int(flattened.numel()):
                patch_size = int(flattened[sample_idx].item())
            else:
                raise ValueError(
                    "C2L block_lora video_temporal_patch_size has fewer entries than video samples: "
                    f"{int(flattened.numel())} <= sample index {sample_idx}."
                )
        elif isinstance(value, (list, tuple)):
            if len(value) == 0:
                return default
            patch_size = int(value[0] if len(value) == 1 else value[sample_idx])
        else:
            patch_size = int(value)

        return max(1, patch_size)

    def _int_if_scalar(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, Tensor):
            return int(value.reshape(-1)[0].item()) if int(value.numel()) == 1 else None
        if isinstance(value, Mapping):
            if len(value) == 1:
                return self._int_if_scalar(next(iter(value.values())))
            return None
        if isinstance(value, (list, tuple)):
            if len(value) == 1:
                return self._int_if_scalar(value[0])
            return None
        if isinstance(value, (int, float, str)):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None

    def _len_if_sequence(self, value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, Tensor):
            return int(value.numel()) if value.dim() == 1 else None
        if isinstance(value, (str, bytes)):
            return None
        if isinstance(value, Mapping):
            if len(value) == 1:
                return self._len_if_sequence(next(iter(value.values())))
            return None
        if isinstance(value, (list, tuple)):
            return len(value)
        return None

    def _num_video_frames_from_metadata(self, video_metadata: Any, sample_idx: int) -> int | None:
        if video_metadata is None or isinstance(video_metadata, Tensor):
            return None
        if isinstance(video_metadata, (list, tuple)):
            if len(video_metadata) == 0:
                return None
            metadata = video_metadata[0] if len(video_metadata) == 1 else video_metadata[sample_idx]
        else:
            metadata = video_metadata

        if isinstance(metadata, Mapping):
            frames_indices = metadata.get("frames_indices")
            total_num_frames = metadata.get("total_num_frames")
        else:
            frames_indices = getattr(metadata, "frames_indices", None)
            total_num_frames = getattr(metadata, "total_num_frames", None)

        frame_count = self._len_if_sequence(frames_indices)
        if frame_count is not None:
            return frame_count

        return self._int_if_scalar(total_num_frames)

    def _effective_temporal_patch_size(
        self,
        temporal_units: int,
        sample_idx: int,
        default: int,
        video_temporal_patch_size: Any = None,
        video_metadata: Any = None,
    ) -> int:
        num_frames = self._num_video_frames_from_metadata(video_metadata, sample_idx)
        if num_frames is not None and temporal_units > 0:
            return max(1, (int(num_frames) + temporal_units - 1) // temporal_units)

        return self._temporal_patch_size_from_value(video_temporal_patch_size, sample_idx, default)

    def _interleave_block_timestamps(
        self,
        video_blocks: list[Tensor],
        timestamp_input_ids: Tensor | None,
        timestamp_attention_mask: Tensor | None,
        timestamp_block_owners: Tensor | None,
        block_owners: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Append frozen timestamp embeddings to their corresponding visual blocks."""
        if (
            timestamp_input_ids is None
            or timestamp_attention_mask is None
            or timestamp_block_owners is None
        ):
            raise ValueError("C2L timestamp interleaving requires timestamp input ids, masks, and block owners.")
        if timestamp_input_ids.dim() != 2 or timestamp_attention_mask.shape != timestamp_input_ids.shape:
            raise ValueError("C2L timestamp interleaving requires matching rank-2 timestamp ids and masks.")
        if timestamp_input_ids.shape[0] != len(video_blocks):
            raise ValueError(
                "C2L timestamp interleaving requires one timestamp block per generated video block: "
                f"timestamps={timestamp_input_ids.shape[0]}, video_blocks={len(video_blocks)}."
            )
        expected_owners = block_owners.to(device=timestamp_block_owners.device, dtype=torch.long)
        actual_owners = timestamp_block_owners.reshape(-1).to(dtype=torch.long)
        if not torch.equal(actual_owners, expected_owners):
            raise ValueError(
                "C2L timestamp interleaving timestamp block owners do not match generated video block owners."
            )

        with torch.no_grad():
            timestamp_embeddings = self.base_model.get_input_embeddings()(timestamp_input_ids)

        interleaved_blocks = []
        for block_idx, video_block in enumerate(video_blocks):
            timestamp_block = timestamp_embeddings[block_idx][timestamp_attention_mask[block_idx].bool()]
            if timestamp_block.shape[-1] != video_block.shape[-1]:
                raise ValueError(
                    "C2L timestamp embedding hidden size must match video feature hidden size: "
                    f"timestamps={timestamp_block.shape[-1]}, video={video_block.shape[-1]}."
                )
            interleaved_blocks.append(torch.cat([video_block, timestamp_block.to(video_block)], dim=0))

        padded_features = pad_sequence(interleaved_blocks, batch_first=True, padding_value=0.0)
        block_attn_mask = torch.zeros(
            (len(interleaved_blocks), padded_features.shape[1]),
            dtype=torch.long,
            device=padded_features.device,
        )
        for block_idx, feature in enumerate(interleaved_blocks):
            block_attn_mask[block_idx, : feature.shape[0]] = 1

        return padded_features, block_attn_mask, block_owners.to(device=padded_features.device, dtype=torch.long)

    def _build_video_block_features(
        self,
        ctx_features: Tensor,
        ctx_attn_mask: Tensor | None,
        video_grid_thw: Tensor | None,
        video_temporal_patch_size: Any = None,
        video_metadata: Any = None,
        timestamp_input_ids: Tensor | None = None,
        timestamp_attention_mask: Tensor | None = None,
        timestamp_block_owners: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if not self._has_nonempty_grid(video_grid_thw):
            raise ValueError("C2L block_lora requires video encoder inputs with `video_grid_thw`.")
        if ctx_features.dim() != 3:
            raise ValueError(
                "C2L block_lora expects vision features shaped [batch, seq_len, hidden], "
                f"got {tuple(ctx_features.shape)}."
            )
        if video_grid_thw.shape[0] != ctx_features.shape[0]:
            raise ValueError(
                "C2L block_lora currently expects exactly one video context per sample: "
                f"video_grid_thw has {video_grid_thw.shape[0]} rows but features batch has {ctx_features.shape[0]}."
            )

        spatial_merge_size, visual_temporal_patch_size = self._video_token_layout()
        valid_lens = (
            ctx_attn_mask.sum(dim=1).to(torch.long)
            if ctx_attn_mask is not None
            else torch.full(
                (ctx_features.shape[0],),
                ctx_features.shape[1],
                dtype=torch.long,
                device=ctx_features.device,
            )
        )

        block_features = []
        block_owners = []
        for sample_idx, grid in enumerate(video_grid_thw.to(device=ctx_features.device)):
            temporal_units = int(grid[0].item())
            temporal_patch_size = self._effective_temporal_patch_size(
                temporal_units,
                sample_idx,
                visual_temporal_patch_size,
                video_temporal_patch_size,
                video_metadata,
            )
            if getattr(self, "interleave_timestamp", False) and self.frame_per_lora % temporal_patch_size:
                raise ValueError(
                    "C2L timestamp interleaving requires frame_per_lora to be divisible by the effective "
                    f"temporal patch size: frame_per_lora={self.frame_per_lora}, "
                    f"temporal patch size={temporal_patch_size}."
                )
            temporal_units_per_lora = max(
                1, (self.frame_per_lora + temporal_patch_size - 1) // temporal_patch_size
            )
            height_patches = int(grid[1].item())
            width_patches = int(grid[2].item())
            tokens_per_temporal = max(
                1,
                (height_patches * width_patches) // (spatial_merge_size**2),
            )
            expected_tokens = temporal_units * tokens_per_temporal
            valid_len = int(valid_lens[sample_idx].item())
            if expected_tokens > valid_len:
                raise ValueError(
                    "C2L block_lora video token metadata is inconsistent with encoder features: "
                    f"expected {expected_tokens} tokens from grid {tuple(int(x.item()) for x in grid)}, "
                    f"but feature length is {valid_len}."
                )

            for temporal_start in range(0, temporal_units, temporal_units_per_lora):
                token_start = temporal_start * tokens_per_temporal
                token_end = min(temporal_units, temporal_start + temporal_units_per_lora) * tokens_per_temporal
                token_end = min(token_end, valid_len)
                if token_end <= token_start:
                    continue
                block_features.append(ctx_features[sample_idx, token_start:token_end])
                block_owners.append(sample_idx)

        if not block_features:
            raise ValueError("C2L block_lora produced no video feature blocks.")

        owners = torch.tensor(block_owners, dtype=torch.long, device=ctx_features.device)
        # Score only pooled visual features. Timestamp embeddings are appended below and
        # must not influence the raw adjacent-block cosine distance.
        pooled_blocks = torch.stack(
            [block.float().mean(dim=0) for block in block_features], dim=0
        )
        if getattr(self, "interleave_timestamp", False):
            padded_features, block_attn_mask, owners = self._interleave_block_timestamps(
                block_features,
                timestamp_input_ids,
                timestamp_attention_mask,
                timestamp_block_owners,
                owners,
            )
            return padded_features, block_attn_mask, owners, pooled_blocks

        padded_features = pad_sequence(block_features, batch_first=True, padding_value=0.0)
        block_attn_mask = torch.zeros(
            (len(block_features), padded_features.shape[1]),
            dtype=torch.long,
            device=padded_features.device,
        )
        for block_idx, feature in enumerate(block_features):
            block_attn_mask[block_idx, : feature.shape[0]] = 1

        return (
            padded_features,
            block_attn_mask,
            owners.to(device=padded_features.device),
            pooled_blocks,
        )

    def _sum_loras_by_owner(
        self,
        generated_loras: dict[str, dict[str, Tensor]],
        block_owners: Tensor,
        num_samples: int,
    ) -> dict[str, dict[str, Tensor]]:
        summed_loras: dict[str, dict[str, Tensor]] = {}
        for module_name, module_loras in generated_loras.items():
            summed_loras[module_name] = {}
            for matrix_key in ("A", "B"):
                values = module_loras[matrix_key]
                summed = values.new_zeros((num_samples, *values.shape[1:]))
                summed.index_add_(0, block_owners, values)
                summed_loras[module_name][matrix_key] = summed
        return summed_loras

    def _sum_lora_embeddings_by_owner(
        self,
        lora_embeddings: Tensor,
        block_owners: Tensor,
        num_samples: int,
    ) -> Tensor:
        """Sum per-block C2L query embeddings into one embedding per video."""
        summed_embeddings = lora_embeddings.new_zeros(
            (num_samples, *lora_embeddings.shape[1:])
        )
        return summed_embeddings.index_add_(0, block_owners, lora_embeddings)

    def _sum_lora_embeddings_by_owner_fp32(
        self,
        lora_embeddings: Tensor,
        block_owners: Tensor,
        num_samples: int,
    ) -> Tensor:
        """Sum per-block C2L query embeddings in FP32 for stable normalization."""
        float_embeddings = lora_embeddings.float()
        summed_embeddings = float_embeddings.new_zeros(
            (num_samples, *float_embeddings.shape[1:])
        )
        return summed_embeddings.index_add_(0, block_owners, float_embeddings)

    def _mean_normalize_summed_lora_embeddings(
        self,
        summed_embeddings: Tensor,
        block_owners: Tensor,
        num_samples: int,
    ) -> Tensor:
        """Mean-pool complete per-video sums and normalize them to sqrt(latent size)."""
        block_counts = torch.bincount(block_owners.to(dtype=torch.long), minlength=num_samples)
        if block_counts.shape[0] != num_samples or torch.any(block_counts == 0):
            raise ValueError(
                "C2L normalized temporal query aggregation requires at least one block per sample."
            )

        scales = block_counts.to(device=summed_embeddings.device, dtype=summed_embeddings.dtype)
        scales = scales.reshape(num_samples, *([1] * (summed_embeddings.dim() - 1)))
        mean_query = summed_embeddings / scales
        return F.normalize(mean_query, dim=-1, eps=1e-6) * sqrt(
            self.hypernet.config.latent_size
        )

    def _raw_temporal_query_weights(
        self,
        pooled_blocks: Tensor,
        block_owners: Tensor,
        num_samples: int,
    ) -> Tensor | None:
        mode = getattr(self, "temporal_query_weighting", None)
        if mode is None:
            return None
        weighting = getattr(self.hypernet, "temporal_query_weighting", None)
        if weighting is None:
            raise ValueError("Raw pooled-cosine weighting is enabled but the hypernet has no weighting module.")

        scores = raw_pooled_cosine_distances(pooled_blocks, block_owners)
        weights = weighting(scores, block_owners, num_samples)
        if mode == "raw_pooled_cosine_gate" and not self._logged_temporal_gate_retention:
            distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
            rank = torch.distributed.get_rank() if distributed else 0
            if rank == 0:
                retained = int((weights > 0).sum().item())
                total = int(weights.numel())
                logger.info(
                    "Raw pooled-cosine gate first-batch retention: %d/%d blocks (%.2f%%), threshold=%.6g",
                    retained,
                    total,
                    100.0 * retained / max(1, total),
                    float(getattr(self.hypernet_config, "temporal_query_gate_threshold", 0.05)),
                )
            self._logged_temporal_gate_retention = True
        return weights

    def _weighted_mean_lora_embeddings(
        self,
        lora_embeddings: Tensor,
        block_owners: Tensor,
        block_weights: Tensor,
        num_samples: int,
    ) -> Tensor:
        """Apply normalized per-block weights and combine query embeddings by video."""
        float_embeddings = lora_embeddings.float()
        scales = block_weights.to(device=float_embeddings.device, dtype=float_embeddings.dtype)
        scales = scales.reshape(-1, *([1] * (float_embeddings.dim() - 1)))
        weighted_embeddings = float_embeddings * scales
        combined = weighted_embeddings.new_zeros(
            (num_samples, *weighted_embeddings.shape[1:])
        )
        return combined.index_add_(0, block_owners, weighted_embeddings)

    def _aggregate_temporal_lora_embeddings(
        self,
        block_embeddings: Tensor,
        block_owners: Tensor,
        num_samples: int,
        block_weights: Tensor | None = None,
    ) -> Tensor:
        """Run every temporal query causally and weight only the gated-residual history."""
        block_counts = torch.bincount(block_owners.to(dtype=torch.long), minlength=num_samples)
        if block_counts.shape[0] != num_samples or torch.any(block_counts == 0):
            raise ValueError(
                "C2L temporal Gated DeltaNet requires at least one block per sample."
            )
        if torch.any(block_owners < 0) or torch.any(block_owners >= num_samples):
            raise ValueError("C2L temporal block owners must index an input sample.")

        if self.temporal_aggregator == "history_only":
            return self._temporal_history_residual(
                block_embeddings, block_owners, num_samples, block_weights
            )

        _, num_layers, num_modules, rank, latent_size = block_embeddings.shape
        aggregator_config = self.hypernet.config.aggregator_config
        layer_slice = slice(None) if getattr(aggregator_config, "query_per_layer", False) else slice(0, 1)
        module_slice = slice(None) if getattr(aggregator_config, "query_per_module", True) else slice(0, 1)
        rank_slice = slice(None) if getattr(aggregator_config, "query_per_rank", True) else slice(0, 1)
        unique_embeddings = block_embeddings[
            :, layer_slice, module_slice, rank_slice, :
        ]
        unique_layers, unique_modules, unique_rank = unique_embeddings.shape[1:4]

        max_blocks = int(block_counts.max().item())
        padded = unique_embeddings.new_zeros(
            (num_samples, max_blocks, unique_layers, unique_modules, unique_rank, latent_size)
        )
        temporal_mask = torch.zeros(
            (num_samples, max_blocks),
            device=block_embeddings.device,
            dtype=torch.long,
        )
        for sample_idx in range(num_samples):
            sample_embeddings = unique_embeddings[block_owners == sample_idx]
            sample_blocks = sample_embeddings.shape[0]
            padded[sample_idx, :sample_blocks] = sample_embeddings
            temporal_mask[sample_idx, :sample_blocks] = 1

        sequence = padded.permute(0, 2, 3, 4, 1, 5).reshape(
            num_samples * unique_layers * unique_modules * unique_rank,
            max_blocks,
            latent_size,
        )
        stream_mask = temporal_mask[:, None, None, None, :].expand(
            num_samples, unique_layers, unique_modules, unique_rank, max_blocks
        ).reshape(-1, max_blocks)

        temporal_module = self.hypernet.temporal_aggregator
        first_parameter = next(temporal_module.parameters(), None)
        if (
            not sequence.is_cuda
            and first_parameter is not None
            and sequence.is_floating_point()
            and sequence.dtype != first_parameter.dtype
        ):
            sequence = sequence.to(first_parameter.dtype)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if sequence.is_cuda
            else nullcontext()
        )
        with autocast_ctx:
            temporal_outputs = temporal_module(sequence, stream_mask)
        temporal_outputs = temporal_outputs.reshape(
            num_samples,
            unique_layers,
            unique_modules,
            unique_rank,
            max_blocks,
            latent_size,
        )
        last_outputs = torch.stack(
            [
                temporal_outputs[sample_idx, :, :, :, int(block_counts[sample_idx].item()) - 1]
                for sample_idx in range(num_samples)
            ],
            dim=0,
        )
        last_outputs = last_outputs.expand(
            num_samples, num_layers, num_modules, rank, latent_size
        )

        if self.temporal_aggregator == "only":
            return last_outputs
        if self.temporal_aggregator not in ("gated_residual", "ssm_only"):
            raise ValueError(f"Unsupported temporal aggregator mode: {self.temporal_aggregator!r}.")

        normalized_temporal = F.normalize(last_outputs.float(), dim=-1, eps=1e-6) * sqrt(
            self.hypernet.config.latent_size
        )
        if self.temporal_aggregator == "ssm_only":
            return normalized_temporal
        return normalized_temporal + self._temporal_history_residual(
            block_embeddings, block_owners, num_samples, block_weights
        )

    def _temporal_history_residual(
        self,
        block_embeddings: Tensor,
        block_owners: Tensor,
        num_samples: int,
        block_weights: Tensor | None,
    ) -> Tensor:
        """The same scaled history term for combined and history-only aggregation."""
        temporal_module = self.hypernet.temporal_aggregator
        if block_weights is None:
            history_sum = self._sum_lora_embeddings_by_owner_fp32(
                block_embeddings, block_owners, num_samples
            )
            normalized_history = self._mean_normalize_summed_lora_embeddings(
                history_sum, block_owners, num_samples
            )
        else:
            weighted_history = self._weighted_mean_lora_embeddings(
                block_embeddings, block_owners, block_weights, num_samples
            )
            normalized_history = F.normalize(weighted_history, dim=-1, eps=1e-6) * sqrt(
                self.hypernet.config.latent_size
            )
        history_gates = temporal_module.history_gate_logits.sigmoid().to(
            device=normalized_history.device,
            dtype=normalized_history.dtype,
        )
        if history_gates.numel() != block_embeddings.shape[1]:
            raise ValueError(
                "Temporal history gate count must match the number of generated base-model layers."
            )
        return history_gates.view(1, -1, 1, 1, 1) * normalized_history

    def _normalize_lora_sums_by_owner(
        self,
        summed_loras: dict[str, dict[str, Tensor]],
        block_owners: Tensor,
        num_samples: int,
    ) -> dict[str, dict[str, Tensor]]:
        if not getattr(self, "normalize_add_lora", False):
            return summed_loras

        block_counts = torch.bincount(block_owners.to(dtype=torch.long), minlength=num_samples)
        if block_counts.shape[0] != num_samples or torch.any(block_counts == 0):
            raise ValueError("C2L block_lora normalization requires at least one temporal block per sample.")

        normalized_loras: dict[str, dict[str, Tensor]] = {}
        for module_name, module_loras in summed_loras.items():
            normalized_loras[module_name] = {}
            for matrix_key in ("A", "B"):
                values = module_loras[matrix_key]
                scales = block_counts.to(device=values.device, dtype=values.dtype).rsqrt()
                scales = scales.reshape(num_samples, *([1] * (values.dim() - 1)))
                normalized_loras[module_name][matrix_key] = values * scales
        return normalized_loras

    def _accumulate_lora_sums(
        self,
        target: dict[str, dict[str, Tensor]] | None,
        increment: dict[str, dict[str, Tensor]],
    ) -> dict[str, dict[str, Tensor]]:
        if target is None:
            return increment
        for module_name, module_loras in increment.items():
            for matrix_key in ("A", "B"):
                target[module_name][matrix_key].add_(module_loras[matrix_key])
        return target

    def _micro_c2l_forward_blocks(self) -> int | None:
        micro_frames = self.micro_c2l_forward_frame
        if micro_frames is None:
            return None
        return max(1, (micro_frames + self.frame_per_lora - 1) // self.frame_per_lora)

    def _generate_temporal_sum_at_query_lora_weights(
        self,
        block_features: Tensor,
        block_attn_mask: Tensor,
        block_owners: Tensor,
        pooled_blocks: Tensor,
        num_samples: int,
    ) -> tuple[dict[str, dict[str, Tensor]], None]:
        """Sum C2L query embeddings over time before the LoRA output head.

        The output head is intentionally called once per original video, so it
        produces one A/B factor for each base layer, module, and rank rather
        than factors with a remaining temporal-block axis.
        """
        micro_blocks = self._micro_c2l_forward_blocks()
        if micro_blocks is None:
            micro_blocks = block_features.shape[0]

        temporal_aggregator = getattr(self, "temporal_aggregator", None)
        block_weights = self._raw_temporal_query_weights(
            pooled_blocks, block_owners, num_samples
        )
        if block_weights is not None and temporal_aggregator not in ("gated_residual", "history_only"):
            raise ValueError(
                "Raw pooled-cosine weighting requires temporal_aggregator=gated_residual or history_only."
            )
        if temporal_aggregator is not None:
            block_embedding_chunks = []
            for start in range(0, block_features.shape[0], micro_blocks):
                end = min(start + micro_blocks, block_features.shape[0])
                block_embedding_chunks.append(
                    self.hypernet.aggregate_lora_embeddings(
                        block_features[start:end], block_attn_mask[start:end], None
                    )
                )
            if not block_embedding_chunks:
                raise ValueError("C2L block_lora produced no temporal query embeddings.")
            temporal_embeddings = self._aggregate_temporal_lora_embeddings(
                torch.cat(block_embedding_chunks, dim=0),
                block_owners,
                num_samples,
                block_weights,
            )
            generated_loras, generated_layernorms = self.hypernet.generate_weights_from_lora_embeddings(
                temporal_embeddings
            )
            if generated_layernorms is not None:
                raise NotImplementedError("C2L block_lora does not support generated layernorm modules.")
            return generated_loras, None

        summed_embeddings = None
        normalize_add_query = bool(getattr(self, "normalize_add_query", False))
        for start in range(0, block_features.shape[0], micro_blocks):
            end = min(start + micro_blocks, block_features.shape[0])
            block_embeddings = self.hypernet.aggregate_lora_embeddings(
                block_features[start:end], block_attn_mask[start:end], None
            )
            if normalize_add_query:
                embedding_sum = self._sum_lora_embeddings_by_owner_fp32(
                    block_embeddings, block_owners[start:end], num_samples
                )
            else:
                embedding_sum = self._sum_lora_embeddings_by_owner(
                    block_embeddings, block_owners[start:end], num_samples
                )
            if summed_embeddings is None:
                summed_embeddings = embedding_sum
            else:
                summed_embeddings = summed_embeddings + embedding_sum

        if summed_embeddings is None:
            raise ValueError("C2L block_lora produced no temporal query embeddings.")

        if normalize_add_query:
            summed_embeddings = self._mean_normalize_summed_lora_embeddings(
                summed_embeddings, block_owners, num_samples
            )

        generated_loras, generated_layernorms = self.hypernet.generate_weights_from_lora_embeddings(
            summed_embeddings
        )
        if generated_layernorms is not None:
            raise NotImplementedError("C2L block_lora does not support generated layernorm modules.")
        return generated_loras, None

    def _generate_block_lora_weights(
        self,
        ctx_features: Tensor,
        ctx_attn_mask: Tensor | None,
        ctx_position_ids: Tensor | None,
        ctx_encoder_kwargs: Mapping[str, Tensor],
    ):
        if ctx_position_ids is not None:
            raise ValueError("C2L block_lora does not support explicit context position ids.")
        if not self.image_context_encoder:
            raise ValueError("C2L block_lora currently requires image/video context encoder inputs.")
        if self._has_nonempty_grid(ctx_encoder_kwargs.get("image_grid_thw")):
            raise ValueError("C2L block_lora currently supports video-only encoder contexts.")

        block_features, block_attn_mask, block_owners, pooled_blocks = self._build_video_block_features(
            ctx_features,
            ctx_attn_mask,
            ctx_encoder_kwargs.get("video_grid_thw"),
            ctx_encoder_kwargs.get("video_temporal_patch_size"),
            ctx_encoder_kwargs.get("video_metadata"),
            ctx_encoder_kwargs.get("timestamp_input_ids"),
            ctx_encoder_kwargs.get("timestamp_attention_mask"),
            ctx_encoder_kwargs.get("timestamp_block_owners"),
        )
        if getattr(self, "temporal_sum_at_query", False):
            return self._generate_temporal_sum_at_query_lora_weights(
                block_features,
                block_attn_mask,
                block_owners,
                pooled_blocks,
                ctx_features.shape[0],
            )

        micro_blocks = self._micro_c2l_forward_blocks()
        if micro_blocks is None or block_features.shape[0] <= micro_blocks:
            generated_loras, generated_layernorms = self.hypernet.generate_weights(
                block_features, block_attn_mask, None
            )
            if generated_layernorms is not None:
                raise NotImplementedError("C2L block_lora does not support generated layernorm modules.")
            summed_loras = self._sum_loras_by_owner(generated_loras, block_owners, ctx_features.shape[0])
            return self._normalize_lora_sums_by_owner(
                summed_loras, block_owners, ctx_features.shape[0]
            ), None

        summed_loras = None
        for start in range(0, block_features.shape[0], micro_blocks):
            end = min(start + micro_blocks, block_features.shape[0])
            generated_loras, generated_layernorms = self.hypernet.generate_weights(
                block_features[start:end], block_attn_mask[start:end], None
            )
            if generated_layernorms is not None:
                raise NotImplementedError("C2L block_lora does not support generated layernorm modules.")
            summed_loras = self._accumulate_lora_sums(
                summed_loras,
                self._sum_loras_by_owner(generated_loras, block_owners[start:end], ctx_features.shape[0]),
            )

        return self._normalize_lora_sums_by_owner(
            summed_loras, block_owners, ctx_features.shape[0]
        ), None

    def generate_weights(
        self,
        ctx_ids: Integer[Tensor, "bs ctx_len"] | Mapping[str, Tensor],
        ctx_attn_mask: Integer[Tensor, "bs ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs ctx_len"] | None = None,
        **kwargs: Any,
    ):
        with torch.no_grad():
            if self.image_context_encoder:
                if not isinstance(ctx_ids, Mapping):
                    raise ValueError(
                        "Image C2L context expects mapping-like encoder inputs, "
                        f"got {type(ctx_ids).__name__}."
                    )
                ctx_encoder_kwargs = {k: v for k, v in ctx_ids.items() if k not in {"labels"}}
                ctx_attn_mask = ctx_encoder_kwargs.get("attention_mask", None)
                ctx_position_ids = None
            else:
                ctx_encoder_kwargs = dict(
                    input_ids=ctx_ids,
                    attention_mask=ctx_attn_mask,
                    position_ids=ctx_position_ids,
                )

            if not self.image_context_encoder and isinstance(self.ctx_encoder.base_model, ModernBertModel):
                position_ids = ctx_position_ids.flatten()
                indices = torch.arange(
                    position_ids.size(0), device=position_ids.device, dtype=torch.int32
                )
                # [bsz + 1]
                cu_seqlens = torch.cat(
                    (
                        indices[position_ids == 0],
                        torch.tensor(
                            position_ids.size(),
                            device=position_ids.device,
                            dtype=torch.int32,
                        ),
                    )
                )
                ctx_encoder_kwargs = dict(
                    input_ids=ctx_ids.squeeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=position_ids.max() + 1,
                    attention_mask=-1,
                    seq_len=-1,
                    batch_size=-1,
                )

            if (
                self.block_lora and self.frame_per_lora == 1 and self.image_context_encoder
                and self._has_nonempty_grid(ctx_encoder_kwargs.get("video_grid_thw"))
            ):
                ctx_encoder_kwargs = self._prepare_single_frame_video_inputs(ctx_encoder_kwargs)
            ctx_features = self.ctx_encoder(**ctx_encoder_kwargs, **kwargs)
            if isinstance(ctx_features, tuple):
                ctx_features, returned_ctx_attn_mask = ctx_features
                if returned_ctx_attn_mask is not None:
                    ctx_attn_mask = returned_ctx_attn_mask
                ctx_position_ids = None
        if not self.image_context_encoder and isinstance(self.ctx_encoder.base_model, ModernBertModel):
            ctx_features = ctx_features.unsqueeze(0)

        if self.block_lora:
            lora_dict, generated_layernorms = self._generate_block_lora_weights(
                ctx_features, ctx_attn_mask, ctx_position_ids, ctx_encoder_kwargs
            )
        else:
            lora_dict, generated_layernorms = self.hypernet.generate_weights(
                ctx_features, ctx_attn_mask, ctx_position_ids
            )

        if self.user_defined_scaling == 1:
            return lora_dict, generated_layernorms

        for module in lora_dict:
            lora_dict[module]["A"] = lora_dict[module]["A"] * self.user_defined_scaling
            lora_dict[module]["B"] = lora_dict[module]["B"] * self.user_defined_scaling
        return lora_dict, generated_layernorms

    def enable_iterative_mode(self, x: bool):
        self.hypernet.enable_iterative_mode(x)


    def combine_lora(self, *args, **kwargs):
        # for timing
        return combine_lora(*args, **kwargs)

    def apply_lora_to_layers(self, *args, **kwargs):
        # for timing
        return apply_lora_to_layers(*args, **kwargs)

    # for simple api usage


    def reset(self):
        self.generated_loras = None
        layers = get_layers(self.base_model)
        if getattr(self.hypernet_config, "use_valid_slots", False):
            for slot in self.hypernet.target_slots:
                module = attrgetter(str(slot["module_path"]))(layers[int(slot["layer_idx"])])
                logger.debug(f"Resetting forward for {slot['slot_key']}")
                module.forward = module.forward_orig
                module.patched_forward = False
            return

        for layer_idx in self.hypernet.layer_indices:
            for module_info in get_lora_target_modules(layers[layer_idx], self.peft_config):
                name = module_info["name"]
                module = module_info["module"]
                logger.debug(f"Resetting forward for {name}")
                module.forward = module.forward_orig
                module.patched_forward = False

    @torch.inference_mode()
    def generate(
        self,
        ctx_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        scalers: Float[Tensor, "n_ctx"] | None = None,
        bias_scaler: float | None = None,
        *model_inputs_args: Any,
        generated_loras: dict[str, dict[str, Tensor]] | None = None,
        **model_inputs_kwargs: dict[str, Any],
    ):
        generated_layernorms = None
        if generated_loras is not None:
            if ctx_ids is not None or self.use_base_input_as_ctx:
                raise ValueError("Pass either generated_loras or context inputs, not both.")
        elif (
            ctx_ids is None
            and not self.generated_loras
            and not self.use_base_input_as_ctx
        ):
            print(
                "*" * 100
                + "\n\nNo ctx_ids provided, using the base model for generation\n\n"
                + "*" * 100
            )
        elif ctx_ids is None and self.generated_loras:
            generated_loras = self.generated_loras
            if n_ctx_chunks is None:
                n_ctx_chunks = torch.tensor((1,), device=self.device)
            print(
                "*" * 100
                + "\n\nUsing internalized LoRAs for generation\n\n"
                + "*" * 100
            )
        else:
            if self.use_base_input_as_ctx:
                ctx_ids = (
                    model_inputs_kwargs["input_ids"]
                    if "input_ids" in model_inputs_kwargs
                    else model_inputs_args[0]
                )
                ctx_attn_mask = (
                    model_inputs_kwargs["attention_mask"]
                    if "attention_mask" in model_inputs_kwargs
                    else None
                )
                ctx_position_ids = (
                    model_inputs_kwargs["position_ids"]
                    if "position_ids" in model_inputs_kwargs
                    else None
                )
            generated_loras, generated_layernorms = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids
            )

        if generated_loras is not None:
            if n_ctx_chunks is None and self.image_context_encoder:
                n_ctx_chunks = self._default_image_n_ctx_chunks(generated_loras)

            generated_loras = self.combine_lora(
                generated_loras,
                n_ctx_chunks,
                lora_bias=self.hypernet.get_head_bias()
                if self.hypernet.config.use_bias
                else None,
                scalers=scalers,
                bias_scaler=bias_scaler,
            )

            # apply lora hook to the base model
            # TODO: we dont this position_ids for generation?
            position_ids = (
                model_inputs_kwargs["position_ids"]
                if "position_ids" in model_inputs_kwargs
                else None
            )
            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        model_inputs_kwargs["input_ids"].shape[0],
                        dtype=torch.int32,
                        device=self.device,
                    )
                else:
                    # quite redundant (we do cu_seqlens many places)
                    # TODO: compute cu_seqlens here and propagate that
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            if getattr(self.hypernet_config, "use_valid_slots", False):
                apply_lora_to_valid_slots(
                    self.base_model,
                    self.hypernet.target_slots,
                    generated_loras,
                    n_queries,
                    position_ids if self.use_sequence_packing else None,
                )
            else:
                apply_lora_to_layers(
                    self.base_model,
                    self.hypernet.layer_indices,
                    generated_loras,
                    n_queries,
                    position_ids if self.use_sequence_packing else None,
                )

        model_outputs = self.base_model.generate(
            *model_inputs_args, **model_inputs_kwargs
        )
        return model_outputs


