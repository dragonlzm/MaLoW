from typing import TYPE_CHECKING, Optional

from peft import LoraConfig, TaskType

from malow.training._vendor.ctx_to_lora.configs import AggregatorArguments, CtxEncoderArguments, HypernetArguments
from malow.training._vendor.ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel, get_hypernet_config

from ...data import SFTDataCollatorWith4DAttentionMask, get_template_and_fix_tokenizer
from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ...extras.misc import calculate_tps
from ...extras.ploting import plot_loss
from ...model import load_model, load_tokenizer
from ...model.model_utils.misc import find_all_linear_modules
from ..trainer_utils import create_modelcard_and_push
from .data import C2LDataCollator, get_c2l_dataset
from .manifest import C2LTrainingManifestCallback
from .trainer import C2LTrainer
from .vision_encoder import (
    load_c2l_vision_encoder,
    load_c2l_vjepa2_encoder,
    load_c2l_vjepa2_video_processor,
)


if TYPE_CHECKING:
    from transformers import Seq2SeqTrainingArguments, TrainerCallback

    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments


logger = get_logger(__name__)


SUPPORTED_C2L_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "qkv_proj",
    "down_proj",
    "up_proj",
    "gate_proj",
}

SUPPORTED_C2L_VALID_SLOT_TARGET_MODULES = SUPPORTED_C2L_TARGET_MODULES | {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "linear_attn.in_proj_qkv",
    "in_proj_qkv",
}


def _configure_base_last_2seconds_processor_fps(
    processor, finetuning_args: "FinetuningArguments"
) -> None:
    if not finetuning_args.base_with_last_2seconds:
        return

    processor.video_fps = 1.0
    logger.info_rank0("C2L base-last-two-seconds reports predecoded video inputs as 1 FPS metadata.")


def _add_c2l_training_manifest_callback(
    callbacks: Optional[list["TrainerCallback"]],
    *,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
) -> list["TrainerCallback"]:
    c2l_callbacks = list(callbacks or [])
    c2l_callbacks.append(
        C2LTrainingManifestCallback(
            model_args=model_args,
            data_args=data_args,
            finetuning_args=finetuning_args,
            generating_args=generating_args,
        )
    )
    return c2l_callbacks


def _get_text_config(model):
    config = model.config
    return getattr(config, "text_config", config)


def _patch_hidden_size_for_hypernet(model) -> None:
    text_config = _get_text_config(model)
    if not hasattr(model.config, "hidden_size") and hasattr(text_config, "hidden_size"):
        setattr(model.config, "hidden_size", text_config.hidden_size)


def _load_c2l_encoder_model(tokenizer, model_args, finetuning_args, base_model_config):
    if finetuning_args.c2l_ctx_encoder_type == "vjepa2_vision_embedder":
        load_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
            "dtype": getattr(model_args, "compute_dtype", "auto"),
            "low_cpu_mem_usage": model_args.low_cpu_mem_usage,
        }
        if model_args.device_map is not None:
            load_kwargs["device_map"] = model_args.device_map
        loaded = load_c2l_vjepa2_encoder(
            finetuning_args.c2l_vjepa2_model_name_or_path,
            **load_kwargs,
        )
        return loaded.model, loaded.feature_config

    if finetuning_args.c2l_vision_encoder_only:
        load_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
            "dtype": "auto",
            "low_cpu_mem_usage": model_args.low_cpu_mem_usage,
        }
        if model_args.device_map is not None:
            load_kwargs["device_map"] = model_args.device_map
        loaded = load_c2l_vision_encoder(
            model_args.model_name_or_path,
            base_model_config,
            **load_kwargs,
        )
        return loaded.model, loaded.feature_config

    encoder_model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    _patch_hidden_size_for_hypernet(encoder_model)
    return encoder_model, _get_text_config(encoder_model)


def _select_c2l_collator_model(base_model, encoder_model, use_base_model: bool):
    return base_model if use_base_model else encoder_model


def _build_c2l_lora_config(model, finetuning_args: "FinetuningArguments") -> LoraConfig:
    if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
        target_modules = [
            module
            for module in find_all_linear_modules(model, finetuning_args.freeze_vision_tower)
            if module in SUPPORTED_C2L_TARGET_MODULES
        ]
    else:
        target_modules = finetuning_args.lora_target

    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=finetuning_args.lora_rank,
        target_modules=target_modules,
        lora_alpha=finetuning_args.lora_alpha,
        lora_dropout=finetuning_args.lora_dropout,
        use_rslora=finetuning_args.use_rslora,
        use_dora=finetuning_args.use_dora,
        modules_to_save=finetuning_args.additional_target,
    )


def _validate_target_modules(lora_config: LoraConfig, use_valid_slots: bool = False) -> None:
    supported = SUPPORTED_C2L_VALID_SLOT_TARGET_MODULES if use_valid_slots else SUPPORTED_C2L_TARGET_MODULES
    unsupported = set(lora_config.target_modules) - supported
    if unsupported:
        raise ValueError(
            "C2L generated LoRA currently supports target modules "
            f"{sorted(supported)}, but got unsupported targets {sorted(unsupported)}."
        )
    if not use_valid_slots and any("." in target or target == "in_proj_qkv" for target in lora_config.target_modules):
        raise ValueError(
            "Dotted or linear-attention C2L targets require c2l_use_valid_slots=true."
        )


def _log_trainable_parameters(model) -> None:
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    total_trainable = sum(param.numel() for _, param in trainable)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_pct = 100 * total_trainable / total_params if total_params else 0.0
    logger.info_rank0(
        "C2L trainable parameters: "
        f"{len(trainable)} tensors, {total_trainable:,} / {total_params:,} scalars "
        f"({trainable_pct:.4f}%)."
    )
    for name, param in trainable:
        logger.info_rank0(f"  {name}: shape={tuple(param.shape)}, dtype={param.dtype}")


def run_c2l(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[list["TrainerCallback"]] = None,
):
    tokenizer_module = load_tokenizer(model_args)
    tokenizer = tokenizer_module["tokenizer"]
    processor = tokenizer_module["processor"]
    if processor is None:
        raise ValueError("C2L image context training requires a multimodal processor.")
    _configure_base_last_2seconds_processor_fps(processor, finetuning_args)

    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    logger.info_rank0("Loading frozen C2L base model for generated LoRA targets.")
    base_model = load_model(tokenizer, model_args, finetuning_args, is_trainable=False)
    _patch_hidden_size_for_hypernet(base_model)
    lora_config = _build_c2l_lora_config(base_model, finetuning_args)
    _validate_target_modules(lora_config, finetuning_args.c2l_use_valid_slots)
    base_model.requires_grad_(False)
    base_model.name_or_path = model_args.model_name_or_path

    if finetuning_args.c2l_ctx_encoder_type == "vjepa2_vision_embedder":
        logger.info_rank0("Loading frozen V-JEPA2 C2L context encoder model.")
    elif finetuning_args.c2l_vision_encoder_only:
        logger.info_rank0("Loading frozen C2L vision-only encoder model.")
    else:
        logger.info_rank0("Loading frozen C2L image encoder model.")
    encoder_model, encoder_feature_config = _load_c2l_encoder_model(
        tokenizer,
        model_args,
        finetuning_args,
        base_model.config,
    )
    encoder_model.requires_grad_(False)
    encoder_model.eval()

    hypernet_args = HypernetArguments(
        latent_size=finetuning_args.c2l_hypernet_latent_size,
        per_rank_gen=finetuning_args.c2l_per_rank_gen,
        head_per_layer=finetuning_args.c2l_hypernet_head_per_layer,
        head_per_module=finetuning_args.c2l_hypernet_head_per_module,
        head_per_rank=finetuning_args.c2l_hypernet_head_per_rank,
        use_valid_slots=finetuning_args.c2l_use_valid_slots,
        temporal_deltanet_num_layers=finetuning_args.temporal_deltanet_num_layers,
    )
    aggregator_args = AggregatorArguments(
        aggregator_type=finetuning_args.c2l_aggregator_type,
        encoder_aggregator_num_layers=finetuning_args.c2l_encoder_aggregator_num_layers,
        encoder_aggregator_d_model=finetuning_args.c2l_encoder_aggregator_d_model,
        layer_specific_queries=finetuning_args.c2l_encoder_aggregator_layer_specific_queries,
        query_per_layer=(
            finetuning_args.c2l_encoder_aggregator_query_per_layer
            or finetuning_args.c2l_encoder_aggregator_layer_specific_queries
        ),
        query_per_module=finetuning_args.c2l_encoder_aggregator_query_per_module,
        query_per_rank=finetuning_args.c2l_encoder_aggregator_query_per_rank,
    )
    ctx_encoder_args = CtxEncoderArguments(
        ctx_encoder_model_name_or_path=(
            finetuning_args.c2l_vjepa2_model_name_or_path
            if finetuning_args.c2l_ctx_encoder_type == "vjepa2_vision_embedder"
            else model_args.model_name_or_path
        ),
        ctx_encoder_type=finetuning_args.c2l_ctx_encoder_type,
        quantize_ctx_encoder=finetuning_args.c2l_quantize_ctx_encoder,
        vjepa2_window_size=finetuning_args.c2l_vjepa2_window_size,
        vjepa2_window_stride=finetuning_args.c2l_vjepa2_window_stride,
        vjepa2_spatial_pool_size=finetuning_args.c2l_vjepa2_spatial_pool_size,
        vjepa2_micro_batch_size=finetuning_args.c2l_vjepa2_micro_batch_size,
    )
    # import ipdb; ipdb.set_trace() # check ctx_encoder_args, hypernet_config, lora_config
    hypernet_config = get_hypernet_config(
        base_model,
        encoder_feature_config,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
        lora_config=lora_config,
    )
    model = ModulatedPretrainedModel(
        base_model,
        hypernet_config,
        ctx_encoder_args,
        encoder_model=encoder_model,
        image_context_encoder=True,
        use_sequence_packing=False,
        block_lora=finetuning_args.block_lora,
        frame_per_lora=finetuning_args.frame_per_lora,
        normalize_add_lora=finetuning_args.normalize_add_lora,
        interleave_timestamp=finetuning_args.interleave_timestamp,
        micro_c2l_forward_frame=finetuning_args.micro_c2l_forward_frame,
        temporal_sum_at_query=finetuning_args.temporal_sum_at_query,
        normalize_add_query=finetuning_args.normalize_add_query,
        temporal_aggregator=finetuning_args.temporal_aggregator,
        temporal_query_weighting=finetuning_args.temporal_query_weighting,
        temporal_query_gate_threshold=finetuning_args.temporal_query_gate_threshold,
    )
    model.base_input_video = finetuning_args.base_input_video
    model.base_input_frame_number = finetuning_args.base_input_frame_number
    model.base_with_last_2seconds = finetuning_args.base_with_last_2seconds
    model.base_tail_seconds = finetuning_args.base_tail_seconds
    model.base_last_2seconds_max_frames = finetuning_args.base_last_2seconds_max_frames
    model.video_merge_extend = finetuning_args.video_merge_extend
    model.video_merge_extend_perc = finetuning_args.video_merge_extend_perc
    model.train()
    _log_trainable_parameters(model)

    if data_args.replay_manifest:
        from malow.training.replay import training_dataset

        train_dataset = training_dataset(
            template, tokenizer, processor, data_args, finetuning_args, training_args
        )
    else:
        train_dataset = get_c2l_dataset(
            template,
            tokenizer,
            processor,
            model_args,
            data_args,
            finetuning_args,
            data_args.dataset,
            training_args.seed,
            repeat_streaming=finetuning_args.c2l_streaming_repeat,
        )
    eval_dataset = get_c2l_dataset(
        template,
        tokenizer,
        processor,
        model_args,
        data_args,
        finetuning_args,
        data_args.eval_dataset,
        training_args.seed,
    )

    collator_model = _select_c2l_collator_model(
        base_model,
        encoder_model,
        finetuning_args.c2l_vision_encoder_only
        or finetuning_args.c2l_ctx_encoder_type == "vjepa2_vision_embedder",
    )
    encoder_collator = SFTDataCollatorWith4DAttentionMask(
        template=template,
        model=collator_model,
        processor=processor,
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.do_train else None,
        label_pad_token_id=IGNORE_INDEX,
        block_diag_attn=model_args.block_diag_attn,
        neat_packing=False,
        attn_implementation=getattr(collator_model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
    )
    data_collator = C2LDataCollator(
        tokenizer=tokenizer,
        encoder_collator=encoder_collator,
        interleave_timestamp=finetuning_args.interleave_timestamp,
        frame_per_lora=finetuning_args.frame_per_lora,
        loss_type=finetuning_args.c2l_loss_type,
        vjepa2_video_processor=(
            load_c2l_vjepa2_video_processor(
                finetuning_args.c2l_vjepa2_model_name_or_path,
                local_files_only=True,
            )
            if finetuning_args.c2l_ctx_encoder_type == "vjepa2_vision_embedder"
            else None
        ),
        vjepa2_model_config=(
            encoder_model.config
            if finetuning_args.c2l_ctx_encoder_type == "vjepa2_vision_embedder"
            else None
        ),
        vjepa2_window_size=finetuning_args.c2l_vjepa2_window_size,
        vjepa2_window_stride=finetuning_args.c2l_vjepa2_window_stride,
        vjepa2_spatial_pool_size=finetuning_args.c2l_vjepa2_spatial_pool_size,
        vjepa2_input_dtype=getattr(model_args, "compute_dtype", None),
    )

    callbacks = _add_c2l_training_manifest_callback(
        callbacks,
        model_args=model_args,
        data_args=data_args,
        finetuning_args=finetuning_args,
        generating_args=generating_args,
    )
    trainer = C2LTrainer(
        model=model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        tokenizer=tokenizer,
        processor=processor,
    )

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        if finetuning_args.include_effective_tokens_per_second:
            train_result.metrics["effective_tokens_per_sec"] = calculate_tps(
                train_dataset, train_result.metrics, stage="sft"
            )

        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        if trainer.is_world_process_zero() and finetuning_args.plot_loss:
            metric_name = "kl_loss" if finetuning_args.c2l_loss_type == "kl" else "sft_loss"
            plot_loss(training_args.output_dir, keys=["loss", metric_name, "gen_lora_l1_norm"])

    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    create_modelcard_and_push(trainer, model_args, data_args, training_args, finetuning_args)
