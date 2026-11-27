import os
from collections import defaultdict
from types import MethodType
from typing import TYPE_CHECKING, Optional

import torch
from transformers import Trainer
from transformers.trainer_utils import IntervalStrategy
from typing_extensions import override

from ..callbacks import SaveProcessorCallback
from ..fp8_utils import configure_fp8_environment, patch_accelerator_for_fp8, verify_fp8_status
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


# MaLoW checkpoints use the existing public non-pickle metadata format.
C2L_METADATA_NAME = "c2l_metadata.json"
C2L_METADATA_KEYS = {"base_model_name_or_path", "hypernet_config", "ctx_encoder_args"}
C2L_OPTIONAL_METADATA_KEYS = {
    "block_lora",
    "frame_per_lora",
    "normalize_add_lora",
    "interleave_timestamp",
    "micro_c2l_forward_frame",
    "temporal_sum_at_query",
    "normalize_add_query",
    "temporal_aggregator",
    "temporal_query_weighting",
    "temporal_query_gate_threshold",
    "base_input_video",
    "base_input_frame_number",
    "base_with_last_2seconds",
    "base_tail_seconds",
    "base_last_2seconds_max_frames",
    "video_merge_extend",
    "video_merge_extend_perc",
}


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from ...hparams import FinetuningArguments, TrainingArguments


def _mean_token_losses_by_sample(
    token_losses: torch.Tensor,
    sample_ids: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Average token losses within each sample, then average across samples.

    A plain token mean weights every sample by its supervised length, so long answers
    dominate the gradient and the effective dataset mixture drifts away from
    `interleave_probs`. Averaging per sample first restores equal sample weight.
    """
    sample_ids = sample_ids.to(device=token_losses.device, dtype=torch.long)
    sums = token_losses.new_zeros(num_samples).index_add(0, sample_ids, token_losses)
    counts = torch.zeros(num_samples, device=token_losses.device, dtype=token_losses.dtype).index_add(
        0, sample_ids, torch.ones_like(token_losses)
    )
    populated = counts > 0
    if not bool(populated.any()):
        # Keep the graph alive rather than returning a detached zero.
        return token_losses.sum()

    return (sums[populated] / counts[populated]).mean()


def compute_sparse_kl_loss(
    token_logits: torch.Tensor,
    target_indices: torch.Tensor,
    target_logp: torch.Tensor,
    sample_ids: Optional[torch.Tensor] = None,
    num_samples: Optional[int] = None,
) -> torch.Tensor:
    logq_denom = torch.logsumexp(token_logits.float(), dim=-1, keepdim=True)
    selected_logits = token_logits.float().gather(1, target_indices)
    logq_selected = selected_logits - logq_denom
    token_losses = -(target_logp.exp() * logq_selected).sum(dim=-1)
    if sample_ids is None or num_samples is None:
        return token_losses.mean()

    return _mean_token_losses_by_sample(token_losses, sample_ids, num_samples)


def compute_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight_by_sample: bool = False,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].float().contiguous()
    shift_labels = labels[..., 1:].to(shift_logits.device).contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
    flat_labels = shift_labels.view(-1)
    if not weight_by_sample:
        return torch.nn.functional.cross_entropy(flat_logits, flat_labels, ignore_index=-100)

    # `ignore_index` already zeroes the ignored positions, so they drop out of both the
    # per-sample sums and the per-sample counts below.
    token_losses = torch.nn.functional.cross_entropy(
        flat_logits, flat_labels, ignore_index=-100, reduction="none"
    )
    supervised = flat_labels != -100
    sample_ids = torch.arange(
        shift_labels.shape[0], device=shift_labels.device
    ).repeat_interleave(shift_labels.shape[1])
    return _mean_token_losses_by_sample(
        token_losses[supervised], sample_ids[supervised], shift_labels.shape[0]
    )


class C2LTrainer(Trainer):
    def __init__(
        self,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        **kwargs,
    ) -> None:
        kwargs["processing_class"] = kwargs.pop("tokenizer")
        training_args: TrainingArguments = kwargs.get("args")
        if training_args.fp8:
            configure_fp8_environment(training_args)
            if getattr(training_args, "fp8_backend", "auto") == "te":
                patch_accelerator_for_fp8()

        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.loss_type = finetuning_args.c2l_loss_type
        self.weight_by_sample = finetuning_args.weight_by_sample
        self.gen_lora_l1_reg_coef = finetuning_args.c2l_gen_lora_l1_reg_coef
        self.strict_target_alignment = finetuning_args.c2l_strict_target_alignment
        self.model_accepts_loss_kwargs = False

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if training_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, training_args)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if state_dict is None:
            state_dict = self.model.state_dict()

        tensor_state_dict = {}
        metadata = {}
        for name, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                tensor_state_dict[name] = value
            else:
                metadata[name] = value

        missing_metadata = C2L_METADATA_KEYS.difference(metadata)
        if missing_metadata:
            model = self.model
            if hasattr(self, "accelerator"):
                model = self.accelerator.unwrap_model(model)

            if "base_model_name_or_path" in missing_metadata and hasattr(model, "base_model"):
                base_model_name = getattr(model.base_model, "name_or_path", None)
                if base_model_name is not None:
                    metadata["base_model_name_or_path"] = base_model_name
            if "hypernet_config" in missing_metadata and hasattr(model, "hypernet_config"):
                metadata["hypernet_config"] = model.hypernet_config
            if "ctx_encoder_args" in missing_metadata and hasattr(model, "ctx_encoder_args"):
                metadata["ctx_encoder_args"] = model.ctx_encoder_args

        missing_optional_metadata = C2L_OPTIONAL_METADATA_KEYS.difference(metadata)
        if missing_optional_metadata:
            model = self.model
            if hasattr(self, "accelerator"):
                model = self.accelerator.unwrap_model(model)
            for key in sorted(missing_optional_metadata):
                if hasattr(model, key):
                    metadata[key] = getattr(model, key)
                elif hasattr(self.finetuning_args, key):
                    metadata[key] = getattr(self.finetuning_args, key)

        if getattr(self.args, "save_safetensors", True):
            ptrs = defaultdict(list)
            for name, tensor in tensor_state_dict.items():
                ptrs[id(tensor)].append(name)

            for names in ptrs.values():
                if len(names) > 1:
                    names.sort()
                    for name in names[1:]:
                        tensor_state_dict.pop(name, None)

        super()._save(output_dir, tensor_state_dict)

        if metadata and self.args.should_save:
            output_dir = output_dir or self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            # MaLoW adaptation: public JSON metadata, never executable pickle.
            from malow.training.checkpointing import save_training_checkpoint_metadata

            save_training_checkpoint_metadata(metadata, output_dir, self.model)

    @override
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # MaLoW adaptation: tensor-only restore preserves optimizer Parameters.
        from malow.training.checkpointing import load_training_weights

        load_training_weights(resume_from_checkpoint, self.model if model is None else model)

    @override
    def _get_train_sampler(self, *args, **kwargs):
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler(*args, **kwargs)

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if self.loss_type == "kl":
            target_logp = inputs.pop("logprobs_vals")
            target_indices = inputs.pop("logprobs_indices")
            target_mask = inputs.pop("logprobs_mask")

            label_pos = torch.where(labels != -100)
            target_logp = target_logp[target_mask].to(labels.device)
            target_indices = target_indices[target_mask].to(labels.device)
            if self.strict_target_alignment and label_pos[0].shape[0] != target_logp.shape[0]:
                raise ValueError(
                    "C2L label/KL target mismatch: "
                    f"labels={label_pos[0].shape[0]}, targets={target_logp.shape[0]}."
                )

        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)
        if self.loss_type == "kl":
            token_logits = outputs.logits[label_pos[0], label_pos[1] - 1].float()
            target_sample_ids = label_pos[0]
            if not self.strict_target_alignment and token_logits.shape[0] != target_logp.shape[0]:
                n_tokens = min(token_logits.shape[0], target_logp.shape[0])
                token_logits = token_logits[:n_tokens]
                target_logp = target_logp[:n_tokens]
                target_indices = target_indices[:n_tokens]
                target_sample_ids = target_sample_ids[:n_tokens]

            primary_loss = compute_sparse_kl_loss(
                token_logits,
                target_indices,
                target_logp,
                sample_ids=target_sample_ids if self.weight_by_sample else None,
                num_samples=labels.shape[0] if self.weight_by_sample else None,
            )
            metric_name = "kl_loss"
        else:
            primary_loss = compute_sft_loss(outputs.logits, labels, self.weight_by_sample)
            metric_name = "sft_loss"

        l1_norm = torch.zeros((), device=primary_loss.device)
        if gen_loras:
            for lora in gen_loras.values():
                l1_norm = l1_norm + lora["A"].abs().mean() + lora["B"].abs().mean()
            l1_norm = l1_norm / len(gen_loras)

        total_loss = primary_loss + self.gen_lora_l1_reg_coef * l1_norm

        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.state.logging_steps == 0
        ):
            self.log({metric_name: primary_loss.detach().item(), "gen_lora_l1_norm": l1_norm.detach().item()})

        return (total_loss, outputs) if return_outputs else total_loss
