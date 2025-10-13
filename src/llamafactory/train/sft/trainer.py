# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer
from typing_extensions import override


import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler
from torch import nn

from transformers import Seq2SeqTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from typing_extensions import override
from torch import nn

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ...extras.packages import is_transformers_version_greater_than
from ...monkeypatch.loss.chunked import patch_chunked_ce_loss_fn, patch_loss_functions as unsolth_patch_loss_functions
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler
from ..fp8_utils import configure_fp8_environment, verify_fp8_status

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments, ModelArguments


logger = logging.get_logger(__name__)

# One-time backend logging switches for CCE
_cce_backend_logged: int = 0  # 0=unset, 1=cce_backend, 2=fallback

if 0:
    patch_chunked_ce_loss_fn(num_output_chunks = 4)
    logger.info_rank0('Patch CE Loss -> chunked CE Loss to reduce vram')
else:
    unsolth_patch_loss_functions()
    logger.info_rank0('Patch CE Loss -> Unsloth chunked CE Loss to reduce vram')

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        model,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        model_args: Optional["ModelArguments"] = None,
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        # Configure FP8 environment if enabled
        if model_args is not None and model_args.fp8:
            configure_fp8_environment(model_args)
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")
        else:
            self.processing_class: PreTrainedTokenizer = kwargs.get("tokenizer")
        # model._loss_function = PatchForCausalLMLoss

        super().__init__(model, **kwargs)
        print('CHECK in CustomSeq2SeqTrainer', model.sequence_parallel_group)
        if processor is not None:
            # avoid wrong loss under gradient accumulation
            # https://github.com/huggingface/transformers/pull/36044#issuecomment-2746657112
            self.model_accepts_loss_kwargs = False

        self.finetuning_args = finetuning_args
        if gen_kwargs is not None:
            # https://github.com/huggingface/transformers/blob/v4.45.0/src/transformers/trainer_seq2seq.py#L287
            self._gen_kwargs = gen_kwargs

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

        if finetuning_args.use_dft_loss:
            from ..trainer_utils import dft_loss_func

            self.compute_loss_func = dft_loss_func

        # Verify FP8 status after trainer initialization (accelerator should be available)
        if model_args is not None and model_args.fp8 and hasattr(self, "accelerator"):
            verify_fp8_status(self.accelerator, model_args)
        # Optional Cut Cross-Entropy support for memory-efficient loss
        if finetuning_args.use_cce:
            from ...extras.packages import is_cce_available

            if not is_cce_available():
                raise ImportError(
                    "Cut Cross-Entropy is not available. Install with: "
                    "pip install 'cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git'"
                )
            # from ..trainer_utils import cce_loss_func

            self.compute_loss_func = self.cce_loss_func
            logger.info_rank0("Cut Cross-Entropy enabled: using memory-efficient loss on standard SFT path.")


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
    def _get_train_sampler(self, *args, **kwargs) -> Optional["torch.utils.data.Sampler"]:

        if self.model.sequence_parallel_group is not None:
            print('Get SP in _get_train_sampler')
            return SequentialSampler(self.train_dataset)
        else:
            if self.finetuning_args.disable_shuffling:
                return torch.utils.data.SequentialSampler(self.train_dataset)
            return super()._get_train_sampler(*args, **kwargs)

    def cce_loss_func(self, outputs, labels, num_items_in_batch=None):
        """Compute loss using Cut Cross-Entropy if available; otherwise fallback to standard CE.

        This mirrors the interface of `dft_loss_func` so it can be plugged into
        the trainer via `self.compute_loss_func`.
        """
        logits = outputs.get("logits")
        if logits is None:
            # Some models may only return loss
            return outputs.get("loss", torch.tensor(0.0))

        # Shift labels to align with next-token prediction
        labels = torch.nn.functional.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()

        # Prefer the memory-efficient op when the optional package is present
        global _cce_backend_logged
        try:
            from cut_cross_entropy import linear_cross_entropy  # type: ignore

            # Use logits and lm_head weight when possible to avoid materializing full probs
            # We expect models to expose last hidden states through outputs.get("hidden_states")
            # only when configured; otherwise we can still use logits fallback.
            embeddings = outputs.get("hidden_states", None)
            classifier = None

            # Try to find the classifier (lm_head) weight on the model if attached
            model = outputs.get("model", None)
            # if model is not None:
            #     for name, module in model.named_modules():
            #         if "lm_head" in name and hasattr(module, "weight"):
            #             classifier = module.weight
            #             break
            classifier = model.lm_head.weight

            if embeddings is not None and classifier is not None:
                if isinstance(embeddings, (list, tuple)):
                    embeddings = embeddings[-1]

                # Remove the last timestep to align with shift_labels
                hidden = embeddings[:, :-1]
                loss = linear_cross_entropy(
                    hidden,
                    classifier,
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="mean",
                )
                if _cce_backend_logged == 0:
                    logger.info_rank0("CCE backend active: using cut_cross_entropy.linear_cross_entropy")
                    _cce_backend_logged = 1
                if num_items_in_batch is not None:
                    # Keep parity with dft_loss_func optional normalization path
                    loss = loss * shift_labels.numel() / num_items_in_batch
                return loss
        except Exception:
            # Package missing or model introspection failed; fall back below
            if _cce_backend_logged == 0:
                logger.info_rank0(
                    "CCE fallback: standard torch.nn.functional.cross_entropy will be used ("
                    "package missing or model lacks hidden_states/lm_head)."
                )
                _cce_backend_logged = 2

        # Fallback: standard cross-entropy over flattened logits
        logits = logits.float()
        vocab_size = logits.size(-1)
        logits = logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1).to(logits.device)

        loss = torch.nn.functional.cross_entropy(logits, shift_labels, ignore_index=-100, reduction="mean")
        if num_items_in_batch is not None:
            loss = loss * shift_labels.numel() / num_items_in_batch
        return loss

    def compute_loss_cce(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, Union[torch.Tensor, Any]]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If num_items_in_batch is not passed,

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculationg might be slightly inacurate when performing gradient accumulation.
        """
        if (self.label_smoother is not None or self.compute_loss_func is not None) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs, output_hidden_states=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(outputs, labels, num_items_in_batch=num_items_in_batch)
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss

    @override
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        # print('CHECK in compute_loss', self.model.sequence_parallel_group)
        if self.model.sequence_parallel_group is None:  # no sequence parallel, compute as it is
            # print(inputs)
            # print(super().compute_loss(model, inputs, return_outputs=True, **kwargs))
            return super().compute_loss(model, inputs, **kwargs)
        else:
            # print(list(inputs.keys()))
            # print('atten_mask', inputs['attention_mask'].shape, inputs['attention_mask'].dtype)
            # print('input_ids', inputs['input_ids'].shape, inputs['input_ids'].dtype)
            # print(dir(inputs))
            # compute loss without shift labels, as we have already shifted labels in data processing when using sequence parallel
            labels = inputs["labels"]
            if self.finetuning_args.use_cce:
                _, outputs = self.compute_loss_cce(model, inputs, return_outputs=True, **kwargs)
            else:
                _, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="sum")
            # print(outputs)
            # logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
            # Get vocab_size
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                vocab_size = unwrapped_model.base_model.model.config.vocab_size
            else:
                vocab_size = unwrapped_model.config.vocab_size
            logits = logits.view(-1, vocab_size)
            labels = labels.view(-1)
            # Enable model parallelism
            labels = labels.to(logits.device)
            loss = loss_fct(logits, labels)
            # loss = fast_cross_entropy_loss(logits, labels)

            # weighted reduce within sequence_parallel_group
            sp_group = self.model.sequence_parallel_group
            loss = dist.nn.all_reduce(loss, op=dist.ReduceOp.SUM, group=sp_group)
            label_num = (labels != loss_fct.ignore_index).sum()
            label_num = dist.nn.all_reduce(label_num, op=dist.ReduceOp.SUM, group=sp_group)
            loss /= label_num

        # now is single-sequence loss
        # print('loss', loss.shape, loss)

        if is_transformers_version_equal_to_4_46() and not getattr(self, "model_accepts_loss_kwargs", False):
            # other model should not scale the loss
            if return_outputs:
                return (loss[0] / self.args.gradient_accumulation_steps, *loss[1:])
            else:
                return loss / self.args.gradient_accumulation_steps

        return loss

    @override
    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: dict[str, Union["torch.Tensor", Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""Remove the prompt part in the generated tokens.

        Subclass and override to inject custom behavior.
        """
        if self.args.predict_with_generate:  # do not pass labels to model when generate
            labels = inputs.pop("labels", None)
        else:
            labels = inputs.get("labels")

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys, **gen_kwargs
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, : inputs["input_ids"].size(-1)] = self.processing_class.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def save_predictions(
        self, dataset: "Dataset", predict_results: "PredictionOutput", skip_special_tokens: bool = True
    ) -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.processing_class.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX,
            predict_results.predictions,
            self.processing_class.pad_token_id,
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.processing_class.pad_token_id)[0]
            if len(pad_len):  # move pad token to last
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)
        # https://github.com/danikhan632/ring-flash-attention.git
        decoded_inputs = self.processing_class.batch_decode(dataset["input_ids"], skip_special_tokens=False)
        decoded_preds = self.processing_class.batch_decode(preds, skip_special_tokens=skip_special_tokens)
        decoded_labels = self.processing_class.batch_decode(labels, skip_special_tokens=skip_special_tokens)

        with open(output_prediction_file, "w", encoding="utf-8") as f:
            for text, pred, label in zip(decoded_inputs, decoded_preds, decoded_labels):
                f.write(json.dumps({"prompt": text, "predict": pred, "label": label}, ensure_ascii=False) + "\n")
