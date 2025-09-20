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
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import SequentialSampler


# import triton
# import triton.language as tl
# import torch
# from torch import nn

# from unsloth.kernels.utils import (
#     calculate_settings,
#     MAX_FUSED_SIZE,
#     triton_tanh,
#     triton_cast,
#     torch_gpu_device,
# )
# from transformers.models.llama.modeling_llama import logger
# from packaging.version import Version

# from unsloth_zoo.loss_utils import (
#     patch_loss_functions as _patch_loss_functions,
#     post_patch_loss_function,
# )

# def _cross_entropy_forward(
#     logits_ptr        ,
#     logits_row_stride ,
#     loss_ptr          ,
#     logsumexp_ptr     ,
#     labels_ptr        ,
#     VOCAB_SIZE        : tl.constexpr,
#     BLOCK_SIZE        : tl.constexpr,
#     DO_SOFTCAPPING    : tl.constexpr,
#     SOFTCAP           : tl.constexpr,
#     DO_LOGIT_SCALING  : tl.constexpr,
#     LOGIT_SCALE       : tl.constexpr,
# ):
#     """
#         Cross Entropy Loss = 1/n sum [ -yi log(Pi) ]
#         Pi = exp(xi) / sum(exp(xi))
#         CE_i = -y log(p) = -y log[ exp(x) / sum(exp(x)) ]
#              = -y [ x - log[sum(exp(x))] ]
#              = y * (log[sum(exp(x))] - x)
#         If y == 0: CE_i = 0
#         If y == 1: CE_i = logsumexp - x

#         logsumexp is also stable
#         Take    y =         log[sum(exp(x))]
#            exp(y) =             sum(exp(x))
#            exp(y) =             sum(exp(x - c)*exp(c)) Since e^(x-c)*e^c = e^x
#            exp(y) =      exp(c)*sum(exp(x - c))
#                y  = log(exp(c)*sum(exp(x - c)))
#                y  = c + log[sum(exp(x - c))]
#         This means we can set c = max(x) to make sure
#         exp(x - c) always is exp(x - max(x)).
#         This ensures exp(x - max(x))'s maximum is 1 as exp(0) = 1.
#     """
#     row_idx = tl.program_id(0)
#     logits_ptr    += row_idx * triton_cast(logits_row_stride, tl.int64)
#     loss_ptr      += row_idx
#     logsumexp_ptr += row_idx
#     labels_ptr    += row_idx

#     col_offsets = tl.arange(0, BLOCK_SIZE)
#     mask = col_offsets < VOCAB_SIZE

#     label_idx = tl.load(labels_ptr).to(tl.int32)
#     logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

#     # Go logit scaling for Cohere: t * x
#     if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
#     # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
#     if DO_SOFTCAPPING:   logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

#     c = tl.max(logits, 0)
#     logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

#     if label_idx != -100:
#         x = tl.load(logits_ptr + label_idx).to(tl.float32)
#         # Go logit scaling for Cohere: t * x
#         if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
#         # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
#         if DO_SOFTCAPPING:   x = SOFTCAP * triton_tanh(x / SOFTCAP)
#         loss = logsumexp - x
#     else:
#         loss = 0.0
#     tl.store(logsumexp_ptr, logsumexp)
#     tl.store(loss_ptr, loss)
# pass
# _cross_entropy_forward = triton.jit(_cross_entropy_forward)
# _cross_entropy_forward = triton.heuristics(
#     {
#         "DO_SOFTCAPPING":   lambda args: bool(args["DO_SOFTCAPPING"  ]),
#         "DO_LOGIT_SCALING": lambda args: bool(args["DO_LOGIT_SCALING"]),
#     }
# )(_cross_entropy_forward)


# def _chunked_cross_entropy_forward(
#     logits_ptr        ,
#     logits_row_stride : tl.constexpr,
#     loss_ptr          ,
#     logsumexp_ptr     ,
#     labels_ptr        ,
#     VOCAB_SIZE        : tl.constexpr,
#     N_CHUNKS          : tl.constexpr,
#     BLOCK_SIZE        : tl.constexpr,
#     DO_SOFTCAPPING    : tl.constexpr,
#     SOFTCAP           : tl.constexpr,
#     DO_LOGIT_SCALING  : tl.constexpr,
#     LOGIT_SCALE       : tl.constexpr,
# ):
#     """
#         256K vocab divided in 4 chunks

#         |-65536-| |-65536-| |-65536-| |-65536-|
#         |-------| |-------| |-------| |-------|
#         |-------| |-------| |-------| |-------|

#         If y == 0: CE_i = 0
#         If y == 1: CE_i = logsumexp - x

#         Notice we can do logsumexp for each chunk and then
#         logsumexp[chunk_sum(logsumexp)] == logsumexp

#         chunk_sum = log[chunk_sum(logsumexp)]
#                   = log[exp(logsumexp(a)) + ... + exp(logsumexp(z))]
#                   = log[exp(log[sum(exp(a))]) + ... + exp(log[sum(exp(z))])]
#                   = log[sum(exp(a)) + ... + sum(exp(z))]
#                   = logsumexp(x)

#         This means we can perform a logsumexp for each chunk, then do a
#         final logsumexp reduction!

#         Ie do: logsumexp(chunked_logsumexp) - x
#     """
#     row_idx   = tl.program_id(0)
#     chunk_idx = tl.program_id(1)
#     logits_ptr    += row_idx * triton_cast(logits_row_stride, tl.int64)
#     loss_ptr      += row_idx
#     logsumexp_ptr += row_idx * N_CHUNKS + chunk_idx
#     labels_ptr    += row_idx

#     col_offsets = chunk_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = col_offsets < VOCAB_SIZE

#     label_idx = tl.load(labels_ptr).to(tl.int32)
#     logits = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

#     # Go logit scaling for Cohere: t * x
#     if DO_LOGIT_SCALING: logits = LOGIT_SCALE * logits
#     # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
#     if DO_SOFTCAPPING:   logits = SOFTCAP * triton_tanh(logits / SOFTCAP)

#     c = tl.max(logits, 0)
#     logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))

#     if chunk_idx == 0:
#         # logsumexp(chunked_logsumexp) - x
#         # Do the -x separately
#         if label_idx != -100:
#             x = tl.load(logits_ptr + label_idx).to(tl.float32)
#             # Go logit scaling for Cohere: t * x
#             if DO_LOGIT_SCALING: x = LOGIT_SCALE * x
#             # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
#             if DO_SOFTCAPPING:   x = SOFTCAP * triton_tanh(x / SOFTCAP)
#             loss = -1.0 * x
#         else:
#             loss = 0.0
#         tl.store(loss_ptr, loss)
#     pass
#     tl.store(logsumexp_ptr, logsumexp)
# pass
# _chunked_cross_entropy_forward = triton.jit(_chunked_cross_entropy_forward)
# _chunked_cross_entropy_forward = triton.heuristics(
#     {
#         "DO_SOFTCAPPING":   lambda args: bool(args["DO_SOFTCAPPING"  ]),
#         "DO_LOGIT_SCALING": lambda args: bool(args["DO_LOGIT_SCALING"]),
#     }
# )(_chunked_cross_entropy_forward)


# def _cross_entropy_backward(
#     logits_ptr        ,
#     logits_row_stride : tl.constexpr,
#     dloss_ptr         ,
#     dloss_row_stride  : tl.constexpr,
#     logsumexp_ptr     ,
#     labels_ptr        ,
#     VOCAB_SIZE        : tl.constexpr,
#     BLOCK_SIZE        : tl.constexpr,
#     DO_SOFTCAPPING    : tl.constexpr,
#     SOFTCAP           : tl.constexpr,
#     DO_LOGIT_SCALING  : tl.constexpr,
#     LOGIT_SCALE       : tl.constexpr,
# ):
#     """
#         CE_i = -y log(P) = y * (log[sum(exp(x))] - x)
#         dC/dx = d/dx (y * log[sum(exp(x))] - x * y)

#         From https://en.wikipedia.org/wiki/LogSumExp
#         d/dx logsumexp = exp(x) / sum(exp(x)) = softmax(x)

#         dC/dx = y * exp(x) / sum(exp(x)) - d/dx (x * y)
#         dC/dx = y * exp[ log[exp(x) / sum(exp(x))] ] using x = exp(log(x)) trick
#         dC/dx = y * exp[x - logsumexp] - d/dx (x * y)

#         If y == 0: dC/dx = 0
#         If y == 1 and x == label: dC/dlabel = exp[x - logsumexp] - 1
#         If y == 1 and x != label: dC/dx     = exp[x - logsumexp]
#     """
#     row_idx   = tl.program_id(0)
#     block_idx = tl.program_id(1)

#     logits_ptr += row_idx * triton_cast(logits_row_stride, tl.int64)
#     dloss_ptr  += row_idx *  dloss_row_stride
#     col_offsets = block_idx*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
#     mask = col_offsets < VOCAB_SIZE
#     label_idx = tl.load(labels_ptr + row_idx).to(tl.int32)

#     if label_idx != -100:
#         dloss = tl.load(dloss_ptr)
#     else:
#         dloss = 0.0

#     x = tl.load(logits_ptr + col_offsets, mask = mask, other = -float("inf")).to(tl.float32)

#     # Do logit scaling for Cohere
#     if DO_LOGIT_SCALING:
#         # d/dx [s * x] = s
#         x = x * LOGIT_SCALE
#     pass

#     # Do logit softcapping for Gemma 2: t * tanh(1/t * x)
#     partial = x
#     if DO_SOFTCAPPING:
#         # d/dx [t * tanh(1/t * x)] = 1 - tanh^2(1/t * x)
#         partial = triton_tanh(x / SOFTCAP)
#         x = SOFTCAP * partial
#     pass

#     logsumexp = tl.load(logsumexp_ptr + row_idx)
#     y = tl.exp(x - logsumexp)
#     y = tl.where(
#         col_offsets == label_idx,
#         y - 1.0, # exp(x - logsumexp) - 1
#         y,       # exp(x - logsumexp)
#     )

#     if DO_LOGIT_SCALING:
#         # d/dx [s * x] = s
#         y = y * LOGIT_SCALE
#     pass

#     if DO_SOFTCAPPING:
#         # d/dx [t * tanh(1/t * x)] = 1 - tanh^2(1/t * x)
#         y = y * (1.0 - partial*partial)
#     pass

#     # If y == 0: dC/dx = 0 ==> we already masked it to be = 0, so dloss = 0.
#     tl.store(logits_ptr + col_offsets, dloss * y, mask = mask)
# pass
# _cross_entropy_backward = triton.jit(_cross_entropy_backward)
# _cross_entropy_backward = triton.heuristics(
#     {
#         "DO_SOFTCAPPING":   lambda args: bool(args["DO_SOFTCAPPING"  ]),
#         "DO_LOGIT_SCALING": lambda args: bool(args["DO_LOGIT_SCALING"]),
#     }
# )(_cross_entropy_backward)


# MAX_FUSED_SIZE = 65536 # 2**16
# class Fast_CrossEntropyLoss(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, logits, labels, logit_softcapping : float = 0, logit_scaling : float = 0):
#         n_rows : int
#         vocab_size : int
#         n_rows, vocab_size = logits.shape
#         device = logits.device

#         div, mod = divmod(vocab_size, MAX_FUSED_SIZE)
#         n_chunks : int = div + (mod != 0)
#         losses = torch.empty(n_rows, dtype = torch.float32, device = device)

#         DO_SOFTCAPPING   : bool = bool(logit_softcapping != 0)
#         DO_LOGIT_SCALING : bool = bool(logit_scaling != 0)

#         BLOCK_SIZE : int
#         num_warps  : int
#         if n_chunks == 1:
#             # For small vocabs <= 65336 like Llama, Mistral
#             BLOCK_SIZE, num_warps = calculate_settings(vocab_size)
#             logsumexp = torch.empty(n_rows, dtype = torch.float32, device = device)

#             with torch_gpu_device(device):
#                 _cross_entropy_forward[(n_rows,)](
#                     logits, logits.stride(0),
#                     losses,
#                     logsumexp,
#                     labels,
#                     VOCAB_SIZE       = vocab_size,
#                     BLOCK_SIZE       = BLOCK_SIZE,
#                     DO_SOFTCAPPING   = DO_SOFTCAPPING,
#                     SOFTCAP          = logit_softcapping,
#                     DO_LOGIT_SCALING = DO_LOGIT_SCALING,
#                     LOGIT_SCALE      = logit_scaling,
#                     num_warps        = num_warps,
#                 )
#         else:
#             # For large vocabs > 65336 like Gemma 256K
#             logsumexp = torch.empty((n_rows, n_chunks,), dtype = torch.float32, device = device)

#             with torch_gpu_device(device):
#                 _chunked_cross_entropy_forward[(n_rows, n_chunks,)](
#                     logits, logits.stride(0),
#                     losses,
#                     logsumexp,
#                     labels,
#                     VOCAB_SIZE       = vocab_size,
#                     N_CHUNKS         = n_chunks,
#                     BLOCK_SIZE       = MAX_FUSED_SIZE,
#                     DO_SOFTCAPPING   = DO_SOFTCAPPING,
#                     SOFTCAP          = logit_softcapping,
#                     DO_LOGIT_SCALING = DO_LOGIT_SCALING,
#                     LOGIT_SCALE      = logit_scaling,
#                     num_warps        = 32,
#                 )
#             # logsumexp(chunked_logsumexp) - x
#             # Do the -x separately
#             logsumexp = torch.logsumexp(logsumexp, dim = 1) # Row sum
#             losses += logsumexp
#             losses.masked_fill_(labels == -100, 0) # Don't forget to mask padding out!
#         pass

#         ctx.save_for_backward(logits, logsumexp, labels)
#         ctx.DO_SOFTCAPPING    = DO_SOFTCAPPING
#         ctx.logit_softcapping = logit_softcapping
#         ctx.DO_LOGIT_SCALING  = DO_LOGIT_SCALING
#         ctx.logit_scaling     = logit_scaling
#         return losses
#     pass


#     @staticmethod
#     def backward(ctx, dlosses):
#         logits, logsumexp, labels = ctx.saved_tensors
#         n_rows : int
#         vocab_size : int
#         n_rows, vocab_size = logits.shape

#         BLOCK_SIZE : int = 4096
#         div : int
#         mod : int
#         div, mod = divmod(vocab_size, BLOCK_SIZE)
#         n_blocks : int = div + (mod != 0)

#         with torch_gpu_device(dlosses.device):
#             _cross_entropy_backward[(n_rows, n_blocks,)](
#                 logits,   logits.stride(0),
#                 dlosses, dlosses.stride(0),
#                 logsumexp,
#                 labels,
#                 VOCAB_SIZE       = vocab_size,
#                 BLOCK_SIZE       = BLOCK_SIZE,
#                 DO_SOFTCAPPING   = ctx.DO_SOFTCAPPING,
#                 SOFTCAP          = ctx.logit_softcapping,
#                 DO_LOGIT_SCALING = ctx.DO_LOGIT_SCALING,
#                 LOGIT_SCALE      = ctx.logit_scaling,
#                 num_warps        = 8,
#             )
#         return logits, None, None, None,
#     pass
# pass


# def fast_cross_entropy_loss(
#     logits,
#     labels,
#     logit_softcapping = 0,
#     logit_scaling = 0,
#     n_items = None,

# ):
#     """
#     Arguments:
#         logits: (batch, seq_len, vocab_size)
#         labels: (batch, seq_len,)
#     Returns:
#         losses: float
#     """
#     batch, seq_len, d = logits.shape
#     assert(labels.shape == (batch, seq_len))

#     loss = Fast_CrossEntropyLoss.apply(
#         logits.view(batch*seq_len, d),
#         labels.view(-1),
#         logit_softcapping,
#         logit_scaling,
#     )
#     if n_items is None:
#         n_items = torch.count_nonzero(labels != -100)
#     return loss.sum() / n_items
# pass
# if (Version(torch.__version__) < Version("2.4.0")) and \
#     not hasattr(fast_cross_entropy_loss, "__wrapped__"):
#     fast_cross_entropy_loss = torch._disable_dynamo(fast_cross_entropy_loss)
# pass

# # Patch CE Losses in transformers
# def patch_loss_functions(torch_compile = True):
#     _patch_loss_functions(fast_cross_entropy_loss, torch_compile = torch_compile)
# pass
# # patch_loss_functions()
# def PatchForCausalLMLoss(
#     logits,
#     labels,
#     vocab_size: int,
#     num_items_in_batch: Optional[torch.Tensor] = None,
#     ignore_index: int = -100,
#     shift_labels: Optional[torch.Tensor] = None,
#     **kwargs,
# ) -> torch.Tensor:
#     print('FUCKKKK')
#     # Upcast to float if we need to compute the loss to avoid potential precision issues
#     logits = logits.float() # ok

#     if shift_labels is None:
#         # Shift so that tokens < n predict n
#         labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
#         shift_labels = labels[..., 1:].contiguous()

#     # Flatten the tokens
#     logits = logits.view(-1, vocab_size)
#     shift_labels = shift_labels.view(-1)
#     # Enable model parallelism
#     shift_labels = shift_labels.to(logits.device)
#     loss = fast_cross_entropy_loss(logits, shift_labels, n_items=num_items_in_batch, **kwargs)
#     return loss

# import transformers.loss.loss_utils
# transformers.loss.loss_utils.ForCausalLMLoss = PatchForCausalLMLoss
# transformers.loss.loss_utils.fixed_cross_entropy = fast_cross_entropy_loss

from transformers import Seq2SeqTrainer
from transformers.trainer import _is_peft_model
from typing_extensions import override

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from ...extras.packages import is_transformers_version_equal_to_4_46
from ...extras.packages import is_transformers_version_greater_than
from ...monkeypatch.loss.chunked import patch_chunked_ce_loss_fn
from ..callbacks import SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import PreTrainedTokenizer, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

patch_chunked_ce_loss_fn()
logger.info_rank0('Patch CE Loss -> chunked CE Loss to reduce vram')

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    r"""Inherits Seq2SeqTrainer to compute generative metrics such as BLEU and ROUGE."""

    def __init__(
        self,
        model,
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        gen_kwargs: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> None:
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
            _, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="sum")
            # print(outputs)
            logits, labels = outputs["logits"] if isinstance(outputs, dict) else outputs[1], inputs["labels"]
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
