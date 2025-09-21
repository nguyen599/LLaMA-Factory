# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import torch
from packaging.version import Version
import os
import math
import functools
from typing import Optional
torch_nn_functional_cross_entropy = torch.nn.functional.cross_entropy
from triton import __version__ as triton_version
import inspect
import sys
import logging
import functools

__all__ = [
    "patch_loss_functions",
    "torch_compile_options",
    "UNSLOTH_ENABLE_LOGGING",
    "get_torch_compile_options",
]

UNSLOTH_ENABLE_LOGGING  = os.environ.get("UNSLOTH_ENABLE_LOGGING",  "0") == "1"

def get_device_type():
    import torch
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    raise NotImplementedError("Unsloth currently only works on NVIDIA GPUs and Intel GPUs.")
pass
DEVICE_TYPE : str = get_device_type()

# Get only allowed options
inductor_config_source = inspect.getsource(torch._inductor.config)

@functools.lru_cache(1)
def determine_compile_threads():
    # See https://github.com/pytorch/pytorch/blob/ab2294d8289a7757a2fc321cdefac88e2b378edf/torch/_inductor/config.py#L771
    # Windows thread count = 1. See https://github.com/unslothai/unsloth-zoo/pull/187
    if sys.platform == "win32": return 1
    cpu_count = os.cpu_count()
    return min(32, max(4, cpu_count))
pass

def get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
    coordinate_descent_tuning = False,
    logging = False,
    combo_kernels = False,
    group_fusion = True,
    memory_planning = True,
    multi_kernel = False,
    use_block_ptr = False,
):
    UNSLOTH_COMPILE_DEBUG         = os.environ.get("UNSLOTH_COMPILE_DEBUG",         "0") == "1"
    UNSLOTH_COMPILE_MAXIMUM       = os.environ.get("UNSLOTH_COMPILE_MAXIMUM",       "0") == "1"
    UNSLOTH_COMPILE_IGNORE_ERRORS = os.environ.get("UNSLOTH_COMPILE_IGNORE_ERRORS", "0") == "1"
    if UNSLOTH_ENABLE_LOGGING: logging = True

    # https://github.com/pytorch/pytorch/blob/c665594c1edca9a507b0ec8b18ab74a0ecb65bc3/torch/_inductor/config.py#L1283
    # Needs integer
    multi_kernel = 1 if multi_kernel else 0

    # Instead of Inductor Compilation:
    try:
        import torch._inductor.async_compile
        from torch.hub import tqdm
        def replaced_tqdm(*args, **kwargs):
            kwargs["desc"] = "Unsloth: Compiling kernels"
            return tqdm(*args, **kwargs)
        torch._inductor.async_compile.tqdm = replaced_tqdm
    except:
        print("Unsloth: Failed editing tqdm to replace Inductor Compilation:")
    pass

    torch_compile_options = {
        "epilogue_fusion"           : epilogue_fusion,
        "max_autotune"              : max_autotune,
        "shape_padding"             : shape_padding,
        "trace.enabled"             : UNSLOTH_COMPILE_DEBUG or debug,
        "triton.cudagraphs"         : cudagraphs,
        "debug"                     : UNSLOTH_COMPILE_DEBUG or debug,
        "dce"                       : True,
        "memory_planning"           : memory_planning,
        "coordinate_descent_tuning" : coordinate_descent_tuning or UNSLOTH_COMPILE_MAXIMUM,
        "trace.graph_diagram"       : UNSLOTH_COMPILE_DEBUG or debug,
        "compile_threads"           : determine_compile_threads(), # Auto detects via https://github.com/unslothai/unsloth-zoo/pull/187
        "group_fusion"              : group_fusion, # [DEPRECATED]
        "disable_progress"          : not logging,
        "verbose_progress"          : logging,

        "triton.multi_kernel"       : multi_kernel, # RuntimeError: name 'multi_kernel_0' is not defined
        "triton.use_block_ptr"      : use_block_ptr,
        "triton.enable_persistent_tma_matmul" : True,
        "triton.autotune_at_compile_time"     : False,
        "triton.cooperative_reductions"       : False,
        # "reorder_for_compute_comm_overlap"  : True, # Fails for single GPU
        "cuda.compile_opt_level"              : "-O2",
        "cuda.enable_cuda_lto"                : True,
        # "cuda.use_fast_math"                : True, # Disable fast math
        # Causes incompatible gradient sizes on 2.6
        # And TypeError: bad operand type for unary -: 'SymbolicCallArg'
        "combo_kernels"                       : combo_kernels,
        "benchmark_combo_kernel"              : True,
        "combo_kernel_foreach_dynamic_shapes" : True,
    }
    final_torch_compile_options = {}
    for key, value in torch_compile_options.items():
        splits = key.split(".")
        if all(k in inductor_config_source for k in splits):
            final_torch_compile_options[key] = value
    return final_torch_compile_options
pass
torch_compile_options = get_torch_compile_options(
    epilogue_fusion = True,
    max_autotune = False,
    shape_padding = True,
    debug = False,
    cudagraphs = False,
    coordinate_descent_tuning = False,
    logging = UNSLOTH_ENABLE_LOGGING,
    combo_kernels = False,
    group_fusion = False,
    memory_planning = False,
    multi_kernel = False,
    use_block_ptr = False,
)

def patch_loss_functions(_fast_cross_entropy_loss, torch_compile = True):
    # All Unsloth Zoo code licensed under LGPLv3
    try:
        import transformers.loss.loss_utils
    except:
        print("Unsloth: Cannot patch loss functions - update transformers for faster modules!")
        return None
    pass

    # Generic cross entropy loss
    def unsloth_fixed_cross_entropy(source, target, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs):
        if ignore_index == -100:
            loss = _fast_cross_entropy_loss(
                logits  = source,
                labels  = target,
                n_items = num_items_in_batch,
            )
        else:
            reduction = "sum" if num_items_in_batch is not None else "mean"
            loss = torch_nn_functional_cross_entropy(
                source,
                target,
                ignore_index = ignore_index,
                reduction    = reduction,
            )
            if reduction == "sum": loss = loss / num_items_in_batch
        return loss
    pass

    # Causal LM loss
    def UnslothForCausalLMLoss(
        logits, labels, vocab_size: int, num_items_in_batch: int = None, ignore_index: int = -100, **kwargs
    ):
        if labels is None: return None
        shift_logits = logits
        shift_labels = torch.empty_like(labels)
        shift_labels[..., :-1] = labels[..., 1:]
        shift_labels[..., -1] = ignore_index
        loss = unsloth_fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
        return loss
    pass

    if (Version(torch.__version__) < Version("2.4.0")):
        UnslothForCausalLMLoss = torch._disable_dynamo(UnslothForCausalLMLoss)

    elif torch_compile:
        UnslothForCausalLMLoss = torch.compile(
            UnslothForCausalLMLoss,
            dynamic = True,
            fullgraph = False,
            options = torch_compile_options,
        )
    pass

    # Now patch the losses!
    import transformers.modeling_utils
    LOSS_MAPPING = transformers.loss.loss_utils.LOSS_MAPPING
    LOSS_MAPPING["ForCausalLM"] = UnslothForCausalLMLoss

    # Remove @property and @lru_cache
    if hasattr(transformers.modeling_utils.PreTrainedModel.loss_function, "fget") and \
        hasattr(transformers.modeling_utils.PreTrainedModel.loss_function.fget, "__wrapped__"):
        transformers.modeling_utils.PreTrainedModel.loss_function = \
            transformers.modeling_utils.PreTrainedModel.loss_function.fget.__wrapped__
    pass
    print("Unsloth: Patched cross entropy losses.")
    os.environ["UNSLOTH_PATCHED"] = "1"
pass



# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
