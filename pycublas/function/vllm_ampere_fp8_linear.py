import os
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from typing import List, Optional

from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

import pycublas.quant.fp8 as fp8_quant
import pycublas.trtllm_moe_grouped_gemm as moe_kernel

from torch.utils.cpp_extension import load, load_inline


class AmpereFP8Gemm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, scale, cfg_id = 0):
        output = torch.empty(input.size(0), weight.size(1), dtype=torch.float16, device='cuda')
        moe_kernel.gemm(input, weight, scale, output, cfg_id)
        return output


class AmpereFP8Linear(torch.nn.Module):
    """FP8 linear layer.

    Args:
        input_size: input dimension of the linear layer.
        output_size: output dimension of the linear layer.
        bias: If true, add bias.
        skip_bias_add: If true, skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        quant_config: Quantization configure.
    """

    def __init__(self,
                 input_size: int,
                 output_size: int
                 ):
        super().__init__()
        self.is_sm80 = fp8_quant.is_sm80()
        assert self.is_sm80, "Only SM80 is supported for Ampere FP8 linear layer."

        self.weight = Parameter(torch.empty(input_size, output_size, dtype=torch.int8, device='cuda'), requires_grad=False)
        self.scale = Parameter(torch.empty(output_size, dtype=torch.float16, device='cuda'), requires_grad=False)

        set_weight_attrs(self.weight, {
            "weight_loader": self.weight_loader,
        })

        self.input_size = input_size
        self.output_size = output_size
        self.cfg = dict()
        self.default_cfg = 70
    
    def import_cfg_from(self, cfg: dict, default_cfg: int = 70):
        self.cfg = cfg
        self.default_cfg = default_cfg
    
    def import_weight_from(self, weight: torch.Tensor):
        assert weight.dtype == torch.float16 or weight.dtype == torch.bfloat16, "Weight must be [b]float16, but got {}".format(weight.dtype)
        assert weight.size(0) == self.input_size, "Input size mismatch, expect {}, but got {}".format(self.input_size, weight.size(0))
        assert weight.size(1) == self.output_size, "Output size mismatch, expect {}, but got {}".format(self.output_size, weight.size(1))
        w, s = fp8_quant.ops.scaled_fp8_quant(weight)
        w, s = fp8_quant.preproces_fp8_linear_weights(w, s)

        self.weight = Parameter(w, requires_grad=False)
        self.scale = Parameter(s, requires_grad=False)

    def forward(self, act: torch.Tensor) -> torch.Tensor:
        output = torch.empty(act.size(0), self.output_size, dtype=torch.float16, device='cuda')
        
        act_dtype = None
        if act.dtype != torch.float16:
            act_dtype = act.dtype
            act = act.to(dtype=torch.float16)

        if (self.input_size, self.output_size) not in self.cfg:
            cfg_id = self.default_cfg
        else:
            cfg_id = self.cfg[(self.input_size, self.output_size)].get(act.size(0), (self.default_cfg, -1))[0]

        moe_kernel.gemm(act, self.weight, self.scale, output, cfg_id)

        if act_dtype is not None:
            output = output.to(dtype=act_dtype)
        return output
    
    def tune_forward(self, act: torch.Tensor, cfg_cache: Optional[dict] = None) -> dict:
        output = torch.empty(act.size(0), self.output_size, dtype=torch.float16, device='cuda')

        if cfg_cache is None:
            cfg_cache = {(self.input_size, self.output_size):{}} # (m, n, k) -> (cfg_id, time)
        elif (self.input_size, self.output_size) not in cfg_cache:
            cfg_cache[(self.input_size, self.output_size)] = {}
        
        act_dtype = None
        if act.dtype != torch.float16:
            act_dtype = act.dtype
            act = act.to(dtype=torch.float16)

        for cfg_id in range(0, moe_kernel.get_gemm_configs_count()):
            try:
                all_time = 0.0
                times = 100
                for j in range(10 + times):
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    moe_kernel.gemm(act, self.weight, self.scale, output, cfg_id)
                    end.record()
                    torch.cuda.synchronize()
                    elapsed_time_ms = start.elapsed_time(end)
                    if j >= 10:
                        all_time += elapsed_time_ms
                old_all_time = cfg_cache[(self.input_size, self.output_size)].get(act.size(0), (cfg_id, -1))[1]
                all_time /= times
                if all_time < old_all_time or old_all_time < 0:
                    cfg_cache[(self.input_size, self.output_size)][act.size(0)] = (cfg_id, all_time)

            except Exception as e:
                continue

        if act_dtype is not None:
            output = output.to(dtype=act_dtype)
        
        return cfg_cache
    
    # This is to fit vllm model executor
    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        self.import_weight_from(loaded_weight.view(loaded_weight.size(0), -1).t())