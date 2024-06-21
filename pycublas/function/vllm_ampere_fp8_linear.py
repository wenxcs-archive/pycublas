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

# reuse from moe group gemm
moe_gg_kernel_config = {
    1: (13, 21, 0.4587008017301559),
    2: (5, 11, 0.4829593604803085),
    3: (11, 4, 0.55322624117136),
    4: (5, 5, 0.6300467216968536),
    5: (5, 9, 0.6892339181900025),
    6: (5, 5, 0.7366860777139663),
    7: (17, 9, 0.7817830407619476),
    8: (5, 8, 0.8124313586950302),
    16: (5, 5, 1.0158489656448364),
    32: (4, 17, 1.0969907104969026),
    48: (5, 4, 1.1068108654022217),
    64: (17, 5, 1.1107225465774535),
    80: (4, 5, 1.1139481484889984),
    96: (16, 16, 1.1225907170772553),
    112: (16, 16, 1.1334041678905487),
    128: (17, 17, 1.137500158548355),
    144: (16, 17, 1.144709119796753),
    160: (16, 17, 1.1540889596939088),
    176: (16, 16, 1.1627110350131988),
    192: (17, 16, 1.1790643167495727),
    208: (22, 16, 1.2127846336364747),
    224: (23, 17, 1.2236697602272033),
    240: (22, 22, 1.2352307152748108),
    256: (23, 22, 1.2356915152072907),
    512: (23, 22, 1.6425676786899566),
    768: (27, 27, 1.7934028828144073),
    1024: (27, 23, 2.4730009508132933),
    1280: (22, 22, 3.02405633687973),
    1536: (27, 22, 3.2711680245399477),
    1792: (27, 26, 3.344619517326355),
    2048: (27, 26, 4.023920638561249),
    2304: (26, 22, 4.71138304233551),
    2560: (27, 27, 4.861614079475403),
    2816: (27, 27, 4.988712968826294),
    3072: (26, 27, 5.624104981422424),
    3328: (27, 26, 6.2363647985458375),
    3584: (26, 26, 6.384680962562561),
    3840: (26, 27, 6.581227521896363),
    4096: (26, 27, 7.1324774312973025),
}


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
        self.scale = Parameter(torch.empty(1, output_size, dtype=torch.float16, device='cuda'), requires_grad=False)

        self.input_size = input_size
        self.output_size = output_size
    
    def import_weight_from(self, weight: torch.Tensor):
        assert weight.dtype == torch.float16 or weight.dtype == torch.bfloat16, "Weight must be [b]float16, but got {}".format(weight.dtype)
        w, s = fp8_quant.ops.scaled_fp8_quant(weight)
        w, s = fp8_quant.preproces_fp8_linear_weights(w, s)

        self.weight = Parameter(w, requires_grad=False)
        self.scale = Parameter(s, requires_grad=False)

    def forward(self, act: torch.Tensor) -> torch.Tensor:
        total_rows_before_expert = torch.tensor([act.size(0)], dtype=torch.int64, device='cuda')
        cfg_id_0, cfg_id_1, _ = moe_gg_kernel_config[min(moe_gg_kernel_config.keys(), key=lambda x: abs(x - 1))]
        cfg_id = max(cfg_id_0, cfg_id_1)
        output = torch.empty(act.size(0), self.weight.size(1), dtype=torch.float16, device='cuda')
        
        act_dtype = None
        if act.dtype != torch.float16:
            act_dtype = act.dtype
            act = act.to(dtype=torch.float16)

        moe_kernel.grouped_gemm(act, self.weight.unsqueeze(0), self.scale, total_rows_before_expert, output, 5, cfg_id)
        output.squeeze_(0)
        if act_dtype is not None:
            output = output.to(dtype=act_dtype)
        return output