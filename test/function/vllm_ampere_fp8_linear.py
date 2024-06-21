import os
import sys
import torch
import pycublas
from pycublas.function.vllm_ampere_fp8_linear import AmpereFP8Linear
from vllm.model_executor.layers.quantization.fp8 import Fp8Config

def test_ampere_fp8_linear():
    fp8linear = AmpereFP8Linear(input_size=4096, output_size=12800, quant_config=Fp8Config)
    act = torch.randn(1024, 4096, dtype=torch.float16).cuda()
    w = torch.randn(4096, 12800, dtype=torch.float16).cuda()
    fp8linear.import_weight_from_fp16(w)

    output_fp8 = fp8linear(act)
    output_ref = torch.matmul(act, w)

    print("output_fp8: ", output_fp8)
    print("output_ref: ", output_ref)