import os
import sys
import torch
import pycublas
from pycublas.function.vllm_ampere_fp8_linear import AmpereFP8Linear
from loguru import logger

logger.warning("fp8 allclose may not be accurate") 

def test_ampere_fp8_linear():
    # Activation: bf16/fp16 as input, but use fp16 inside the layer, the output's dtype will be same as input
    act = torch.ones(1024, 4096, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()
    # Weight: fp16/bf16, shape is [hidden_size, output_size](Not same as normal linear layer)
    w = torch.ones(4096, 2048, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()
    
    fp8linear = AmpereFP8Linear(input_size=4096, output_size=2048)
    # After create fp8linear, we need to import the weight, no quant flag is required here.
    fp8linear.import_weight_from(w)
    output_fp8 = fp8linear(act)
    output_ref = torch.matmul(act, w)

    print("output_fp8: ", output_fp8)
    print("output_ref: ", output_ref)
    torch.testing.assert_close(output_fp8, output_ref, rtol=1e-0, atol=1e-1)