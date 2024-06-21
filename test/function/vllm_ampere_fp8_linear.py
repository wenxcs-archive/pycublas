import os
import sys
import torch
import pycublas
from pycublas.function.vllm_ampere_fp8_linear import AmpereFP8Linear
from loguru import logger

logger.warning("fp8 allclose may not be accurate") 

def test_ampere_fp8_linear():
    act = torch.ones(1024, 4096, dtype=torch.float16).cuda()
    w = torch.ones(4096, 2048, dtype=torch.float16).cuda()
    
    fp8linear = AmpereFP8Linear(input_size=4096, output_size=12800)
    fp8linear.import_weight_from(w)
    output_fp8 = fp8linear(act)

    output_ref = torch.matmul(act, w)

    print("output_fp8: ", output_fp8.shape)
    print("output_ref: ", output_ref.shape)

    print("output_fp8: ", output_fp8)
    print("output_ref: ", output_ref)
    
    torch.testing.assert_close(output_fp8, output_ref, rtol=1e-0, atol=1e-1)