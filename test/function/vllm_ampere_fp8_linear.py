import os
import sys
import torch
import pycublas
from pycublas.function.vllm_ampere_fp8_linear import AmpereFP8Linear
from loguru import logger

logger.warning("fp8 allclose may not be accurate") 

def test_ampere_fp8_linear():

    cfg_cache = {}
    # Weight: fp16/bf16, shape is [hidden_size, output_size](Not same as normal linear layer)
    #w_0 = torch.ones(3072, 16384, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()
    #w_1 = torch.ones(8192, 3072, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()
    w_0 = torch.ones(4096, 12800, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()
    w_1 = torch.ones(12800//2, 4096, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()

    fp8linear_0 = AmpereFP8Linear(input_size=4096, output_size=12800)
    fp8linear_1 = AmpereFP8Linear(input_size=12800//2, output_size=4096)
    # After create fp8linear, we need to import the weight, no quant flag is required here.
    fp8linear_0.import_weight_from(w_0)
    fp8linear_1.import_weight_from(w_1)

    for m in [1] + list(range(256, 1024, 256)):
        # Activation: bf16/fp16 as input, but use fp16 inside the layer, the output's dtype will be same as input
        act_0 = torch.ones(m, 4096, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()
        act_1 = torch.ones(m, 12800//2, dtype=torch.float16).uniform_(-0.1, 0.1).cuda()
        cfg_cache = fp8linear_0.tune_forward(act_0, cfg_cache)
        cfg_cache = fp8linear_1.tune_forward(act_1, cfg_cache)
    
    print("cfg: ", cfg_cache)

    for k in cfg_cache:
        isz, osz = k
        for m in cfg_cache[k]:
            cfg_id, time = cfg_cache[k][m]

            tflo = m * isz * osz * 2 / 1e12
            tflops = tflo / time * 1e3

            data_size = (m * isz * 2 + isz * osz + m * osz * 2) / 1e9
            bandwidth = data_size / time * 1e3

            print (f"m={m}, isz={isz}, osz={osz}, cfg_id={cfg_id}, time={time:.2f}ms, tflops={tflops:.2f}TFLOPS, bandwidth={bandwidth:.2f}GB/s")

    #output_fp8 = fp8linear(act)
    #output_ref = torch.matmul(act, w)
    #torch.testing.assert_close(output_fp8, output_ref, rtol=1e-0, atol=1e-1)