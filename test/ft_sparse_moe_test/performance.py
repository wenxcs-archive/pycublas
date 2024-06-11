import os
import sys
import torch
import pycublas
import pycublas.fasttransformer_moe_sparse_gemm as ft_moe
from vllm import _custom_ops as ops

#!pip install vllm to test

def moe_perf(
    tokens=1024,
    experts=1,
    topk=1,
    intermediate_size=6400,
    hidden_size=4096,
    config=None,
    times=100,
    use_fp8 = True
):
    torch.manual_seed(0)
    hidden_state = torch.ones(tokens*topk, hidden_size).cuda().half()
    w1 = (torch.ones(experts, hidden_size, intermediate_size * 2)).to(torch.int8).cuda()
    w2 = (torch.ones(experts, intermediate_size, hidden_size)).to(torch.int8).cuda()
    w1_scale = torch.ones([experts]).cuda().half() / 10000
    w2_scale = torch.ones([experts]).cuda().half()
    rows_per_expert = (torch.ones([experts])*(tokens*topk//experts)).cuda().to(torch.int64)

    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        intermediate_cache2 = torch.empty((tokens*topk, intermediate_size),
                                      device=hidden_state.device,
                                      dtype=hidden_state.dtype)
        start.record()
        a1 = ft_moe.grouped_gemm(hidden_state, w1, w1_scale, rows_per_expert)
        print(a1)
        exit(0)
        ops.silu_and_mul(intermediate_cache2, a1)
        a2 = ft_moe.grouped_gemm(intermediate_cache2, w2, w2_scale, rows_per_expert)
        end.record()
        torch.cuda.synchronize()
        if j >= 10:
            all_time += start.elapsed_time(end)
    
    return all_time/times

searchspace = [1] + list(range(0, 256, 32))[1:] + list(range(256, 4097, 256))

searchspace = [4096]

for tk in searchspace:
    print(
        tk,
        ",",
        moe_perf(tokens=tk),
    )