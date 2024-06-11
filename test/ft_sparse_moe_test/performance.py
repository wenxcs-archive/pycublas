import os
import sys
import torch
import pycublas
import pycublas.trtllm_moe_grouped_gemm as ft_moe
from vllm import _custom_ops as ops

#!pip install vllm to test

def test_grouped_gemm(
    tokens=1024,
    experts=2,
    topk=1,
    hidden_size=4096,
    intermediate_size=6400,
):
    assert tokens*topk % experts == 0, "tokens*topk % experts != 0"
    torch.manual_seed(12345)
    hidden_state = torch.ones(tokens*topk, hidden_size).cuda().half()
    w1 = (torch.ones(experts, hidden_size, intermediate_size * 2)*1).to(torch.int8).cuda()
    w1_scale = torch.ones([experts]).cuda().half()
    total_rows_before_expert = (torch.ones([experts])*(tokens*topk//experts)).cuda().to(torch.int64)
    for i in range(1, experts):
        total_rows_before_expert[i] = total_rows_before_expert[i-1] + total_rows_before_expert[i]
    for i in range(1, experts):
        total_rows_before_expert[i] = total_rows_before_expert[i] - total_rows_before_expert[0]
    total_rows_before_expert[0] = 0

    print(total_rows_before_expert, hidden_state.shape[0])

    a1 = ft_moe.grouped_gemm(hidden_state, w1, w1_scale, total_rows_before_expert)
    print(a1)


def moe_perf(
    tokens=4096,
    experts=16,
    topk=2,
    intermediate_size=6400,
    hidden_size=4096,
    config=None,
    times=100,
    use_fp8 = True
):
    torch.manual_seed(0)
    hidden_state = torch.ones(tokens*topk, hidden_size).cuda().half()
    w1 = (torch.ones(experts, hidden_size, intermediate_size * 2)*1).to(torch.int8).cuda()
    w2 = (torch.ones(experts, intermediate_size, hidden_size)*1).to(torch.int8).cuda()
    w1_scale = torch.ones([experts]).cuda().half()
    w2_scale = torch.ones([experts]).cuda().half()
    rows_per_expert = (torch.ones([experts])*(tokens*topk//experts)).cuda().int()
    total_rows_before_expert = (torch.ones([experts])*(tokens*topk//experts)).cuda().to(torch.int64)
    for i in range(1, experts):
        total_rows_before_expert[i] = total_rows_before_expert[i-1] + total_rows_before_expert[i]
    for i in range(1, experts):
        total_rows_before_expert[i] = total_rows_before_expert[i] - total_rows_before_expert[0]
    total_rows_before_expert[0] = 0

    all_time = 0.0
    for j in range(10 + times):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        intermediate_cache2 = torch.empty((tokens*topk, intermediate_size),
                                      device=hidden_state.device,
                                      dtype=hidden_state.dtype)
        start.record()
        a1 = ft_moe.grouped_gemm(hidden_state, w1, w1_scale, total_rows_before_expert)
        ops.silu_and_mul(intermediate_cache2, a1)
        a2 = ft_moe.grouped_gemm(intermediate_cache2, w2, w2_scale, total_rows_before_expert)
        end.record()
        torch.cuda.synchronize()
        if j >= 10:
            all_time += start.elapsed_time(end)
    
    return all_time/times

searchspace = list(range(2048, 4097, 256))

searchspace = [512, 1024, 2048, 4096]

for tk in searchspace:
    print(
        tk,
        ",",
        moe_perf(tokens=tk, topk=1),
    )