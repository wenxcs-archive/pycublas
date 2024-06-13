import os
import sys
import torch
import pycublas
import pycublas.trtllm_moe_grouped_gemm as ft_moe
from vllm import _custom_ops as ops

#in_size=6400/3200, out_size=4096, cfg_id=14
#in_size=4096, out_size=12800, cfg_id=26 - 29

import functools

def timeit_decorator_event(times=100):
    def decodrator(function_call):
        @functools.wraps(function_call)
        def wrapper(*args, **kwargs):
            all_time = 0.0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            for j in range(10):
                function_call(*args, **kwargs)
            start.record()
            for j in range(times):
                function_call(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end)
            all_time = elapsed_time_ms
            avg_time = all_time / times
            print(f"{function_call.__name__} average time: {avg_time} ms")
            return function_call(*args, **kwargs)

def timeit_decorator_cudagraph(times=100):
    def decorator(function_call):
        @functools.wraps(function_call)
        def wrapper(*args, **kwargs):

            # cuda graph
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for i in range(3):
                    function_call(*args, **kwargs)
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                function_call(*args, **kwargs)

            all_time = 0.0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for j in range(times):
                #function_call(*args, **kwargs)
                g.replay()

            end.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end)
            all_time = elapsed_time_ms
            
            avg_time = all_time / times
            print(f"{function_call.__name__} average time: {avg_time} ms")
            return function_call(*args, **kwargs)
        
        return wrapper
    return decorator

def test_grouped_gemm(
    tokens=1024,
    experts=16,
    topk=2,
    in_size=4096,
    out_size=12800,
    times=100,
):
    assert tokens*topk % experts == 0, "tokens*topk % experts != 0"
    torch.manual_seed(1234)

    # input output
    hidden_state = torch.randn(tokens*topk, in_size).cuda().half()
    out = torch.zeros(tokens*topk, out_size).cuda().half()

    # w1
    w1_f32 = torch.randn(experts, in_size, out_size).cuda()
    w1, w1_scale = ops.scaled_fp8_quant(
        w1_f32.half(), torch.ones(experts, dtype=torch.float32, device=w1_f32.device) * 0.0022
    )
    w1 = w1.view(dtype=torch.int8)

    g_w1_scale = w1_scale.to(dtype=torch.float16).unsqueeze(1).expand(-1, out_size).contiguous()

    total_rows_before_expert = (torch.ones([experts])*(tokens*topk//experts)).cuda().to(torch.int64)
    for i in range(1, experts):
        total_rows_before_expert[i] = total_rows_before_expert[i-1] + total_rows_before_expert[i]
    
    cfg_id = 29 if out_size == 12800 else 14
    
    def run_cuda_graph(*args, **kwargs):
        ft_moe.grouped_gemm(*args, **kwargs)

    for cfg_id in range(0, 30):
        all_time = 0.0
        for j in range(10 + times):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_cuda_graph(hidden_state, w1, g_w1_scale, total_rows_before_expert, out, 5, cfg_id)
            end.record()
            torch.cuda.synchronize()
            elapsed_time_ms = start.elapsed_time(end)

            if j >= 10:
                all_time += elapsed_time_ms
            
        print(f"{tokens}, {cfg_id}, {in_size}, {out_size}, {all_time/times:.3f}")

    for expert in range(experts):
        w = w1_f32[expert, :, :]
        h = hidden_state.view(experts, tokens*topk // experts, in_size) [expert, :, :]
        c = h @ w.half()
        o = out.view(experts, tokens*topk // experts, out_size) [expert, :, :]
        print(c)
        print(o)
        torch.testing.assert_close(c, o, rtol=1e-1, atol=1e-2)
    
    #print(f"{tokens}, {topk}, {experts}, {tokens*topk}, {in_size}, {out_size}, {all_time/times:.3f}")

#for t in [128, 256, 512, 1024, 2048]:
#    for shape in [(4096, 12800), (6400, 4096)]:
#        test_grouped_gemm(tokens=t, experts=16, topk=2, in_size=shape[0], out_size=shape[1])


# Config performance tuning

# Corectness check

def test_grouped_gemm_correctness(
    tokens=1024,
    experts=16,
    topk=2,
    in_size=4096,
    out_size=12800,
    times=100,
    cfg_id=0,
):
    assert tokens*topk % experts == 0, "tokens*topk % experts != 0"
    torch.manual_seed(1234)

    # input output
    hidden_state = torch.ones(tokens*topk, in_size).cuda().uniform_(-1, 1).half()
    out = torch.empty(tokens*topk, out_size).cuda().half()

    # w1
    w1_f32 = torch.randn(experts, in_size, out_size).uniform_(-1, 1).cuda()
    w1, w1_scale = ops.scaled_fp8_quant(
        w1_f32.half(), torch.ones(experts, dtype=torch.float32, device=w1_f32.device) * 0.0022
    )

    w1 = w1.view(dtype=torch.int8)
    w1_i = ft_moe.preprocess_weights_for_mixed_gemm(w1.cpu()).to(hidden_state.device)

    g_w1_scale = w1_scale.to(dtype=torch.float16).unsqueeze(1).expand(-1, out_size).contiguous()

    total_rows_before_expert = (torch.ones([experts])*(tokens*topk//experts)).cuda().to(torch.int64)
    for i in range(1, experts):
        total_rows_before_expert[i] = total_rows_before_expert[i-1] + total_rows_before_expert[i]
    
    ft_moe.grouped_gemm(hidden_state, w1_i, g_w1_scale, total_rows_before_expert, out, 5, cfg_id)

    for expert in range(experts):
        w = w1_f32[expert, :, :]
        h = hidden_state.view(experts, tokens*topk // experts, in_size) [expert, :, :]
        c = h @ w.half()
        o = out.view(experts, tokens*topk // experts, out_size) [expert, :, :]
        torch.testing.assert_close(c, o, rtol=1e-0, atol=1e-1)