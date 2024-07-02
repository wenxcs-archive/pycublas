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

def test_grouped_gemm_perf(
    experts=2,
    topk=2,
    in_size=4096,
    out_size=12800,
    times=100,
):
    torch.manual_seed(1234)
    w1_f32 = torch.randn(experts, in_size, out_size).cuda()
    w1, w1_scale = ops.scaled_fp8_quant(
        w1_f32.half(), torch.ones(experts, dtype=torch.float32, device=w1_f32.device) * 0.0022
    )
    w1 = w1.view(dtype=torch.int8)
    g_w1_scale = w1_scale.to(dtype=torch.float16).unsqueeze(1).expand(-1, out_size).contiguous()
    
    searchspace = list(range(32, 256, 32)) + list(range(256, 4097, 256))
    searchspace = [1]
    best_configs = dict()
    for tokens in searchspace:
        # input output
        assert tokens*topk % experts == 0, "tokens*topk % experts != 0"
        hidden_state = torch.randn(tokens*topk, in_size).cuda().half()
        out = torch.zeros(tokens*topk, out_size).cuda().half()

        total_rows_before_expert = (torch.ones([experts])*(tokens*topk//experts)).cuda().to(torch.int64)
        for i in range(1, experts):
            total_rows_before_expert[i] = total_rows_before_expert[i-1] + total_rows_before_expert[i]
        
        min_time = 1000000.0
        min_cfg_id = 0

        for cfg_id in range(0, 30):
            all_time = 0.0
            for j in range(10 + times):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                ft_moe.grouped_gemm(hidden_state, w1, g_w1_scale, total_rows_before_expert, out, 5, cfg_id)
                end.record()
                torch.cuda.synchronize()
                elapsed_time_ms = start.elapsed_time(end)

                if j >= 10:
                    all_time += elapsed_time_ms
            
            if all_time < min_time:
                min_time = all_time
                min_cfg_id = cfg_id
                print(f"new config id > {tokens}, {cfg_id}, {in_size}, {out_size}, {all_time/times:.3f}")
        
        best_configs[tokens] = min_cfg_id
    
    print({ in_size:best_configs})

# key = sorted_ids.size(0) // 2  // 16
# {4096:{1:4, 32: 5, 64: 4, 96: 5, 128: 4, 160: 17, 192: 17, 224: 16, 256: 16, 512: 23, 768: 26, 1024: 26, 1280: 23, 1536: 23, 1792: 26, 2048: 26, 2304: 27, 2560: 27, 2816: 27, 3072: 27, 3328: 27, 3584: 26, 3840: 26, 4096: 26}}
# {6400: {1:4, 32: 4, 64: 4, 96: 4, 128: 4, 160: 16, 192: 16, 224: 16, 256: 16, 512: 20, 768: 26, 1024: 26, 1280: 22, 1536: 20, 1792: 26, 2048: 26, 2304: 20, 2560: 21, 2816: 26, 3072: 27, 3328: 26, 3584: 26, 3840: 26, 4096: 26}}
        

def test_grouped_gemm_correctness(
    tokens=1024,
    experts=1,
    topk=2,
    in_size=4096,
    out_size=12800,
    times=100,
    cfg_id=4,
):
    assert tokens*topk % experts == 0, "tokens*topk % experts != 0"
    torch.manual_seed(1234)

    # input output
    hidden_state = torch.ones(tokens*topk, in_size).cuda().half()
    out = torch.empty(tokens*topk, out_size).cuda().half()

    # w1
    w1_f32 = torch.ones(experts, in_size, out_size).cuda().uniform_(-1, 1).cuda()
    w1 = torch.empty(experts, in_size, out_size, dtype=torch.float8_e4m3fn).cuda()
    w1_scale = torch.empty(experts, dtype=torch.float).cuda()

    for i in range(experts):
        w1[i, :, :], w1_scale[i] = ops.scaled_fp8_quant(w1_f32.half()[i, :, :])

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