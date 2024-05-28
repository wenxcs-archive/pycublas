import torch
import numpy as np
import pycublas
import pycublas.vllm_moe_sparse_gemm as vllm_moe


def test_vllm_moe_f16xf8_ampere():
    torch.manual_seed(12345)
    M = 32
    I = 6400
    H = 4096
    E = 16
    topk = 2
    tile_m = 32

    sorted_token_ids = torch.from_numpy(np.array(range(0, 64))).cuda().int()
    expert_id = torch.from_numpy(np.array([0, 1])).cuda().int()
    tokens_after_padded = torch.from_numpy(np.array([64])).cuda().int()
    num_valid_tokens = 64

    act = torch.randn(M * 2, 4096).cuda().half()
    weight = (torch.ones(E, I * 2, H) * 8).to(torch.uint8).cuda()
    wscale = torch.ones(E).cuda().half()
    outp = torch.empty(M * 2, I * 2, device=act.device, dtype=act.dtype)
    topk_weight = torch.ones(M * 2).cuda().half()

    vllm_moe.vllm_sparse_moe_gemm_kernel(
        act, weight, outp, wscale, topk_weight, sorted_token_ids, expert_id,
        tokens_after_padded, num_valid_tokens, tile_m, 0
    )
