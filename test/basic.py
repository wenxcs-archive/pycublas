import pycublas
import cupy
import torch
from statistics import mean

def test_basic():
    cublas = pycublas.interop.CublasInterop()
    h = cublas.cublasCreate_v2()
    cublas.cublasDestroy_v2(h)

def test_simple_gemm_bf16_fp32():
    M = 1024
    K = 1024 * 2
    N = 1024 // 2
    A = torch.randn([M, K]).cuda().bfloat16() * 2.0 - 1.0
    B = torch.randn([K, N]).cuda().bfloat16() * 2.0 - 1.0
    C = torch.randn([M, N]).cuda()
    ref_C = torch.mm(A.float(), B.float()) + C
    B = B.T
    cublas = pycublas.interop.CublasInterop()
    handle = cublas.cublasCreate_v2()
    cublas.cublasGemmEX(
        handle,
        1, 0,
        N, M, K,
        pycublas.constant.bfloat16_one,
        B.data_ptr(), "CUDA_R_16BF", K,
        A.data_ptr(), "CUDA_R_16BF", K,
        pycublas.constant.bfloat16_one,
        C.data_ptr(), "CUDA_R_32F", N,
        "CUBLAS_COMPUTE_32F_FAST_16BF",
        "CUBLAS_GEMM_DEFAULT_TENSOR_OP"
    )
    cublas.cublasDestroy_v2(handle)
    torch.testing.assert_close(ref_C, C, rtol=1.6e-2, atol=1e-4)