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
    A = torch.randn([M, K]).cuda().bfloat16() - 0.5
    B = torch.randn([N, K]).cuda().bfloat16() - 0.5
    C = torch.randn([M, N]).cuda() - 0.5
    ref_C = torch.mm(A.float(), B.T.float()) + C
    cublas = pycublas.interop.CublasInterop()
    handle = cublas.cublasCreate_v2()
    cublas.cublasGemmEX(
        handle,
        1, 0,
        N, M, K,
        pycublas.constant.float32_one,
        B.data_ptr(), "CUDA_R_16BF", K,
        A.data_ptr(), "CUDA_R_16BF", K,
        pycublas.constant.float32_one,
        C.data_ptr(), "CUDA_R_32F", N,
        "CUBLAS_COMPUTE_32F",
        "CUBLAS_GEMM_DEFAULT_TENSOR_OP"
    )
    cublas.cublasDestroy_v2(handle)
    torch.testing.assert_close(ref_C, C.float(), rtol=1.6e-2, atol=1e-4)


def test_simple_gemm_f16_fp32():
    M = 1024
    K = 1024 * 2
    N = 1024 // 2
    A = torch.randn([M, K]).cuda().half() - 0.5
    B = torch.randn([N, K]).cuda().half() - 0.5
    C = torch.randn([M, N]).cuda() - 0.5
    ref_C = torch.mm(A.float(), B.T.float()) + C
    cublas = pycublas.interop.CublasInterop()
    handle = cublas.cublasCreate_v2()
    cublas.cublasGemmEX(
        handle,
        1, 0,
        N, M, K,
        pycublas.constant.float32_one,
        B.data_ptr(), "CUDA_R_16F", K,
        A.data_ptr(), "CUDA_R_16F", K,
        pycublas.constant.float32_one,
        C.data_ptr(), "CUDA_R_32F", N,
        "CUBLAS_COMPUTE_32F",
        "CUBLAS_GEMM_DEFAULT_TENSOR_OP"
    )
    cublas.cublasDestroy_v2(handle)
    torch.testing.assert_close(ref_C, C.float(), rtol=1e-3, atol=1e-5)