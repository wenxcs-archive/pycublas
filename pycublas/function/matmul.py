import torch
import pycublas

cublas = pycublas.interop.CublasInterop()


class matmul_nt_bf16_fp32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, C=None):
        ctx.save_for_backward(A, B)
        assert A.dtype == torch.bfloat16
        assert B.dtype == torch.bfloat16
        assert A.device == B.device
        assert A.device.type == "cuda"
        M = A.shape[0]
        N = B.shape[0]
        K = A.shape[1]
        assert K == B.shape[1]
        if C is None:
            C = torch.empty(
                A.shape[0], B.shape[0], dtype=torch.float32, device=A.device
            )
        else:
            assert C.dtype == torch.float32
            assert C.device == A.device
            assert C.shape == (M, N)
        handle = torch.cuda.current_blas_handle()

        cublas.cublasGemmEX(
            handle,
            1,
            0,
            N,
            M,
            K,
            pycublas.constant.float32_one,
            B.data_ptr(),
            "CUDA_R_16BF",
            K,
            A.data_ptr(),
            "CUDA_R_16BF",
            K,
            pycublas.constant.float32_one,
            C.data_ptr(),
            "CUDA_R_32F",
            N,
            "CUBLAS_COMPUTE_32F",
            "CUBLAS_GEMM_DEFAULT_TENSOR_OP",
        )

        return C
