/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <torch/extension.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


namespace cuda_kernels{
  template <class ProblemShape, class CtaTiler,
            class TA, class AStride, class ASmemLayout, class TiledCopyA,
            class TB, class BStride, class BSmemLayout, class TiledCopyB,
            class TC, class CStride, class CSmemLayout, class TiledMma,
            class Alpha, class Beta>
  __global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void
  sparse_gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
                     half * A, int dA, ASmemLayout sA_layout, TiledCopyA copy_a,
                     uint8_t *B, int dB, BSmemLayout sB_layout, TiledCopyB copy_b,
                     float *C, CStride dC, CSmemLayout, TiledMma mma,
                     float alpha, float beta, half *b_scale, half *topk_weight,
                     int32_t *sorted_token_ids, int32_t *expert_ids,
                     int32_t *num_tokens_post_padded, int32_t num_valid_tokens)
  {
    using namespace cute;
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shape_MNK), dA); // (M,K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shape_MNK), dB); // (N,K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shape_MNK), dC); // (M,N)
  }

  void sparse_gemm_nt(int m, int n, int k,
                      float alpha,
                      half *A, int ldA,
                      uint8_t *B, int ldB,
                      float beta,
                      float *C, int ldC,
                      cudaStream_t stream = 0)
  {
    using namespace cute;

    // Define shapes (dynamic)
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K); // (M, N, K)

    // Define NT strides (mixed)
    auto dA = make_stride(Int<1>{}, ldA); // (dM, dK)
    auto dB = make_stride(Int<1>{}, ldB); // (dN, dK)
    auto dC = make_stride(Int<1>{}, ldC); // (dM, dN)

    // Define CTA tile sizes (static)
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
    auto bP = Int<3>{};                      // Pipeline

    // Define the smem layouts (static)
    auto sA = make_layout(make_shape(bM, bK, bP)); // (m,k,p) -> smem_idx; m-major
    auto sB = make_layout(make_shape(bN, bK, bP)); // (n,k,p) -> smem_idx; n-major
    auto sC = make_layout(make_shape(bM, bN));     // (m,n) -> smem_idx; m-major

    // Define the thread layouts (static)

    TiledCopy copyA = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint64_t>, half>{},
                                      Layout<Shape<_32, _8>>{}, // Thr layout 32x8 m-major
                                      Layout<Shape<_4, _1>>{}); // Val layout  4x1 m-major
    TiledCopy copyB = make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<uint32_t>, uint8_t>{},
                                      Layout<Shape<_32, _8>>{}, // Thr layout 32x8 n-major
                                      Layout<Shape<_4, _1>>{}); // Val layout  4x1 n-major

    TiledMMA mmaC = make_tiled_mma(UniversalFMA<uint8_t, half, uint8_t>{},
                                   Layout<Shape<_16, _16, _1>>{}); // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  /*
  sparse_gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta, nullptr,
       nullptr, nullptr, nullptr,
       nullptr, 0
      );
  */
}

void vllm_sparse_moe_gemm_kernel(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor output,
    torch::Tensor w_scales,
    torch::Tensor topk_weight,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int num_valid_tokens,
    int block_m_size,
    int64_t stream)
{
  TORCH_CHECK(activation.is_cuda(), "activation must be a CUDA tensor");
  TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
  TORCH_CHECK(w_scales.is_cuda(), "w_scales must be a CUDA tensor");
  TORCH_CHECK(topk_weight.is_cuda(), "topk_weight must be a CUDA tensor");
  TORCH_CHECK(sorted_token_ids.is_cuda(), "sorted_token_ids must be a CUDA tensor");
  TORCH_CHECK(expert_ids.is_cuda(), "expert_ids must be a CUDA tensor");
  TORCH_CHECK(num_tokens_post_padded.is_cuda(), "num_tokens_post_padded must be a CUDA tensor");

  TORCH_CHECK(activation.dtype() == torch::kHalf, "activation must be half");
  TORCH_CHECK(weight.dtype() == torch::kUInt8, "weight must be fp8(uint8)");
  TORCH_CHECK(w_scales.dtype() == torch::kHalf, "w_scales must be half");
  TORCH_CHECK(topk_weight.dtype() == torch::kHalf, "topk_weight must be half");
  TORCH_CHECK(sorted_token_ids.dtype() == torch::kInt32, "sorted_token_ids must be int32");
  TORCH_CHECK(expert_ids.dtype() == torch::kInt32, "expert_ids must be int32");
  TORCH_CHECK(num_tokens_post_padded.dtype() == torch::kInt32, "num_tokens_post_padded must be int32");

  auto M = activation.size(0);
  auto H = activation.size(1);
  auto I = weight.size(1);
  auto E = weight.size(0);

  TORCH_CHECK(H == weight.size(2), "weight shape mismatch");

  sparse_gemm_nt(M, I, H,
                 1.0f,
                 (half*)activation.data_ptr(),
                 H,
                 (uint8_t*)weight.data_ptr(),
                 H,
                 0.0f,
                 (float*)output.data_ptr(),
                 I,
                 (cudaStream_t)stream);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("vllm_sparse_moe_gemm_kernel", &cuda_kernels::vllm_sparse_moe_gemm_kernel, "");
}