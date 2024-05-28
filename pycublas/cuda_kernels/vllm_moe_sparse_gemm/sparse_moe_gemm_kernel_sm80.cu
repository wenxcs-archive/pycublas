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
  using namespace cute;

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
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

  // activation: M x K
  // weight: E, N x K
  // output: M x K
  // w_scales: E x 1
  // topk_weight: M x topk
  // sorted_token_ids: ? block_m_size x blocks
  // expert_ids: blocks
  // num_tokens_post_padded: ? int
  // num_valid_tokens: int
  // block_m_size: int
  // stream: int64_t

  auto M = activation.size(0);
  auto K = activation.size(1);
  auto N = weight.size(1);
  auto E = weight.size(0);

  TORCH_CHECK(K == weight.size(2), "weight shape mismatch");

  auto prob_shape = make_shape(M, N, K); // (M, N, K)

  auto ldA = K;
  auto ldB = K;
  auto ldC = N;

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

  auto A = activation.data_ptr();
  auto B = weight.data_ptr();
  auto C = output.data_ptr();
  float alpha = 1.0;
  float beta = 0.0;

  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  dim3 dimBlock(size(mmaC));
  gemm_device<<<dimGrid, dimBlock, 0, (CUstream_st*)stream>>>(prob_shape, cta_tiler,
                                                A, dA, sA, copyA,
                                                B, dB, sB, copyB,
                                                C, dC, sC, mmaC,
                                                alpha, beta);
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("vllm_sparse_moe_gemm_kernel", &cuda_kernels::vllm_sparse_moe_gemm_kernel, "");
}