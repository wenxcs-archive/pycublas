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

struct gemm_config{
  static constexpr int tile_m = 128;
  static constexpr int tile_n = 128;
  static constexpr int tile_k = 32;

  static constexpr auto bM = Int<tile_m>{};
  static constexpr auto bN = Int<tile_n>{};
  static constexpr auto bK = Int<tile_k>{};
  static constexpr auto cta_tiler = make_shape(bM, bN, bK); // (BLK_M, BLK_N, BLK_K)
  static constexpr auto bP = Int<3>{};                      // Pipeline

  static constexpr TiledMMA mmaC = make_tiled_mma(UniversalFMA<half,half,uint8_t>{},
                                 Layout<Shape<_16,_16,_1>>{});  // 16x16x1 TiledMMA
  
  static constexpr int expert_num = Int<16>{};
};

template<class gemm_config>
__global__ void moe_gemm(
            half* a_ptr, uint8_t* b_ptr, half* c_ptr,
            int m, int n, int k, 
            half* w_scales_ptr,
            half* topk_weight_ptr,
            uint32_t * sorted_token_ids_ptr,
            uint32_t * expert_ids_ptr,
            uint32_t* num_token_post_padded,
            uint32_t num_valid_tokens)
{
  // Get block of activation information:
  auto a_block_id = blockIdx.x;
  // sorted_token_ids_ptr[a_block_id * tile_m :(a_block_id + 1) * tile_m]
  //  -> a_ptr[sorted_token_ids_ptr[a_block_id * tile_m :(a_block_id + 1) * tile_m], :]
  auto a_block_expert_id = expert_ids_ptr[a_block_id];
  // b_ptr[a_block_expert_id, :, :]
  auto prob_shape = make_shape(m, n, k);
  auto b_block_id = blockIdx.x;
  int expert_num = 16;
  auto thr_id = threadIdx.x;

  Tensor mA = make_tensor(make_gmem_ptr(a_ptr), make_shape(m,k), make_stride(k, 1));
  Tensor gA = local_tile(mA, make_tile(gemm_config::bM, gemm_config::bK), make_coord(a_block_id, _));

  Tensor mB = make_tensor(make_gmem_ptr(b_ptr), make_shape(expert_num, n, k ), make_stride(n, k, 1));
  Tensor gB = local_tile(mB, make_tile(gemm_config::bN, gemm_config::bK), make_coord(a_block_expert_id, b_block_id, _));

  gemm_config::mmaC mma;
  auto thr_mma = mma.get_slice(thr_id);
  auto tA = mma.partition_A(gA(_, _))
  auto tB = mma.partition_B(gB(_, _))

  auto g2s = U
}

void vllm_sparse_moe_gemm_kernel(
    torch::Tensor output,
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor w_scales,
    torch::Tensor topk_weight,
    torch::Tensor sorted_token_ids,
    torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_padded,
    int num_valid_tokens,
    int that_block_m_size,
    int64_t stream)
{
  auto M = activation.size(0);
  auto K = activation.size(1);
  auto N = weight.size(1);
  auto E = weight.size(0);

  dim3 dimGrid(size(ceil_div(sorted_token_ids.size(0), gemm_config::bM)),
               size(ceil_div(N, gemm_config::bN)));
  dim3 dimBlock(size(gemm_config::mmaC));

  moe_gemm<gemm_config><<<dimGrid, dimBlock, 0, (CUstream_st*)stream>>>(
    (half*)activation.data_ptr(),
    (uint8_t*)weight.data_ptr(),
    (half*)output.data_ptr(),
    M, N, K,
    (half*)w_scales.data_ptr(),
    (half*)topk_weight.data_ptr(),
    (uint32_t*)sorted_token_ids.data_ptr(),
    (uint32_t*)expert_ids.data_ptr(),
    (uint32_t*)num_tokens_post_padded.data_ptr(),
    num_valid_tokens
  );
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("vllm_sparse_moe_gemm_kernel", &cuda_kernels::vllm_sparse_moe_gemm_kernel, "");
}