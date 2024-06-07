/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// DISABLE Pytorch CUDAExtension Flags
#undef __CUDA_NO_HALF_CONVERSIONS__ 
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__

#include "tensorrt_llm/common/workspace.h"
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#pragma GCC diagnostic pop

#include "tensorrt_llm/common/cudaUtils.h"
#include "moe_kernels.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

static constexpr int WARP_SIZE = 32;

namespace detail
{
// ========================== CUB Sorting things ====================================
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0)
    , num_bits_(sizeof(int) * 8)
{
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts)
    , num_bits_((int) log2(num_experts) + 1)
{
}

void CubKeyValueSorter::updateNumExperts(int const num_experts)
{
    num_experts_ = num_experts;
    num_bits_ = (int) log2(num_experts) + 1;
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts)
{
    int num_bits = static_cast<int>(log2(num_experts)) + 1;
    size_t required_storage = 0;
    int* null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(
        nullptr, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits);
    return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size, int const* keys_in, int* keys_out,
    int const* values_in, int* values_out, size_t const num_key_value_pairs, cudaStream_t stream)
{
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
    size_t actual_ws_size = workspace_size;

    TLLM_CHECK_WITH_INFO(expected_ws_size <= workspace_size,
        "[CubKeyValueSorter::run] The allocated workspace is too small to run this problem.");
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLeqTarget(T const* sorted_indices, int64_t const arr_length, T const target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] > target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

namespace detail
{
// TODO these are copied from CUTLASS because the cutlass version is missing __device__ decorator
template <class StrideIntT>
CUTLASS_HOST_DEVICE cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> make_cute_packed_stride(
    cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL)
{
    static_assert(std::is_integral_v<StrideIntT>,
        "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
    auto s_copy = s;
    cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
    return s_copy;
}

template <class StrideIntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> make_cute_packed_stride(
    cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL)
{
    static_assert(std::is_integral_v<StrideIntT>,
        "Stride must have an integral type so it can be set dynamically. Static strides not supported.");
    auto s_copy = s;
    cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
    return s_copy;
}

} // namespace detail

__device__ void computeHopperInputStrides(
    HopperGroupedGemmInput layout_info, int gemm_m, int gemm_n, int gemm_k, int64_t out_idx)
{
    layout_info.stride_a[out_idx] = detail::make_cute_packed_stride(
        HopperGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, cute::Int<1>{}));
    layout_info.stride_b[out_idx] = detail::make_cute_packed_stride(
        HopperGroupedGemmInput::StrideB{}, cute::make_shape(gemm_n, gemm_k, cute::Int<1>{}));
    if (layout_info.stride_c)
    {
        assert(false && "CUTLASS does not support a 1xN bias");
        //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
        layout_info.stride_c[out_idx] = detail::make_cute_packed_stride(
            HopperGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, cute::Int<1>{}));
    }
    layout_info.stride_d[out_idx] = detail::make_cute_packed_stride(
        HopperGroupedGemmInput::StrideD{}, cute::make_shape(gemm_n, gemm_m, cute::Int<1>{}));
}

template <class T, class WeightType>
__device__ void computeHopperInputPointers(HopperGroupedGemmInput layout_info, int64_t gemm_m, int64_t gemm_n,
    int64_t gemm_k, int num_tokens_before_expert, int64_t expert, T const* in, WeightType const* weights, T const* bias,
    HopperGroupedGemmInput::OutputTypeAdaptor_t<T>* output, int64_t const out_idx)
{
    // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
    layout_info.ptr_a[out_idx] = in + num_tokens_before_expert * gemm_k;

    // Each expert's weight matrix is a constant size NxK, with `expert` experts
    layout_info.ptr_b[out_idx] = weights + expert * (gemm_n * gemm_k);

    if (bias)
    {
        // Each expert's bias is a constant size N, with `expert` experts
        layout_info.ptr_c[out_idx] = bias + expert * gemm_n;
    }

    // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
    layout_info.ptr_d[out_idx] = output + num_tokens_before_expert * gemm_n;
}

// TODO Some of this setup could be cached
template <class T, class WeightType>
__global__ void computeStridesHopperKernel(int64_t const* total_rows_before_expert, HopperGroupedGemmInput layout_info,
    int64_t gemm_n, int64_t gemm_k, int64_t const num_experts, T const* in, WeightType const* weights,
    float const* fp8_dequant, T const* bias, typename HopperGroupedGemmInput::OutputTypeAdaptor_t<T>* output)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    auto const num_tokens_including_expert = total_rows_before_expert[expert];
    auto const num_tokens_before_expert = expert > 0 ? total_rows_before_expert[expert - 1] : 0;
    auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
    auto const gemm_m = num_tokens_to_expert;

    // M and N transposed since we are using the #tokens as the N dimension
    layout_info.shape_info.problem_shapes[expert]
        = HopperGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm_n, gemm_m, gemm_k);

    if (fp8_dequant)
    {
        layout_info.alpha_scale_ptr_array[expert] = fp8_dequant + expert;
    }

    assert(gemm_m <= INT32_MAX);
    assert(gemm_n <= INT32_MAX);
    assert(gemm_k <= INT32_MAX);
    computeHopperInputStrides(layout_info, gemm_m, gemm_n, gemm_k, expert);

    computeHopperInputPointers(
        layout_info, gemm_m, gemm_n, gemm_k, num_tokens_before_expert, expert, in, weights, bias, output, expert);
}

// ========================== Permutation things =======================================

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

template <typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsKernel(T const* unpermuted_input, T* permuted_output,
    int const* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const* num_dest_rows, int64_t const cols)
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    int64_t const expanded_dest_row = blockIdx.x;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0)
    {
        assert(expanded_dest_row <= INT32_MAX);
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = static_cast<int>(expanded_dest_row);
    }

    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows)
    {
        // Duplicate and permute rows
        int64_t const source_row = expanded_source_row % num_rows;

        T const* source_row_ptr = unpermuted_input + source_row * cols;
        T* dest_row_ptr = permuted_output + expanded_dest_row * cols;

        for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)
        {
            dest_row_ptr[tid] = source_row_ptr[tid];
        }
    }
}

template <typename T>
void expandInputRowsKernelLauncher(T const* unpermuted_input, T* permuted_output,
    int const* expanded_dest_row_to_expanded_source_row, int* expanded_source_row_to_expanded_dest_row,
    int64_t const num_rows, int64_t const* num_valid_tokens_ptr, int64_t const cols, int const k, cudaStream_t stream)
{
    int64_t const blocks = num_rows * k;
    int64_t const threads = std::min(cols, int64_t{1024});
    auto func = (num_valid_tokens_ptr != nullptr) ? expandInputRowsKernel<T, true> : expandInputRowsKernel<T, false>;
    func<<<blocks, threads, 0, stream>>>(unpermuted_input, permuted_output, expanded_dest_row_to_expanded_source_row,
        expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, cols);
}

enum class ScaleMode : int
{
    NO_SCALE = 0,
    DEFAULT = 1,
    RENORM_SCALE = 2,
};

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename T, typename OutputType, class GemmOutputType, ScaleMode SCALE_MODE, bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, T const* bias, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const cols,
    int64_t const k, int64_t const* num_valid_ptr)
{
    int64_t const original_row = blockIdx.x;
    int64_t const num_rows = gridDim.x;
    auto const offset = original_row * cols;
    OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;
    int64_t const num_valid = *num_valid_ptr;
    for (int tid = threadIdx.x; tid < cols; tid += blockDim.x)
    {
        float thread_output{0.f};
        float row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx)
        {
            int64_t const expanded_original_row = original_row + k_idx * num_rows;
            int64_t const expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            int64_t const k_offset = original_row * k + k_idx;
            float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE)
            {
                row_rescale = row_rescale + row_scale;
            }

            // Check after row sum has accumulated
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
            {
                continue;
            }

            auto const* expanded_permuted_rows_row_ptr = expanded_permuted_rows + expanded_permuted_row * cols;

            int64_t const expert_idx = expert_for_source_row[k_offset];

            T const* bias_ptr = bias + expert_idx * cols;
            float const bias_value = bias ? static_cast<float>(bias_ptr[tid]) : 0.f;

            float const row_value = static_cast<float>(expanded_permuted_rows_row_ptr[tid]);

            thread_output = static_cast<float>(thread_output) + row_scale * (row_value + bias_value);
        }

        if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || thread_output != 0.f))
        {
            assert(row_rescale != 0.f);
            thread_output = thread_output / row_rescale;
        }

        reduced_row_ptr[tid] = static_cast<OutputType>(thread_output);
    }
}

template <class T, class OutputType, class GemmOutputType = T>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, T const* bias, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const num_rows,
    int64_t const cols, int64_t const k, int64_t const* num_valid_ptr, MOEParallelismConfig parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    int64_t const blocks = num_rows;
    int64_t const threads = std::min(cols, int64_t{1024});

    // Only add bias on rank 0 for tensor parallelism
    bool const is_rank_0 = parallelism_config.tp_rank == 0;
    T const* bias_ptr = is_rank_0 ? bias : nullptr;

    bool const check_finished = num_valid_ptr != nullptr;

    ScaleMode renorm_scales = ScaleMode::DEFAULT;
    if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
    {
        renorm_scales = k == 1 ? ScaleMode::NO_SCALE : ScaleMode::RENORM_SCALE;
    }

    using FuncPtr = decltype(&finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::DEFAULT, false>);
    FuncPtr func_map[2][3] = {
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::NO_SCALE, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::DEFAULT, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::RENORM_SCALE, false>,
        },
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::NO_SCALE, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::DEFAULT, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleMode::RENORM_SCALE, true>,
        },
    };
    auto* const func = func_map[check_finished][int(renorm_scales)];
    func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, scales,
        expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k, num_valid_ptr);
}

// ============================== Gated Activation =================================

template <class T, class ActFn>
__global__ void doGatedActivationKernel(
    T* output, T const* gemm_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    ActFn fn{};
    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;
    for (int64_t i = tid; i < inter_size; i += blockDim.x)
    {
        auto fc1_value = static_cast<float>(gemm_result[i]);
        // BF16 isn't supported, use FP32 for activation function
        auto gate_value = static_cast<float>(gemm_result[i + inter_size]);
        float gate_act = fn(gate_value);
        output[i] = static_cast<T>(fc1_value * gate_act);
    }
}

template <class T>
void doGatedActivation(T* output, T const* gemm_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
    int64_t num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = std::min(inter_size, int64_t{1024});

    // TODO Instead of T use a vectored type if performance would benefit
    // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.
    auto* fn = activation_type == ActivationType::Swiglu
        ? &doGatedActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>
        : &doGatedActivationKernel<T, cutlass::epilogue::thread::GELU<float>>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

// ============================== Activation =================================

template <class T, class ActFn>
__global__ void doActivationKernel(T* output, HopperGroupedGemmInput::OutputTypeAdaptor_t<T> const* gemm_result,
    float const* fp8_quant, T const* bias_ptr, int64_t const* total_rows_before_expert_, int num_experts,
    int64_t inter_size, bool gated)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (token >= total_rows_before_expert_[num_experts - 1])
    {
        return;
    }

    size_t gated_mul = gated ? 2 : 1;
    size_t gated_off = gated ? inter_size : 0;

    ActFn fn{};
    gemm_result = gemm_result + token * inter_size * gated_mul;
    output = output + token * inter_size; // Aliases gemm_result for non-gated, non-fp8 cases

    int64_t expert = 0;
    if (bias_ptr)
    {
        // TODO this is almost certainly faster as a linear scan
        expert = findTotalEltsLeqTarget(total_rows_before_expert_, num_experts, (int64_t) token);
    }

    float const quant_scale = fp8_quant ? *fp8_quant : 1.f;

    if (bias_ptr)
    {
        bias_ptr = bias_ptr + expert * inter_size * gated_mul;
    }
    for (int64_t i = tid; i < inter_size; i += blockDim.x)
    {
        auto fc1_value = static_cast<float>(gemm_result[i + gated_off]);
        if (bias_ptr)
        {
            fc1_value += static_cast<float>(bias_ptr[i + gated_off]);
        }

        float gate_act = fn(fc1_value);

        if (gated)
        {
            gate_act *= static_cast<float>(gemm_result[i]) + (bias_ptr ? static_cast<float>(bias_ptr[i]) : 0.0f);
        }

        output[i] = static_cast<T>(gate_act * quant_scale);
    }
}

template <class T>
void doActivation(T* output, HopperGroupedGemmInput::OutputTypeAdaptor_t<T> const* gemm_result, float const* fp8_quant,
    T const* bias, int64_t const* total_rows_before_expert_, int num_experts, int64_t inter_size, int64_t num_tokens,
    ActivationType activation_type, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = std::min(inter_size, int64_t{1024});

    // TODO Instead of T use a vectored type if performance would benefit
    auto fn_list = std::array{
        &doActivationKernel<T, cutlass::epilogue::thread::GELU<float>>,    // Gelu
        &doActivationKernel<T, cutlass::epilogue::thread::ReLu<float>>,    // Relu
        &doActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>,    // Silu
        &doActivationKernel<T, cutlass::epilogue::thread::SiLu<float>>,    // Swiglu
        &doActivationKernel<T, cutlass::epilogue::thread::GELU<float>>,    // Geglu
        &doActivationKernel<T, cutlass::epilogue::thread::Identity<float>> // Identity
    };
    auto fn = fn_list[static_cast<int>(activation_type)];
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, fp8_quant, bias, total_rows_before_expert_, num_experts,
        inter_size, isGatedActivation(activation_type));
}

template <class T, class WeightType, class OutputType, class Enable>
std::vector<size_t> CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::getWorkspaceBufferSizes(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const num_experts_per_node, int const k, ActivationType activation_type) const
{
    size_t const num_moe_inputs = k * num_rows;
    size_t const permuted_elems = num_moe_inputs * hidden_size;
    size_t const interbuf_elems = num_moe_inputs * inter_size;
    size_t glu_inter_elems = 0;
    bool is_gated_activation = isGatedActivation(activation_type);
    if (is_gated_activation)
    {
        glu_inter_elems = interbuf_elems * 2;
    }
    else if (mayHaveDifferentGEMMOutputType())
    {
        // In this case we are using activation quantization, and some intermediate buffers will be unquantized
        // We need to have separate memory for these as we can no longer alias the output buffer for reuse
        glu_inter_elems = interbuf_elems;
    }
    size_t num_softmax_outs = 0;

    bool using_hopper = moe_gemm_runner_.supportsHopperSpecialisation();
    size_t const gemm_output_dtype = using_hopper ? sizeof(HopperGemmOutputType) : sizeof(T);

    bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        num_softmax_outs = num_rows * num_experts;
    }

    size_t const source_rows_size = num_moe_inputs * sizeof(int);
    size_t const permuted_rows_size = num_moe_inputs * sizeof(int);
    size_t const permuted_experts_size = num_moe_inputs * sizeof(int);
    size_t const permuted_data_size = permuted_elems * sizeof(T);
    size_t const total_rows_before_expert_size = num_experts_per_node * sizeof(int64_t);
    size_t const softmax_out_size = num_softmax_outs * sizeof(float);
    size_t const glu_inter_size = glu_inter_elems * gemm_output_dtype; // May be an intermediate type for quantization
    size_t const fc1_result_size = interbuf_elems * sizeof(T);         // Acitvation quantizes so back to sizeof(T)
    size_t const sorter_size = CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts);
    size_t const fc2_result_size = permuted_elems * gemm_output_dtype; // May be an intermediate type for quantization
    size_t const hopper_size = using_hopper ? HopperGroupedGemmInput::workspaceSize(num_experts_per_node) : 0;
    size_t const gemm_workspace_size = moe_gemm_runner_.calcMaxWorkspaceSize(num_experts_per_node);

    std::vector<size_t> workspace{source_rows_size, permuted_rows_size, permuted_experts_size, permuted_data_size,
        total_rows_before_expert_size, softmax_out_size, glu_inter_size,
        // These pointers reuse the same memory
        std::max(fc1_result_size, sorter_size), fc2_result_size, hopper_size, gemm_workspace_size};
    return workspace;
}

template <class T, class WeightType, class OutputType, class Enable>
size_t CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::getWorkspaceSize(int64_t const num_rows,
    int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const k,
    ActivationType activation_type, MOEParallelismConfig parallelism_config) const
{
    int const ep_size = parallelism_config.ep_size;
    TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of tp size");
    auto workspace = getWorkspaceBufferSizes(
        num_rows, hidden_size, inter_size, num_experts, num_experts / ep_size, k, activation_type);
    return tensorrt_llm::common::calculateTotalWorkspaceSize(workspace.data(), workspace.size());
}

template <class T, class WeightType, class OutputType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::configureWsPtrs(char* ws_ptr, int64_t const num_rows,
    int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const num_experts_per_node,
    int const k, ActivationType activation_type)
{
    auto ws_sizes = getWorkspaceBufferSizes(
        num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k, activation_type);

    std::vector<int8_t*> ws_sliced{(int8_t*) ws_ptr};
    for (auto size : ws_sizes)
    {
        ws_sliced.push_back(nextWorkspacePtr(ws_sliced.back(), size));
    }
    ws_sliced.pop_back();

    source_rows_ = (int*) ws_sliced[0];
    permuted_rows_ = (int*) ws_sliced[1];
    permuted_experts_ = (int*) ws_sliced[2];
    permuted_data_ = (T*) ws_sliced[3];

    total_rows_before_expert_ = (int64_t*) ws_sliced[4];

    softmax_out_ = nullptr;
    bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        softmax_out_ = (float*) ws_sliced[5];
    }

    glu_inter_result_ = (T*) ws_sliced[6];

    // These pointers are aliased. Since the sort ws can be overwritten after it is finished
    sorter_ws_ = (char*) ws_sliced[7];
    fc1_result_ = (T*) ws_sliced[7];

    fc2_result_ = (T*) ws_sliced[8];

    hopper_grouped_gemm_input_ = {};
    if (moe_gemm_runner_.isHopperSpecialised())
    {
        hopper_grouped_gemm_input_.configureWorkspace(ws_sliced[9], num_experts_per_node, ws_sliced[10], ws_sizes[10]);
    }
}

template <class T, class WeightType, class OutputType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::runMoe(void const* input_activations_void,
    float const* gating_output, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    ActivationType fc1_activation_type, void const* fc2_expert_weights_void, void const* fc2_expert_biases_void,
    QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts, int const k, char* workspace_ptr, void* final_output_void, bool const* finished,
    int64_t const active_rows, void* expert_scales_void, int* expanded_source_row_to_expanded_dest_row,
    int* expert_for_source_row, MOEParallelismConfig parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    static constexpr bool int_scales_required
        = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
    static constexpr bool fp8_scales_required
        = std::is_same<WeightType, __nv_fp8_e4m3>::value || std::is_same<WeightType, __nv_fp8_e5m2>::value;

    auto const* input_activations = static_cast<T const*>(input_activations_void);
    auto const* fc1_expert_weights = static_cast<WeightType const*>(fc1_expert_weights_void);
    auto const* fc1_expert_biases = static_cast<T const*>(fc1_expert_biases_void);
    auto const* fc2_expert_weights = static_cast<WeightType const*>(fc2_expert_weights_void);
    auto const* fc1_int_scales = static_cast<T const*>(quant_params.fc1_weight_scales);
    auto const* fc2_int_scales = static_cast<T const*>(quant_params.fc2_weight_scales);
    auto const* fc1_fp8_dequant = quant_params.dequant_fc1;
    auto const* fc2_fp8_quant = quant_params.quant_fc2;
    auto const* fc2_fp8_dequant = quant_params.dequant_fc2;
    auto const* fc2_expert_biases = static_cast<T const*>(fc2_expert_biases_void);
    auto* final_output = static_cast<OutputType*>(final_output_void);
    auto* expert_scales = static_cast<float*>(expert_scales_void);

    TLLM_CHECK(input_activations);
    TLLM_CHECK(gating_output);
    TLLM_CHECK(fc1_expert_weights);
    TLLM_CHECK(fc2_expert_weights);
    TLLM_CHECK(workspace_ptr);
    TLLM_CHECK(expert_scales);
    TLLM_CHECK(expanded_source_row_to_expanded_dest_row);
    TLLM_CHECK(expert_for_source_row);
    TLLM_CHECK(num_experts % parallelism_config.ep_size == 0);
    TLLM_CHECK_WITH_INFO(hidden_size >= 128 / cutlass::sizeof_bits<WeightType>::value,
        "Hidden size is too small to meet alignment requirements for MOE GEMM");
    TLLM_CHECK_WITH_INFO(hidden_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
        "Hidden size does not meet minimum alignment requirements for MOE GEMM");
    TLLM_CHECK_WITH_INFO(inter_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
        "Inter size does not meet minimum alignment requirements for MOE GEMM");

    // These values must fit into an int for building the source maps
    TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
    TLLM_CHECK_WITH_INFO(
        num_rows * num_experts <= std::numeric_limits<int>::max(), "Number of rows * num_experts is too large");
    TLLM_CHECK_WITH_INFO(k * num_experts <= std::numeric_limits<int>::max(), "k * num_experts is too large");

    if (int_scales_required)
    {
        TLLM_CHECK_WITH_INFO(
            fc1_int_scales != nullptr, "Weight scales expected but scale for first matmul is a null pointer");
        TLLM_CHECK_WITH_INFO(
            fc2_int_scales != nullptr, "Weight scales expected but scale for second matmul is a null pointer");

        TLLM_CHECK_WITH_INFO(fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
            "FP8 scales are provided for integer quantization");
    }
    else if (fp8_scales_required)
    {
        TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Bias is not supported with FP8");
        TLLM_CHECK_WITH_INFO(fc2_expert_biases == nullptr, "Bias is not supported with FP8");

        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC1 is a null pointer");
        TLLM_CHECK_WITH_INFO(fc2_fp8_quant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");

        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr && fc2_int_scales == nullptr, "Integer scales are provided for FP8 quantization");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_quant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    }

    int const num_experts_per_node = num_experts / parallelism_config.ep_size;
    int const start_expert = num_experts_per_node * parallelism_config.ep_rank;
    int const end_expert = start_expert + num_experts_per_node;

    configureWsPtrs(
        workspace_ptr, num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k, fc1_activation_type);
    topkGatingSoftmaxKernelLauncher(gating_output, finished, expert_scales, softmax_out_, expert_for_source_row,
        source_rows_, num_rows, num_experts, k, start_expert, end_expert, stream);

    sync_check_cuda_error();

    // We need to use the full num_experts because that is the sentinel value used by topk for disabled experts
    sorter_.updateNumExperts(num_experts);
    size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(sorter_.getWorkspaceSize(k * num_rows, num_experts));
    sorter_.run((void*) sorter_ws_, sorter_ws_size_bytes, expert_for_source_row, permuted_experts_, source_rows_,
        permuted_rows_, k * num_rows, stream);

    sync_check_cuda_error();

    bool const is_gated_activation = isGatedActivation(fc1_activation_type);
    bool const use_fused_moe = moe_gemm_runner_.isFusedGatedActivation(is_gated_activation, inter_size, hidden_size);
    size_t const fc1_out_size = ((!use_fused_moe) && is_gated_activation) ? inter_size * 2 : inter_size;

    // Upper bound on number of expanded rows
    int64_t const expanded_active_expert_rows = k * active_rows;
    computeTotalRowsBeforeExpert(
        permuted_experts_, expanded_active_expert_rows, num_experts_per_node, total_rows_before_expert_, stream);

    bool const needs_num_valid = finished || parallelism_config.ep_size > 1;
    int64_t const* num_valid_tokens_ptr
        = needs_num_valid ? total_rows_before_expert_ + num_experts_per_node - 1 : nullptr;
    expandInputRowsKernelLauncher(input_activations, permuted_data_, permuted_rows_,
        expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, hidden_size, k, stream);

    sync_check_cuda_error();

    bool const using_hopper = moe_gemm_runner_.isHopperSpecialised();
    HopperGroupedGemmInput hopper_input = hopper_grouped_gemm_input_;
    if (using_hopper)
    {
        bool has_different_gemm_output_type = using_hopper && mayHaveDifferentGEMMOutputType();
        auto* gemm_output = (has_different_gemm_output_type || is_gated_activation) ? glu_inter_result_
                                                                                    : static_cast<void*>(fc1_result_);

        hopper_input = computeStridesHopper(total_rows_before_expert_, hopper_input, fc1_out_size, hidden_size,
            num_experts_per_node, permuted_data_, fc1_expert_weights, fc1_fp8_dequant, nullptr,
            static_cast<HopperGemmOutputType*>(gemm_output), stream);
        sync_check_cuda_error();

        moe_gemm_runner_.moeGemm(permuted_data_, nullptr, nullptr, nullptr, total_rows_before_expert_, hopper_input,
            expanded_active_expert_rows, fc1_out_size, hidden_size, num_experts_per_node, false, stream);

        sync_check_cuda_error();

        doActivation<T>(fc1_result_, static_cast<HopperGemmOutputType const*>(gemm_output), fc2_fp8_quant,
            fc1_expert_biases, total_rows_before_expert_, num_experts_per_node, inter_size, num_rows * k,
            fc1_activation_type, stream);

        sync_check_cuda_error();
    }
    else if (!is_gated_activation)
    {
        moe_gemm_runner_.moeGemmBiasAct(permuted_data_, fc1_expert_weights, fc1_int_scales, fc1_expert_biases,
            fc1_result_, total_rows_before_expert_, HopperGroupedGemmInput{}, expanded_active_expert_rows, fc1_out_size,
            hidden_size, num_experts_per_node, fc1_activation_type, use_fused_moe, stream);

        sync_check_cuda_error();
    }
    else
    {
        // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
        ActivationType activation_type = (use_fused_moe) ? fc1_activation_type : ActivationType::Identity;
        T* gemm_result = (use_fused_moe) ? fc1_result_ : static_cast<T*>(glu_inter_result_);
        moe_gemm_runner_.moeGemmBiasAct(permuted_data_, fc1_expert_weights, fc1_int_scales, fc1_expert_biases,
            gemm_result, total_rows_before_expert_, HopperGroupedGemmInput{}, expanded_active_expert_rows, fc1_out_size,
            hidden_size, num_experts_per_node, activation_type, use_fused_moe, stream);

        sync_check_cuda_error();
        if (!use_fused_moe)
        {
            doGatedActivation<T>(fc1_result_, static_cast<T const*>(glu_inter_result_), num_valid_tokens_ptr,
                inter_size, num_rows * k, fc1_activation_type, stream);

            sync_check_cuda_error();
        }
    }

    sync_check_cuda_error();

    if (using_hopper)
    {
        hopper_input = computeStridesHopper(total_rows_before_expert_, hopper_input, hidden_size, inter_size,
            num_experts_per_node, fc1_result_, fc2_expert_weights, fc2_fp8_dequant, nullptr,
            static_cast<HopperGemmOutputType*>(fc2_result_), stream);
        sync_check_cuda_error();
    }

    moe_gemm_runner_.moeGemm(fc1_result_, fc2_expert_weights, fc2_int_scales, static_cast<T*>(fc2_result_),
        total_rows_before_expert_, hopper_input, expanded_active_expert_rows, hidden_size, inter_size,
        num_experts_per_node, false, stream);

    sync_check_cuda_error();

    if (using_hopper)
    {
        finalizeMoeRoutingKernelLauncher<T, OutputType, HopperGemmOutputType>(
            static_cast<HopperGemmOutputType const*>(fc2_result_), final_output, fc2_expert_biases, expert_scales,
            expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows, hidden_size, k,
            num_valid_tokens_ptr, parallelism_config, normalization_mode, stream);
    }
    else
    {
        finalizeMoeRoutingKernelLauncher<T, OutputType>(static_cast<T const*>(fc2_result_), final_output,
            fc2_expert_biases, expert_scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows,
            hidden_size, k, num_valid_tokens_ptr, parallelism_config, normalization_mode, stream);
    }

    sync_check_cuda_error();
}

template <class T, class WeightType, class OutputType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::computeTotalRowsBeforeExpert(int const* sorted_indices,
    int const total_indices, int const num_experts, int64_t* total_rows_before_expert, cudaStream_t stream)
{
    int const threads = std::min(1024, num_experts);
    int const blocks = (num_experts + threads - 1) / threads;

    computeTotalRowsBeforeExpertKernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, total_rows_before_expert);
}

template <class T, class WeightType, class OutputType, class Enable>
HopperGroupedGemmInput CutlassMoeFCRunner<T, WeightType, OutputType, Enable>::computeStridesHopper(
    int64_t const* total_rows_before_expert, HopperGroupedGemmInput layout_info, int64_t gemm_n, int64_t gemm_k,
    int const num_experts, T const* in, WeightType const* weights, float const* fp8_dequant, T const* bias,
    HopperGemmOutputType* output, cudaStream_t stream)
{
    if (!bias)
    {
        layout_info.ptr_c = nullptr;
        layout_info.stride_c = nullptr;
    }

    if (!fp8_dequant)
    {
        layout_info.alpha_scale_ptr_array = nullptr;
    }

    int const threads = std::min(1024, num_experts);
    int const blocks = (num_experts + threads - 1) / threads;

    computeStridesHopperKernel<<<blocks, threads, 0, stream>>>(
        total_rows_before_expert, layout_info, gemm_n, gemm_k, num_experts, in, weights, fp8_dequant, bias, output);

    return layout_info;
}

// ==================== Helper for getting load balanced routing for profiling ==================================

template <class T>
__global__ void initRoutingKernelDiagonal(void* data_void, int num_experts, int num_tokens, int k, int stride)
{
    assert(k == 1 || (stride % num_experts) != 0);
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens)
    {
        return;
    }
    T* data = reinterpret_cast<T*>(data_void) + token * num_experts;
    int start = token % num_experts;
    for (int i = 0; i < k; i++)
    {
        data[start] = T{1.f};
        start += stride;
        if (start >= num_experts) // Wrap
            start -= num_experts;
    }
}

/*
void makeLoadBalancedRoutingConfiguration(
    void* data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(type == nvinfer1::DataType::kFLOAT, "Routing configuration must be float");
    check_cuda_error(
        cudaMemsetAsync(data_void, 0x0, int64_t{num_experts} * int64_t{num_tokens} * sizeof(float), stream));

    int stride = tensorrt_llm::common::ceilDiv(num_experts, k);

    int blockDim = 256;
    int gridDim = tensorrt_llm::common::ceilDiv(num_tokens, blockDim);
    initRoutingKernelDiagonal<float><<<gridDim, blockDim, 0, stream>>>(data_void, num_experts, num_tokens, k, stride);

    sync_check_cuda_error();
}
*/

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;
#ifdef ENABLE_FP8
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
#endif
#endif

} // namespace tensorrt_llm::kernels
