/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define ENABLE_BF16

#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "moe_gemm_kernels.h"
#include "utils/cuda_bf16_wrapper.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

#define CHECK_INPUT(x, y) TORCH_CHECK(x.scalar_type() == y, #x " must be of type " #y)

namespace torch_ext
{
    template <typename T>
    T* get_ptr(Tensor t)
    {
        return (T*)t.data_ptr();
    }

    template <typename T, typename WeightType>
    Tensor grouped_gemm_bias_helper(Tensor activations,
                                    Tensor weights,
                                    Tensor weight_scales,
                                    Tensor biases,
                                    Tensor rows_per_expert,
                                    fastertransformer::ActivationType activation_type)
    {
        const at::ScalarType _st = activations.scalar_type();
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const int num_rows = activations.size(0);
        const int64_t gemm_k = activations.size(1);
        const bool is_packed_int4s = weights.size(-1) == weight_scales.size(-1) / 2;
        const int64_t gemm_n = is_packed_int4s ? 2 * weights.size(-1) : weights.size(-1);
        const int64_t experts = weights.size(0);

        auto res = torch::zeros({num_rows, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        auto total_rows_before_expert =
            torch::zeros({experts}, torch::dtype(torch::kInt64).device(torch::kCUDA).requires_grad(false));
        int64_t *total_rows_before_expert_ptr = get_ptr<int64_t>(total_rows_before_expert);

        int *rows_per_expert_ptr = get_ptr<int>(rows_per_expert);

        std::vector<int> rows_per_expert_h(experts);
        cudaError_t result =
            cudaMemcpy(rows_per_expert_h.data(), rows_per_expert_ptr, sizeof(int) * experts, cudaMemcpyDeviceToHost);
        TORCH_CHECK(result == cudaSuccess, "First memcpy failed");

        std::vector<int64_t> total_rows_before_expert_h(experts);
        for (int expert = 0; expert < experts; ++expert)
        {
            const int64_t last_row_for_prev_expert = expert == 0 ? 0 : total_rows_before_expert_h[expert - 1];
            total_rows_before_expert_h[expert] = last_row_for_prev_expert + rows_per_expert_h[expert];
        }
        result = cudaMemcpy(total_rows_before_expert_ptr,
                            total_rows_before_expert_h.data(),
                            sizeof(int64_t) * experts,
                            cudaMemcpyHostToDevice);
        TORCH_CHECK(result == cudaSuccess, "Second memcpy failed");

        T *act_ptr = get_ptr<T>(activations);
        T *bias_ptr = get_ptr<T>(biases);
        T *res_ptr = get_ptr<T>(res);
        T *weight_scale_ptr = get_ptr<T>(weight_scales);

        fastertransformer::MoeGemmRunner<T, WeightType> moe_gemm_runner;
        WeightType *wt_ptr = get_ptr<WeightType>(weights);

        moe_gemm_runner.moe_gemm_bias_act(act_ptr,
                                          wt_ptr,
                                          weight_scale_ptr,
                                          bias_ptr,
                                          res_ptr,
                                          total_rows_before_expert_ptr,
                                          num_rows,
                                          gemm_n,
                                          gemm_k,
                                          experts,
                                          activation_type,
                                          stream);

        return res;
    }

    Tensor grouped_gemm_bias(Tensor activations,
                             Tensor weights,
                             Tensor weight_scales,
                             Tensor biases,
                             Tensor rows_per_expert,
                             std::string activation_type_str)
    {

        const at::ScalarType _st = activations.scalar_type();
        CHECK_INPUT(activations, _st);
        CHECK_INPUT(biases, _st);
        CHECK_INPUT(weight_scales, _st);
        CHECK_INPUT(rows_per_expert, torch::kInt32);

        const bool is_packed_int4s = weights.size(-1) == weight_scales.size(-1) / 2;

        fastertransformer::ActivationType activation_type = fastertransformer::ActivationType::InvalidType;
        if (activation_type_str == "identity")
        {
            activation_type = fastertransformer::ActivationType::Identity;
        }
        else
        {
            activation_type = fastertransformer::getActivationType(activation_type_str);
        }

        switch (_st)
        {
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
        {
            if (weights.scalar_type() == torch::kInt8 && !is_packed_int4s)
            {
                CHECK_INPUT(weights, torch::kInt8);
                return grouped_gemm_bias_helper<__nv_bfloat16, uint8_t>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else
            {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
#endif
        default:
            TORCH_CHECK(false, "Incompatible tensor type for grouped gemm bias");
        }
    }

    TORCH_LIBRARY(moe_unit_ops, m)
    {
        m.def("grouped_gemm_bias", grouped_gemm_bias);
    }
} // namespace torch_ext