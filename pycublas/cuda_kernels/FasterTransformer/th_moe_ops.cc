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
#define BUILD_CUTLASS_MOE

#include <iostream>
#include <vector>
#include <torch/extension.h>

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
    T *get_ptr(Tensor t)
    {
        return (T *)t.data_ptr();
    }

    template <typename T, typename WeightType>
    Tensor grouped_gemm_helper(Tensor activations,
                               Tensor weights,
                               Tensor weight_scales,
                               Tensor total_rows_before_expert)
    {
        const at::ScalarType _st = activations.scalar_type();
        auto stream = at::cuda::getCurrentCUDAStream().stream();
        const int num_rows = activations.size(0);
        const int64_t gemm_k = activations.size(1);
        const int64_t gemm_n = weights.size(-1);
        const int64_t experts = weights.size(0);

        // mxk, kxn -> mxn
        assert(activations.size(1) == weights.size(2));
        assert(experts == weight_scales.size(0));
        assert(total_rows_before_expert.dtype() == torch::kInt64);

        auto res = torch::zeros({num_rows, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        T *act_ptr = get_ptr<T>(activations);
        WeightType *wt_ptr = get_ptr<WeightType>(weights);
        T *weight_scale_ptr = get_ptr<T>(weight_scales);
        T *res_ptr = get_ptr<T>(res);
        int64_t *total_rows_before_expert_ptr = get_ptr<int64_t>(total_rows_before_expert);

        fastertransformer::MoeGemmRunner<T, WeightType> moe_gemm_runner;
        moe_gemm_runner.moe_gemm(act_ptr,
                                wt_ptr,
                                weight_scale_ptr,
                                res_ptr,
                                total_rows_before_expert_ptr,
                                num_rows,
                                gemm_n,
                                gemm_k,
                                experts,
                                stream);
        return res;
    }

    Tensor grouped_gemm(Tensor activations,
                        Tensor weights,
                        Tensor weight_scales,
                        Tensor total_rows_before_expert)
    {
        const at::ScalarType _st = activations.scalar_type();
        CHECK_INPUT(activations, _st);
        CHECK_INPUT(weight_scales, _st);
        CHECK_INPUT(total_rows_before_expert, torch::kInt64);

        switch (_st)
        {
        case at::ScalarType::Half:
        {
            if (weights.scalar_type() == torch::kInt8)
            {
                CHECK_INPUT(weights, torch::kInt8);
                return grouped_gemm_helper<__half, uint8_t>(
                    activations, weights, weight_scales, total_rows_before_expert);
            }
            else
            {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
        default:
            TORCH_CHECK(false, "Incompatible tensor type for grouped gemm bias");
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("grouped_gemm", &torch_ext::grouped_gemm, "Grouped GEMM with bias");
}