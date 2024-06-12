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
// DISABLE Pytorch CUDAExtension Flags
#undef __CUDA_NO_HALF_CONVERSIONS__ 
#undef __CUDA_NO_HALF_OPERATORS__
#undef __CUDA_NO_BFLOAT16_CONVERSIONS__
#undef __CUDA_NO_HALF2_OPERATORS__
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
#include "moe_kernels.h"
#include "cutlass_preprocessors.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

#define CHECK_INPUT(x, y) TORCH_CHECK(x.scalar_type() == y, #x " must be of type " #y)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")


template <typename T>
T *get_ptr(Tensor t)
{
    return (T *)t.data_ptr();
}

namespace torch_ext
{
    namespace ft = fastertransformer;

    ft::QuantType get_ft_quant_type(torch::ScalarType quant_type)
    {
        if (quant_type == torch::kInt8) {
            return ft::QuantType::INT8_WEIGHT_ONLY;
        }
    #ifdef TORCH_IS_AT_LEAST_v190
        else if (quant_type == at::ScalarType::QUInt4x2) {
            return ft::QuantType::PACKED_INT4_WEIGHT_ONLY;
        }
    #endif
        else {
            TORCH_CHECK(false, "Invalid quantization type");
        }
    }

    void check_quant_type_allowed(torch::ScalarType quant_type)
    {
    #ifdef TORCH_IS_AT_LEAST_v190
        TORCH_CHECK(quant_type == torch::kInt8 || quant_type == at::ScalarType::QUInt4x2,
                    "Must be int4 or int8 quantization");
    #else
        TORCH_CHECK(quant_type == torch::kInt8, "Must be int8 quantization");
    #endif
    }

    std::vector<Tensor>
    symmetric_quantize_helper(Tensor weight, torch::ScalarType quant_type, bool return_unprocessed_quantized_tensor)
    {
        CHECK_CPU(weight);
        CHECK_CONTIGUOUS(weight);
        TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
        TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3");

        auto _st = weight.scalar_type();
        TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16 || _st == torch::kBFloat16,
                    "Invalid datatype. Weight must be FP16 or BF16");
        check_quant_type_allowed(quant_type);
        ft::QuantType ft_quant_type = get_ft_quant_type(quant_type);

        const size_t num_experts = weight.dim() == 2 ? 1 : weight.size(0);
        const size_t num_rows    = weight.size(-2);
        const size_t num_cols    = weight.size(-1);

        const size_t bits_in_type      = get_bits_in_quant_type(ft_quant_type);
        const size_t bytes_per_out_col = num_cols * bits_in_type / 8;

        const size_t input_mat_size     = num_rows * num_cols;
        const size_t quantized_mat_size = num_rows * bytes_per_out_col;

        std::vector<long int> quantized_weight_shape;
        std::vector<long int> scale_shape;
        if (weight.dim() == 2) {
            quantized_weight_shape = {long(num_rows), long(bytes_per_out_col)};
            scale_shape            = {long(num_cols)};
        }
        else if (weight.dim() == 3) {
            quantized_weight_shape = {long(num_experts), long(num_rows), long(bytes_per_out_col)};
            scale_shape            = {long(num_experts), long(num_cols)};
        }
        else {
            TORCH_CHECK(false, "Invalid weight dimension. Weight must have dim 2 or 3");
        }

        Tensor unprocessed_quantized_weight =
            torch::empty(quantized_weight_shape, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

        Tensor processed_quantized_weight = torch::empty_like(unprocessed_quantized_weight);

        Tensor scales = torch::empty(scale_shape, torch::dtype(weight.dtype()).device(torch::kCPU).requires_grad(false));

        int8_t* unprocessed_quantized_weight_ptr = get_ptr<int8_t>(unprocessed_quantized_weight);
        int8_t* processed_quantized_weight_ptr   = get_ptr<int8_t>(processed_quantized_weight);

        if (weight.scalar_type() == at::ScalarType::Float) {
            ft::symmetric_quantize<float, float>(processed_quantized_weight_ptr,
                                                unprocessed_quantized_weight_ptr,
                                                get_ptr<float>(scales),
                                                get_ptr<const float>(weight),
                                                {num_experts, num_rows, num_cols},
                                                ft_quant_type);
        }
        else if (weight.scalar_type() == at::ScalarType::Half) {
            ft::symmetric_quantize<half, half>(processed_quantized_weight_ptr,
                                            unprocessed_quantized_weight_ptr,
                                            get_ptr<half>(scales),
                                            get_ptr<const half>(weight),
                                            {num_experts, num_rows, num_cols},
                                            ft_quant_type);
        }
    #ifdef ENABLE_BF16
        else if (weight.scalar_type() == at::ScalarType::BFloat16) {
            ft::symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(processed_quantized_weight_ptr,
                                                                unprocessed_quantized_weight_ptr,
                                                                get_ptr<__nv_bfloat16>(scales),
                                                                get_ptr<const __nv_bfloat16>(weight),
                                                                {num_experts, num_rows, num_cols},
                                                                ft_quant_type);
        }
    #endif
        else {
            TORCH_CHECK(false, "Invalid datatype. Weight must be BF16/FP16");
        }

        if (return_unprocessed_quantized_tensor) {
            return std::vector<Tensor>{unprocessed_quantized_weight, processed_quantized_weight, scales};
        }

        return std::vector<Tensor>{processed_quantized_weight, scales};
    }

    // Same as symmetric_quantize_last_axis_of_batched_matrix but returns a tuple of:
    // (unprocessed_quantized_weights, preprocessed_quantized_weights, scales)
    // Exposed mainly for testing, so that the unprocessed weights can be passed to torch functions.
    std::vector<Tensor> _symmetric_quantize_last_axis_of_batched_matrix(Tensor weight, torch::ScalarType quant_type)
    {
        return symmetric_quantize_helper(weight, quant_type, true);
    }

    Tensor preprocess_weights_for_mixed_gemm(Tensor row_major_quantized_weight, torch::ScalarType quant_type)
    {
        auto _st = row_major_quantized_weight.scalar_type();
        CHECK_CPU(row_major_quantized_weight);
        CHECK_CONTIGUOUS(row_major_quantized_weight);
        TORCH_CHECK(_st == torch::kInt8, "Quantized tensor must be int8 dtype");
        check_quant_type_allowed(quant_type);
        TORCH_CHECK(row_major_quantized_weight.dim() == 2 || row_major_quantized_weight.dim() == 3,
                    "Invalid dim. The dim of weight should be 2 or 3");

        ft::QuantType ft_quant_type      = get_ft_quant_type(quant_type);
        const size_t  bits_in_quant_type = get_bits_in_quant_type(ft_quant_type);

        const size_t num_experts = row_major_quantized_weight.dim() == 2 ? 1 : row_major_quantized_weight.size(0);
        const size_t num_rows    = row_major_quantized_weight.size(-2);
        const size_t num_cols    = (8 / bits_in_quant_type) * row_major_quantized_weight.size(-1);

        Tensor  processed_tensor = torch::zeros_like(row_major_quantized_weight);
        int8_t* input_byte_ptr   = get_ptr<int8_t>(row_major_quantized_weight);
        int8_t* output_byte_ptr  = get_ptr<int8_t>(processed_tensor);

        ft::preprocess_weights_for_mixed_gemm(
            output_byte_ptr, input_byte_ptr, {num_experts, num_rows, num_cols}, ft_quant_type);

        return processed_tensor;
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
        case at::ScalarType::Half:
        {
            if (weights.scalar_type() == torch::kInt8 && !is_packed_int4s)
            {
                CHECK_INPUT(weights, torch::kInt8);
                return grouped_gemm_bias_helper<half, uint8_t>(
                    activations, weights, weight_scales, biases, rows_per_expert, activation_type);
            }
            else
            {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(weights.scalar_type()));
                TORCH_CHECK(false, err_msg);
            }
            break;
        }
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
        {
            else if (weights.scalar_type() == torch::kInt8 && !is_packed_int4s)
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

    template <typename T, typename WeightType>
    Tensor run_moe_fc_helper(Tensor input_activations,
                             Tensor gating_output,
                             Tensor fc1_expert_weights,
                             Tensor fc1_scales,
                             Tensor fc1_expert_biases,
                             fastertransformer::ActivationType fc1_activation_type,
                             Tensor fc2_expert_weights,
                             Tensor fc2_scales,
                             Tensor fc2_expert_biases,
                             Tensor skip_layer,
                             Tensor finished,
                             const int active_rows,
                             const int k)
    {

        const int num_rows = input_activations.size(0);
        const int hidden_size = input_activations.size(1);
        const int inter_size = fc2_expert_weights.size(1);
        const int num_experts = gating_output.size(-1);
        auto stream = at::cuda::getCurrentCUDAStream().stream();

        T *input_act_ptr = get_ptr<T>(input_activations);
        T *gating_output_ptr = get_ptr<T>(gating_output);

        WeightType *fc1_expert_weights_ptr = get_ptr<WeightType>(fc1_expert_weights);
        static constexpr bool is_fp16_or_fp32 =
            std::is_same<WeightType, float>::value || std::is_same<WeightType, half>::value;
        static constexpr bool ignore_scales = is_fp16_or_fp32 || std::is_same<WeightType, __nv_bfloat16>::value;

        T *fc1_scales_ptr = ignore_scales ? nullptr : get_ptr<T>(fc1_scales);
        T *fc1_expert_biases_ptr = get_ptr<T>(fc1_expert_biases);

        WeightType *fc2_expert_weights_ptr = get_ptr<WeightType>(fc2_expert_weights);
        T *fc2_scales_ptr = ignore_scales ? nullptr : get_ptr<T>(fc2_scales);
        T *fc2_expert_biases_ptr = get_ptr<T>(fc2_expert_biases);

        T *skip_layer_ptr = get_ptr<T>(skip_layer);
        bool *finished_ptr = get_ptr<bool>(finished);

        fastertransformer::CutlassMoeFCRunner<T, WeightType> moe_runner;
        long int bytes = moe_runner.getWorkspaceSize(num_rows, hidden_size, inter_size, num_experts, k);
        auto workspace_tensor = torch::empty({bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));
        char *workspace_ptr = get_ptr<char>(workspace_tensor);

        const at::ScalarType _st = input_activations.scalar_type();
        auto fc2_output =
            torch::empty({k * num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        T *fc2_output_ptr = get_ptr<T>(fc2_output);

        auto expert_scales = torch::empty({num_rows, k}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        T *expert_scales_ptr = get_ptr<T>(expert_scales);

        auto expanded_source_row_to_expanded_dest_row =
            torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
        int *expanded_source_row_to_expanded_dest_row_ptr = get_ptr<int>(expanded_source_row_to_expanded_dest_row);

        auto expert_for_source_row =
            torch::empty({num_rows, k}, torch::dtype(torch::kInt32).device(torch::kCUDA).requires_grad(false));
        int *expert_for_source_row_ptr = get_ptr<int>(expert_for_source_row);

        moe_runner.run_moe_fc(input_act_ptr,
                              gating_output_ptr,
                              fc1_expert_weights_ptr,
                              fc1_scales_ptr,
                              fc1_expert_biases_ptr,
                              fc1_activation_type,
                              fc2_expert_weights_ptr,
                              fc2_scales_ptr,
                              num_rows,
                              hidden_size,
                              inter_size,
                              num_experts,
                              k,
                              workspace_ptr,
                              fc2_output_ptr,
                              finished_ptr,
                              active_rows,
                              expert_scales_ptr,
                              expanded_source_row_to_expanded_dest_row_ptr,
                              expert_for_source_row_ptr,
                              stream);

        auto output_tensor =
            torch::empty({num_rows, hidden_size}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        T *output_tensor_ptr = get_ptr<T>(output_tensor);

        fastertransformer::finalize_moe_routing_kernelLauncher(fc2_output_ptr,
                                                               output_tensor_ptr,
                                                               skip_layer_ptr,
                                                               fc2_expert_biases_ptr,
                                                               expert_scales_ptr,
                                                               expanded_source_row_to_expanded_dest_row_ptr,
                                                               expert_for_source_row_ptr,
                                                               num_rows,
                                                               hidden_size,
                                                               k,
                                                               stream);
        return output_tensor;
    }

    Tensor run_moe_fc(Tensor input_activations,
                      Tensor gating_output,
                      Tensor fc1_expert_weights,
                      Tensor fc1_scales,
                      Tensor fc1_expert_biases,
                      std::string fc1_activation_type_str,
                      Tensor fc2_expert_weights,
                      Tensor fc2_scales,
                      Tensor fc2_expert_biases,
                      Tensor skip_layer,
                      Tensor finished,
                      int64_t active_rows,
                      int64_t k)
    {

        const at::ScalarType _st = input_activations.scalar_type();

        const int num_rows = input_activations.size(0);
        const int hidden_size = input_activations.size(1);
        const int inter_size = fc2_expert_weights.size(1);
        const int num_experts = gating_output.size(-1);

        // We signal int4 by having the last weight dim be half the size of the scales. This is because int4 elements are
        // packed into a single byte.
        torch::ScalarType quant_type = fc2_expert_weights.scalar_type();
        TORCH_CHECK(fc2_expert_weights.scalar_type() == fc1_expert_weights.scalar_type(),
                    "FC1 and FC2 must be quantized to the same type");
        if (fc1_scales.dim() > 0 && fc1_expert_weights.size(-1) == fc1_scales.size(-1) / 2)
        {
            TORCH_CHECK(fc2_expert_weights.size(-1) == fc2_scales.size(-1) / 2, "FC1 and FC2 must be both be int4.");
            quant_type = at::ScalarType::QUInt4x2;
        }

        CHECK_INPUT(input_activations, _st);
        TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");

        CHECK_INPUT(gating_output, _st);
        TORCH_CHECK(gating_output.dim() == 2, "Invalid rank for gating output");
        TORCH_CHECK(gating_output.size(0) == num_rows, "gating output and activations must have same number of rows");

        CHECK_TH_CUDA(fc1_expert_weights);
        CHECK_CONTIGUOUS(fc1_expert_weights);
        TORCH_CHECK(fc1_expert_weights.dim() == 3, "Invalid rank for fc1 weights");
        TORCH_CHECK(fc1_expert_weights.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 weights");
        TORCH_CHECK(fc1_expert_weights.size(1) == hidden_size,
                    "Activation last dim must equal size of dim 1 for fc1 weight");

        const int fc1_num_cols =
            quant_type == at::ScalarType::QUInt4x2 ? 2 * fc1_expert_weights.size(-1) : fc1_expert_weights.size(-1);
        if (_st != torch::kFloat32 && _st != torch::kFloat16)
        {
            CHECK_INPUT(fc1_scales, _st);
            TORCH_CHECK(fc1_scales.dim() == 2, "Invalid rank for fc1 scales");
            TORCH_CHECK(fc1_scales.size(0) == num_experts, "Experts mismatch between gate outputs and fc1 scales");
            TORCH_CHECK(fc1_scales.size(-1) == fc1_num_cols, "Mismatch between fc1 weights and scale shapes");
            TORCH_CHECK(fc1_scales.size(-1) == fc1_expert_biases.size(-1), "Mismatch between fc1 scale and bias shapes");
        }

        CHECK_INPUT(fc1_expert_biases, _st);
        TORCH_CHECK(fc1_expert_biases.dim() == 2, "Invalid rank for fc1 biases");
        TORCH_CHECK(fc1_expert_biases.size(0) == gating_output.size(-1),
                    "Experts mismatch between gate outputs and fc1 biases");

        CHECK_TH_CUDA(fc2_expert_weights);
        CHECK_CONTIGUOUS(fc2_expert_weights);
        TORCH_CHECK(fc2_expert_weights.dim() == 3, "Invalid rank for fc2 weights");
        TORCH_CHECK(fc2_expert_weights.size(0) == gating_output.size(-1),
                    "Experts mismatch between gate outputs and fc2 weights");
        TORCH_CHECK(fc2_expert_weights.size(1) == fc1_num_cols, "fc1 weight last dim must equal dim 1 of fc2 weights");

        if (_st != torch::kFloat32 && _st != torch::kFloat16)
        {
            CHECK_INPUT(fc2_scales, _st);
            TORCH_CHECK(fc2_scales.dim() == 2, "Invalid rank for fc2 scales");
            TORCH_CHECK(fc2_scales.size(0) == gating_output.size(-1),
                        "Experts mismatch between gate outputs and fc2 scales");
            const int fc2_num_cols =
                quant_type == at::ScalarType::QUInt4x2 ? 2 * fc2_expert_weights.size(-1) : fc2_expert_weights.size(-1);
            TORCH_CHECK(fc2_scales.size(-1) == fc2_num_cols, "Mismatch between fc2 weights and scale shapes");
            TORCH_CHECK(fc2_scales.size(-1) == fc2_expert_biases.size(-1), "Mismatch between fc2 scale and bias shapes");
        }

        CHECK_INPUT(fc2_expert_biases, _st);
        TORCH_CHECK(fc2_expert_biases.dim() == 2, "Invalid rank for fc2 biases");
        TORCH_CHECK(fc2_expert_biases.size(0) == num_experts, "Experts mismatch between gate outputs and fc2 biases");

        CHECK_INPUT(skip_layer, _st);
        TORCH_CHECK(skip_layer.sizes() == input_activations.sizes(), "Invalid rank for skip connection");

        CHECK_INPUT(finished, torch::kBool);
        TORCH_CHECK(finished.dim() == 1, "Invalid rank for finished tensor");
        TORCH_CHECK(finished.size(0) == input_activations.size(0),
                    "Finished and activations must have same number of rows");

        Tensor output_tensor;

        fastertransformer::ActivationType fc1_activation_type = fastertransformer::ActivationType::InvalidType;
        if (fc1_activation_type_str == "identity")
        {
            fc1_activation_type = fastertransformer::ActivationType::Identity;
        }
        else
        {
            fc1_activation_type = fastertransformer::getActivationType(fc1_activation_type_str);
        }

        switch (_st)
        {
        case at::ScalarType::Half:
        {
            if (quant_type == torch::kInt8)
            {
                output_tensor = run_moe_fc_helper<half, uint8_t>(input_activations,
                                                                 gating_output,
                                                                 fc1_expert_weights,
                                                                 fc1_scales,
                                                                 fc1_expert_biases,
                                                                 fc1_activation_type,
                                                                 fc2_expert_weights,
                                                                 fc2_scales,
                                                                 fc2_expert_biases,
                                                                 skip_layer,
                                                                 finished,
                                                                 active_rows,
                                                                 k);
            }
            else
            {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
        case at::ScalarType::BFloat16:
        {
            if (quant_type == torch::kInt8)
            {
                output_tensor = run_moe_fc_helper<__nv_bfloat16, uint8_t>(input_activations,
                                                                          gating_output,
                                                                          fc1_expert_weights,
                                                                          fc1_scales,
                                                                          fc1_expert_biases,
                                                                          fc1_activation_type,
                                                                          fc2_expert_weights,
                                                                          fc2_scales,
                                                                          fc2_expert_biases,
                                                                          skip_layer,
                                                                          finished,
                                                                          active_rows,
                                                                          k);
            }
            else
            {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
        default:
            throw std::runtime_error("Wrong Tensor type.");
        }
        return output_tensor;
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
    {
        m.def("grouped_gemm_bias", &grouped_gemm_bias, "");
        m.def("run_moe_fc", &run_moe_fc, "");
        m.def("preprocess_weights_for_mixed_gemm", &preprocess_weights_for_mixed_gemm, "");
        m.def("_symmetric_quantize_last_axis_of_batched_matrix", &_symmetric_quantize_last_axis_of_batched_matrix, "");
    }
}