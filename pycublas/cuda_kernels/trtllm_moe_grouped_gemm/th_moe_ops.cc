#include "moe_gemm_kernels.h"
#include "torch/extension.h"

#include "c10/cuda/CUDAStream.h"

#include "cutlass_extensions/gemm_configs.h"

using torch::Tensor;

#define CHECK_TYPE(x, st) TORCH_CHECK(x.scalar_type() == st, "Inconsistency of Tensor type: " #x)
#define CHECK_TH_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x, st)                                                                                             \
    CHECK_TH_CUDA(x);                                                                                                  \
    CHECK_CONTIGUOUS(x);                                                                                               \
    CHECK_TYPE(x, st)

namespace torch_ext
{
    template <typename T>
    T* get_ptr(Tensor t)
    {
        return (T*)t.data_ptr();
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

        // auto res = torch::zeros({num_rows, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
        auto res = torch::empty({num_rows, gemm_n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));

        T *act_ptr = get_ptr<T>(activations);
        WeightType *wt_ptr = get_ptr<WeightType>(weights);
        T *weight_scale_ptr = get_ptr<T>(weight_scales);
        T *res_ptr = get_ptr<T>(res);
        int64_t *total_rows_before_expert_ptr = get_ptr<int64_t>(total_rows_before_expert);

        tensorrt_llm::MoeGemmRunner<T, WeightType> moe_gemm_runner;
        auto configs = moe_gemm_runner.getConfigs();
        assert(configs.size() > 1);
        // estimate_best_config_from_occupancies()
        moe_gemm_runner.setBestConfig(configs[0]);
        moe_gemm_runner.moeGemmBiasAct(
            act_ptr,
            wt_ptr,
            weight_scale_ptr,
            nullptr,
            res_ptr,
            total_rows_before_expert_ptr,
            tensorrt_llm::HopperGroupedGemmInput{},
            (int64_t)num_rows,
            (int64_t)gemm_n,
            (int64_t)gemm_k,
            experts,
            tensorrt_llm::ActivationType::Identity,
            false,
            stream
        );
        /*
        moe_gemm_runner.moeGemm(act_ptr,
                                wt_ptr,
                                weight_scale_ptr,
                                res_ptr,
                                total_rows_before_expert_ptr,
                                tensorrt_llm::HopperGroupedGemmInput{},
                                num_rows,
                                gemm_n,
                                gemm_k,
                                experts,
                                false,
                                stream);
                                */
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