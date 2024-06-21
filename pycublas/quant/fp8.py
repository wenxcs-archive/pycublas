import torch
from vllm import _custom_ops as ops
import pycublas.trtllm_moe_grouped_gemm as moe_kernel

# ops.scaled_fp8_quant

def remove_subnormal_fp8(tensor, verbose=False):
    assert tensor.dtype == torch.uint8, "Tensor must be a byte tensor representing fp8 values"
    exponent_mask = 0b11111000
    mantissa_mask = 0b00000111
    exponents = (tensor & exponent_mask) >> 3
    mantissas = tensor & mantissa_mask
    subnormal_mask = (exponents == 0) & (mantissas != 0)
    if verbose and subnormal_mask.any():
        print(subnormal_mask.sum().item() / subnormal_mask.numel() * 100, "% of values are subnormal")
    tensor[subnormal_mask] = 0
    return tensor

def preproces_fp8_linear_weights(w, scale):
    assert w.dtype == torch.float16, "Weights must be fp16"
    assert scale.numel() == 1, "Only one scale value is supported"
    device = w.device
    # Preprocess weights for mixed gemm
    w = moe_kernel.preprocess_weights_for_mixed_gemm(w.view(torch.int8).transpose(1,2).contiguous().cpu()).to(device)
    # Preprocess scale for mixed gemm
    w_scale = scale.to(dtype=torch.float16).reshape(1).unsqueeze(1).expand(-1, w.size(-1)).contiguous()
    return w, w_scale

def is_sm80(device_id=0):
    if not torch.cuda.is_available():
        return False
    device_properties = torch.cuda.get_device_properties(device_id)
    return (device_properties.major == 8 and device_properties.minor == 0)