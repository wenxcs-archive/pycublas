import os
import sys
import ctypes

# https://github.com/tpn/cuda-samples/blob/master/v12.0/include/cublas_api.h

CUBLAS_OP = {
    0: 0,  # CUBLAS_OP_N
    "n": 0,
    "N": 0,
    1: 1,  # CUBLAS_OP_T
    "t": 1,
    "T": 1,
    2: 2,  # CUBLAS_OP_C, conjugate transpose
    "c": 2,
    "C": 2,
}

CUBLAS_FILL_MODE = {
    0: 0,  # CUBLAS_FILL_MODE_LOWER
    "l": 0,
    "L": 0,
    1: 1,  # CUBLAS_FILL_MODE_UPPER
    "u": 1,
    "U": 1,
}

CUBLAS_DIAG = {
    0: 0,  # CUBLAS_DIAG_NON_UNIT,
    "n": 0,
    "N": 0,
    1: 1,  # CUBLAS_DIAG_UNIT
    "u": 1,
    "U": 1,
}

CUBLAS_SIDE_MODE = {
    0: 0,  # CUBLAS_SIDE_LEFT
    "l": 0,
    "L": 0,
    1: 1,  # CUBLAS_SIDE_RIGHT
    "r": 1,
    "R": 1,
}

CUBLAS_DATA_TYPE = {
    "CUDA_R_16F": 2,  # real as a half
    "CUDA_C_16F": 6,  # complex as a pair of half numbers
    "CUDA_R_16BF": 14,  # real as a nv_bfloat16
    "CUDA_C_16BF": 15,  # complex as a pair of nv_bfloat16 numbers
    "CUDA_R_32F": 0,  # real as a float
    "CUDA_C_32F": 4,  # complex as a pair of float numbers
    "CUDA_R_64F": 1,  # real as a double
    "CUDA_C_64F": 5,  # complex as a pair of double numbers
    "CUDA_R_4I": 16,  # real as a signed 4-bit int
    "CUDA_C_4I": 17,  # complex as a pair of signed 4-bit int numbers
    "CUDA_R_4U": 18,  # real as a unsigned 4-bit int
    "CUDA_C_4U": 19,  # complex as a pair of unsigned 4-bit int numbers
    "CUDA_R_8I": 3,  # real as a signed 8-bit int
    "CUDA_C_8I": 7,  # complex as a pair of signed 8-bit int numbers
    "CUDA_R_8U": 8,  # real as a unsigned 8-bit int
    "CUDA_C_8U": 9,  # complex as a pair of unsigned 8-bit int numbers
    "CUDA_R_16I": 20,  # real as a signed 16-bit int
    "CUDA_C_16I": 21,  # complex as a pair of signed 16-bit int numbers
    "CUDA_R_16U": 22,  # real as a unsigned 16-bit int
    "CUDA_C_16U": 23,  # complex as a pair of unsigned 16-bit int numbers
    "CUDA_R_32I": 10,  # real as a signed 32-bit int
    "CUDA_C_32I": 11,  # complex as a pair of signed 32-bit int numbers
    "CUDA_R_32U": 12,  # real as a unsigned 32-bit int
    "CUDA_C_32U": 13,  # complex as a pair of unsigned 32-bit int numbers
    "CUDA_R_64I": 24,  # real as a signed 64-bit int
    "CUDA_C_64I": 25,  # complex as a pair of signed 64-bit int numbers
    "CUDA_R_64U": 26,  # real as a unsigned 64-bit int
    "CUDA_C_64U": 27,  # complex as a pair of unsigned 64-bit int numbers
}

CUBLAS_COMPUTE_TYPE = {
    "CUBLAS_COMPUTE_16F": 64,  # half - default
    "CUBLAS_COMPUTE_16F_PEDANTIC": 65,  # half - pedantic
    "CUBLAS_COMPUTE_32F": 68,  # float - default
    "CUBLAS_COMPUTE_32F_PEDANTIC": 69,  # float - pedantic
    # float - fast, allows down-converting inputs to half or TF32
    "CUBLAS_COMPUTE_32F_FAST_16F": 74,
    # float - fast, allows down-converting inputs to bfloat16 or TF32
    "CUBLAS_COMPUTE_32F_FAST_16BF": 75,
    # float - fast, allows down-converting inputs to TF32
    "CUBLAS_COMPUTE_32F_FAST_TF32": 77,
    "CUBLAS_COMPUTE_64F": 70,  # double - default
    "CUBLAS_COMPUTE_64F_PEDANTIC": 71,  # double - pedantic
    "CUBLAS_COMPUTE_32I": 72,  # signed 32-bit int - default
    "CUBLAS_COMPUTE_32I_PEDANTIC": 73,  # signed 32-bit int - pedantic
}

CUBLAS_GEMM_ALGO = {
    "CUBLAS_GEMM_DFALT": -1,
    "CUBLAS_GEMM_DEFAULT": -1,
    "CUBLAS_GEMM_ALGO0": 0,
    "CUBLAS_GEMM_ALGO1": 1,
    "CUBLAS_GEMM_ALGO2": 2,
    "CUBLAS_GEMM_ALGO3": 3,
    "CUBLAS_GEMM_ALGO4": 4,
    "CUBLAS_GEMM_ALGO5": 5,
    "CUBLAS_GEMM_ALGO6": 6,
    "CUBLAS_GEMM_ALGO7": 7,
    "CUBLAS_GEMM_ALGO8": 8,
    "CUBLAS_GEMM_ALGO9": 9,
    "CUBLAS_GEMM_ALGO10": 10,
    "CUBLAS_GEMM_ALGO11": 11,
    "CUBLAS_GEMM_ALGO12": 12,
    "CUBLAS_GEMM_ALGO13": 13,
    "CUBLAS_GEMM_ALGO14": 14,
    "CUBLAS_GEMM_ALGO15": 15,
    "CUBLAS_GEMM_ALGO16": 16,
    "CUBLAS_GEMM_ALGO17": 17,
    "CUBLAS_GEMM_ALGO18": 18,  # sliced 32x32
    "CUBLAS_GEMM_ALGO19": 19,  # sliced 64x32
    "CUBLAS_GEMM_ALGO20": 20,  # sliced 128x32
    "CUBLAS_GEMM_ALGO21": 21,  # sliced 32x32  -splitK
    "CUBLAS_GEMM_ALGO22": 22,  # sliced 64x32  -splitK
    "CUBLAS_GEMM_ALGO23": 23,  # sliced 128x32 -splitK
    "CUBLAS_GEMM_DEFAULT_TENSOR_OP": 99,
    "CUBLAS_GEMM_DFALT_TENSOR_OP": 99,
    "CUBLAS_GEMM_ALGO0_TENSOR_OP": 100,
    "CUBLAS_GEMM_ALGO1_TENSOR_OP": 101,
    "CUBLAS_GEMM_ALGO2_TENSOR_OP": 102,
    "CUBLAS_GEMM_ALGO3_TENSOR_OP": 103,
    "CUBLAS_GEMM_ALGO4_TENSOR_OP": 104,
    "CUBLAS_GEMM_ALGO5_TENSOR_OP": 105,
    "CUBLAS_GEMM_ALGO6_TENSOR_OP": 106,
    "CUBLAS_GEMM_ALGO7_TENSOR_OP": 107,
    "CUBLAS_GEMM_ALGO8_TENSOR_OP": 108,
    "CUBLAS_GEMM_ALGO9_TENSOR_OP": 109,
    "CUBLAS_GEMM_ALGO10_TENSOR_OP": 110,
    "CUBLAS_GEMM_ALGO11_TENSOR_OP": 111,
    "CUBLAS_GEMM_ALGO12_TENSOR_OP": 112,
    "CUBLAS_GEMM_ALGO13_TENSOR_OP": 113,
    "CUBLAS_GEMM_ALGO14_TENSOR_OP": 114,
    "CUBLAS_GEMM_ALGO15_TENSOR_OP": 115,
}

# https://gregstoll.com/~gregstoll/floattohex/
float32_one = ctypes.c_uint32(0x3F800000)
float32_zero = ctypes.c_uint32(0x00000000)
float64_one = ctypes.c_uint64(0x3FF0000000000000)
float64_zero = ctypes.c_uint64(0x0000000000000000)
# https://evanw.github.io/float-toy/
float16_one = ctypes.c_uint16(0x3C00)
float16_zero = ctypes.c_uint16(0x0000)
float16_zero_point_one = ctypes.c_uint16(0x2E66)
float16_negative_one = ctypes.c_uint16(0xBC00)
float16_negative_zero_point_one = ctypes.c_uint16(0xAE66)
bfloat16_one = ctypes.c_uint16(0x3F80)
bfloat16_zero = ctypes.c_uint16(0x0000)
