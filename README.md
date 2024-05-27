# pycublas
Python interface updated for cublas. This project is inspired by discontinued cublas interface inside scikit project and provides updated features such as mixed precisions supports for BF16, FP8 etc.

## Features
- Updated features with latest cublas
- Friendly interfaces compatible with pytorch/python

## Install
```bash
pip install git+https://github.com/wenxcs/pycublas.git
```
## Usage
```python
#[optional] CUBLAS_LIBNAME=/home/wenxh/workspace/llama_inference/env/lib/libcublas.so.12
import pycublas

a = torch.randn(1024, 512).cuda().bfloat16()
b = torch.randn(256, 512).cuda().bfloat16()
c = pycublas.function.matmul.matmul_nt_bf16_fp32(a, b)
```

## Disclaim
This project is under MIT license, and "cublas" is belong to NVidia.