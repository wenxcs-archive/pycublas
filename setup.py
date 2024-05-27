import os
import shutil
from setuptools import setup, find_packages
import torch
from torch.utils import cpp_extension
from git import Repo
from loguru import logger

compute_capability = torch.cuda.get_device_capability()
cuda_arch = compute_capability[0] * 100 + compute_capability[1] * 10
cutlass_path = os.path.dirname(__file__) + "/cutlass"

if not os.path.exists(cutlass_path):
    logger.info("Cloning cutlass repository.")
    try:
        Repo.clone_from("https://github.com/NVIDIA/cutlass.git", cutlass_path)
        print(f"Repository cloned successfully to {cutlass_path}")
    except Exception as e:
        print(f"Failed to clone repository: {e}")
else:
    logger.info("Cutlass repository already exists.")

if shutil.which("ninja") is None:
    raise RuntimeError("The ninja is not found. ")
    
project_name = "pycublas"
version = "0.1"
project_path = os.path.dirname(__file__)
cur_repo = Repo(os.path.dirname(__file__))
version = version + "+" + cur_repo.head.commit.hexsha[:7]

ext_modules = []

if cuda_arch == 800:
    logger.info("CUDA SM80 detected -> building vllm_moe_sparse_gemm sm80 kernel.")
    ext_modules.append(cpp_extension.CUDAExtension(
            name=f"{project_name}.vllm_moe_sparse_gemm",
            sources=[
                f"{project_name}/cuda_kernels/vllm_moe_sparse_gemm/kernels_sm80.cu",
            ],
            include_dirs=[f"{project_name}/cuda_kernels/vllm_moe_sparse_gemm/",
                          os.path.join(cutlass_path, "tools/util/include"),
                          os.path.join(cutlass_path, "include")],
            extra_link_args=[
                "-lcuda",
                "-lculibos",
                "-lcudart",
                "-lcudart_static",
                "-lrt",
                "-lpthread",
                "-ldl",
                "-L/usr/lib/x86_64-linux-gnu/",
            ],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-DCUDA_ARCH=80",
                    "-gencode=arch=compute_80,code=compute_80",
                ],
            },
        ))

    if True:
        # Build faster transformer MOE
        ft_cutlass_path = os.path.join(project_path, f"{project_name}/cuda_kernels/FasterTransformer/cutlass")
        if not os.path.exists(ft_cutlass_path):
            ft_cutlass = Repo.clone_from("https://github.com/NVIDIA/cutlass.git", ft_cutlass_path)
            ft_cutlass.git.checkout("cc85b64cf676c45f98a17e3a47c0aafcf817f088")
        ext_modules.append(cpp_extension.CUDAExtension(
                name=f"{project_name}.fasttransformer_moe_sparse_gemm",
                sources=[
                    f"{project_name}/cuda_kernels/FasterTransformer/moe_gemm_kernels_bf16_fp8.cu",
                    f"{project_name}/cuda_kernels/FasterTransformer/cutlass_heuristic.cc",
                    f"{project_name}/cuda_kernels/FasterTransformer/th_moe_ops.cc",
                    f"{project_name}/cuda_kernels/FasterTransformer/logger.cc",
                ],
                include_dirs=[os.path.join(project_path, f"{project_name}/cuda_kernels/FasterTransformer/"),
                              os.path.join(project_path, f"{project_name}/cuda_kernels/FasterTransformer/cutlass_extensions/include"),
                              os.path.join(ft_cutlass_path, "tools/util/include"),
                              os.path.join(ft_cutlass_path, "include")
                            ],
                extra_link_args=[
                    "-lcuda",
                    "-lculibos",
                    "-lcudart",
                    "-lcudart_static",
                    "-lrt",
                    "-lpthread",
                    "-ldl",
                    "-L/usr/lib/x86_64-linux-gnu/",
                ],
                extra_compile_args={
                    "cxx": ["-std=c++17", "-O3"],
                    "nvcc": [
                        "-O3",
                        "-std=c++17",
                        "-DCUDA_ARCH=80",
                        "-gencode=arch=compute_80,code=compute_80",
                    ],
                },
            ))

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name=f"{project_name}",
    version=f"{version}",
    author="Wenxiang@Microsoft Research",
    ext_modules=ext_modules,
    cmdclass={"build_ext": cpp_extension.BuildExtension.with_options(use_ninja=True)},
    packages=find_packages(exclude=["notebook", "scripts", "test"]),
)