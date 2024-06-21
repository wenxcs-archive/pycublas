try:
    import loguru
    import git
except:
    import os
    os.system("pip install loguru gitpython pytest")

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
try :
    cur_repo = Repo(os.path.dirname(__file__))
    version = version + "+" + cur_repo.head.commit.hexsha[:7]
except:
    pass
features = ["trtllm_moe_grouped_gemm"]
ext_modules = []

if cuda_arch == 800:
    if "trtllm_moe_grouped_gemm" in features:
        logger.info("CUDA SM80 detected -> building trtllm_moe_grouped_gemm sm80 kernel.")
        module_name = "trtllm_moe_grouped_gemm"
        module_folder = os.path.join(project_path, project_name, "cuda_kernels", module_name)
        module_cutlass_path = os.path.join(module_folder, "cutlass")

        if not os.path.exists(module_cutlass_path):
            module_cutlass_repo = Repo.clone_from("https://github.com/NVIDIA/cutlass.git", module_cutlass_path)
            module_cutlass_repo.git.checkout("7d49e6c")

        ext_modules.append(cpp_extension.CUDAExtension(
                name=f"{project_name}.{module_name}",
                sources=[
                    os.path.join(module_folder, "fp16_int8_gemm_fg_scaleonly.cu"),
                    os.path.join(module_folder, "fused_moe_gemm_launcher_sm80.cu"),
                    os.path.join(module_folder, "cutlass_heuristic.cpp"),
                    os.path.join(module_folder, "cutlass_preprocessors.cpp"),
                    os.path.join(module_folder, "moe_gemm_kernels_fp16_uint8.cu"),
                    # os.path.join(module_folder, "fpA_intB_launcher_sm90.cu"),
                    # os.path.join(module_folder, "moe_kernels.cu"),
                    os.path.join(module_folder, "th_moe_ops.cc"),
                    os.path.join(module_folder, "tllmException.cpp"),
                    os.path.join(module_folder, "logger.cpp"),
                    os.path.join(module_folder, "stringUtils.cpp"),
                ],
                include_dirs=[os.path.join(module_folder, "cutlass_extensions", "include"),
                              os.path.join(module_folder),
                              os.path.join(module_folder, "include"),
                              os.path.join(module_cutlass_path, "tools/util/include"),
                              os.path.join(module_cutlass_path, "include")
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
                    "cxx": ["-std=c++20", "-O3"],
                    "nvcc": [
                        "-O3",
                        "-std=c++20",
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