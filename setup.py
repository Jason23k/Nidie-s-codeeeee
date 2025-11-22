import os
import torch
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# 检查是否是 Windows 系统
is_windows = os.name == 'nt'

# 定义需要编译的扩展模块
ext_modules = [
    CUDAExtension(
        name='models.ops.modules',
        sources=[
            'models/ops/src/vision.cpp',
            'models/ops/src/cpu/ms_deform_attn_cpu.cpp',
            'models/ops/src/cuda/ms_deform_attn_cuda.cu'
        ],
        include_dirs=[
            'models/ops/src',           # ✅ 确保这个目录在最前面
            'models/ops/src/cpu',
            'models/ops/src/cuda',
            # 可选：CUDA 路径
            # '/usr/local/cuda/include',
        ],
        library_dirs=[
            # 可选：CUDA 库路径
            # '/usr/local/cuda/lib64',
        ],
        libraries=['cudart'],
        extra_compile_args={
            'cxx': ['-O3', '-std=c++14'],
            'nvcc': [
                '-O3',
                '-std=c++14',
                '-arch=sm_70',  # 根据你的 GPU 修改
            ]
        }
    )
]

setup(
    name='transvod',
    version='0.1',
    description='TransVOD Project',
    ext_modules=ext_modules,
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)