from setuptools import setup, Extension
import sys
import os
# 从 pybind11.setup_helpers 导入 build_ext 类
from pybind11.setup_helpers import build_ext
# 导入 get_include 以获取头文件路径
import pybind11

# 获取 pybind11 包含目录
pybind11_include = pybind11.get_include()

# 定义扩展模块
ext_modules = [
    Extension(
        "idm_module",
        ["run_IDM.cpp"],
        include_dirs=[
            pybind11_include,
            "/home/dayun/PhaseTracerForMe/example/IDM", # 假设头文件在此处，请根据实际路径调整
            "/usr/include",
            "/usr/include/eigen3",
            "/home/dayun/PhaseTracerForMe/include/",
            "/home/dayun/PhaseTracerForMe/EffectivePotential/include/effectivepotential/"
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-O3"],
        # 如果依赖其他库，需要在这里链接
        libraries=["phasetracer"], 
        library_dirs=["/home/dayun/PhaseTracerForMe/lib"],
        # 添加 rpath，使得生成的 .so 文件在运行时能自动找到 libphasetracer.so
        extra_link_args=["-Wl,-rpath,/home/dayun/PhaseTracerForMe/lib"],
    ),
]

setup(
    name="idm_module",
    version="0.0.1",
    author="Dayun",
    description="IDM Module for Python",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.0"],
    cmdclass={"build_ext": build_ext},
)