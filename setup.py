import os
import glob
from setuptools import setup, Extension
from torch.utils import cpp_extension

import torch_sparse_solve

libroot = os.path.dirname(os.path.dirname(os.__file__))
if os.name == "nt":  # Windows
    suitesparse_lib = os.path.join(libroot, "Library", "lib")
    suitesparse_include = os.path.join(libroot, "Library", "include", "suitesparse")
else:  # Linux / Mac OS
    suitesparse_lib = os.path.join(os.path.dirname(libroot), "lib")
    suitesparse_include = os.path.join(os.path.dirname(libroot), "include")

torch_sparse_solve_cpp = Extension(
    name="torch_sparse_solve_cpp",
    sources=["torch_sparse_solve.cpp"],
    include_dirs=[*cpp_extension.include_paths(), suitesparse_include],
    library_dirs=[*cpp_extension.library_paths(), suitesparse_lib],
    extra_compile_args=[],
    libraries=[
        "c10",
        "torch",
        "torch_cpu",
        "torch_python",
        "klu",
        "btf",
        "amd",
        "colamd",
        "suitesparseconfig",
    ],
    language="c++",
)

try:
    with open("readme.md", "r") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = torch_sparse_solve.__doc__

setup(
    name="torch_sparse_solve",
    version=torch_sparse_solve.__version__,
    author=torch_sparse_solve.__author__,
    author_email="floris.laporte@gmail.com",
    description=torch_sparse_solve.__doc__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/flaport/torch_sparse_solve",
    py_modules=["torch_sparse_solve"],
    ext_modules=[torch_sparse_solve_cpp],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 2.7",
        "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)
