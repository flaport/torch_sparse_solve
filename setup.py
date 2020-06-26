import os
import glob
from setuptools import setup, Extension
from torch.utils import cpp_extension

conda = os.path.dirname(os.path.dirname(os.path.dirname(os.__file__)))
folders = glob.glob(os.path.join(conda, "pkgs", "suitesparse-*"))
folders = [folder for folder in folders if not folder.endswith(".conda")]
suitesparse = sorted(folders)[-1] # latest version if multiple installed.

torch_sparse_solve = Extension(
    name="torch_sparse_solve_cpp",
    sources=["torch_sparse_solve.cpp"],
    include_dirs=[
        *cpp_extension.include_paths(),
        os.path.join(suitesparse, "include"),
    ],
    library_dirs=[*cpp_extension.library_paths(), os.path.join(suitesparse, "lib"),],
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
        "rt",
    ],
    language="c++",
)

setup(
    name="torch_sparse_solve_cpp",
    ext_modules=[torch_sparse_solve],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
