from setuptools import setup, Extension
from torch.utils import cpp_extension

torch_sparse_solve = Extension(
    name = "torch_sparse_solve_cpp",
    sources = ["torch_sparse_solve.cpp"],
    include_dirs=cpp_extension.include_paths(),
    library_dirs=cpp_extension.library_paths(),
    libraries=["c10", "torch", "torch_cpu", "torch_python"],
    language="c++",
)

setup(
    name="torch_sparse_solve_cpp",
    ext_modules=[torch_sparse_solve],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
