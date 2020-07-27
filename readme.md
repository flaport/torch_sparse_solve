# Torch Sparse Solve

An alternative to `torch.solve` for sparse PyTorch CPU tensors using
the efficient
[KLU algorithm](https://ufdcimages.uflib.ufl.edu/UF/E0/01/17/21/00001/palamadai_e.pdf).

## CPU tensors only

This library is a wrapper around the
[SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse) KLU
algorithms. This means the algorithm is only implemented for
C-arrays and hence is only available for PyTorch CPU
tensors. However, for large, sparse enough tensors, it might still be
worth doing the GPU→CPU conversion.

## Installation

So far, wheels only exist for Python 3.7 on Windows and Linux:

```bash
pip install torch_sparse_solve
```

## Usage

The `torch_sparse_solve` library provides a single function `solve(A, b)`, which solves for `x` in the **batched matrix × batched matrix**
system `Ax=b` for `torch.float64` tensors (notice the different API in comparison to `torch.solve`):

```python
import torch
from torch_sparse_solve import solve
torch.manual_seed(42)
mask = torch.tensor([[[1,0,0],[1,1,0],[0,0,1]]], dtype=torch.float64)
A = (mask * torch.randn(4, 3, 3, dtype=torch.float64)).to_sparse()
b = torch.randn(4, 3, 2, dtype=torch.float64)
x = solve(A, b)

# compare to torch.solve:
A = A.to_dense()
print( (x - torch.solve(b, A)[0] < 1e-9).all() )
```

`True`

## Caveats

There are two major caveats you should be aware of when using
`torch_sparse_solve.solve(A, b)`:

- `A` should be 'dense' in the first dimension, i.e. the batch dimension
  should contain as many elements as the batch size.

- `A` should have the same sparsity pattern for every element in the batch.
  If this is not the case, you have two options:
  1. Create a new sparse matrix with the same sparsity pattern for
     every element in the batch by adding zeros to the sparse
     representation.
  2. **OR** loop over the batch dimension and solve sequentially, i.e.
     with shapes `(1, m, m)` and `(1, m, n)` for each element in `A` and `b`
     respectively.

## Development installation

If a wheel for your environment does not exist, you can try installing from source:

### Linux

Run the following commands in a **fresh Anaconda environment**:
```bash
conda install pytorch cpuonly -c pytorch
conda install suitesparse scipy
python setup.py install
```

### Windows
First download the Visual Studio Community 2017 installer from [here](https://my.visualstudio.com/Downloads?q=visual%20studio%202017&wt.mc_id=o~msft~vscom~older-downloads).
During installation, go to **Workloads** and select the following two workloads:
* Desktop development with C++
* Python development

Then go to **Individual Components** and select the following additional items:
* C++/CLI support
* VC++ 2015.3 v14.00 (v140) toolset for desktop

Then run the following commands *inside* a **x64 Native Tools Command Prompt for VS 2017** in a **fresh Anaconda environment**:

```bash
conda install pytorch cpuonly -c pytorch
conda install suitesparse scipy
python setup.py install
```

## License & Credits

© Floris Laporte 2020, LGPL-2.1

This library was partly based on:

- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), LGPL-2.1
- [kagami-c/PyKLU](https://github.com/kagami-c/PyKLU), LGPL-2.1
- [scipy.sparse](https://github.com/scipy/scipy/tree/master/scipy/sparse), BSD-3
