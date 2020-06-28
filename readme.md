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

Installation only confirmed on linux for now.

```bash
conda install suitesparse
python setup.py install
```

## Usage

The `torch_sparse_solve` library provides a single function `solve(A, b)`, which solves for `x` in the **batched matrix × batched matrix**
system `Ax=b` for `torch.float64` tensors:

```python
import torch
torch.manual_seed(42)
from torch_sparse_solve import solve
A = torch.randn(4, 5, 5, dtype=torch.float64)
b = torch.randn(4, 5, 2, dtype=torch.float64)
A[A<-0.2] = 0 # enforce sparse matrix
A = A.to_sparse()
x = solve(A, b)

# compare to torch.solve:
A = A.to_dense()
print( (x - torch.solve(b, A)[0] < 1e-5).all() )
```

`True`

Notice the API differences between `torch_sparse_solve.solve` and
`torch.solve`.

## License & Credits

© Floris Laporte 2020, LGPL-2.1

This library was partly based on:

- [SuiteSparse](https://github.com/DrTimothyAldenDavis/SuiteSparse), LGPL-2.1
- [kagami-c/PyKLU](https://github.com/kagami-c/PyKLU), LGPL-2.1
- [scipy.sparse](https://github.com/scipy/scipy/tree/master/scipy/sparse), BSD-3

