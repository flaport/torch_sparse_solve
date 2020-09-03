""" A sparse KLU solver for PyTorch """

import torch

__version__ = "0.0.4"
__author__ = "Floris Laporte"
__all__ = ["solve"]


def solve(A, b):
    """solve a sparse system Ax = b

    Args:
        A (torch.sparse.Tensor[b, m, m]): the sparse matrix defining the system.
        b (torch.Tensor[b, m, n]): the target matrix b

    Returns:
        x (torch.Tensor[b, m, n]): the initially unknown matrix x

    Note:
        'A' should be 'dense' in the first dimension, i.e. the batch dimension
        should contain as many elements as the batch size.

        'A' should have the same sparsity pattern for every element in the batch.
        If this is not the case, you have two options:
            1. Create a new sparse matrix with the same sparsity pattern for
            every element in the batch by adding zeros to the sparse
            representation.
            2. OR loop over the batch dimension and solve sequentially, i.e.
            with shapes (1, m, m) and (1, m, n) for each element in 'A' and 'b'
            respectively.
    """
    return Solve.apply(A, b)


class Solve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        if A.ndim != 3 or (A.shape[1] != A.shape[2]) or not A.is_sparse:
            raise ValueError(
                "'A' should be a batch of square 2D sparse matrices with shape (b, m, m)."
            )
        if b.ndim != 3:
            raise ValueError("'b' should be a batch of matrices with shape (b, m, n).")
        if not A.dtype == torch.float64:
            raise ValueError("'A' should be a sparse float64 tensor.")
        if not b.dtype == torch.float64:
            raise ValueError("'b' should be a float64 tensor.")
        from torch_sparse_solve_cpp import solve_forward

        x = solve_forward(A, b)
        ctx.save_for_backward(A, b, x)
        return x

    @staticmethod
    def backward(ctx, grad):
        A, b, x = ctx.saved_tensors
        from torch_sparse_solve_cpp import solve_backward

        gradA, gradb = solve_backward(grad, A, b, x)
        return gradA, gradb
