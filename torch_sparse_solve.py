""" A sparse KLU solver for PyTorch """

import torch

__version__ = "0.0.2"
__author__ = "Floris Laporte"
__all__ = ["solve"]


def solve(A, b):
    """ solve a sparse system Ax = b

    Args:
        A (torch.sparse.Tensor[b, m, m]): the sparse matrix defining the system.
        b (torch.Tensor[b, m, n]): the target matrix b

    Returns:
        x (torch.Tensor[b, m, n]): the initially unknown matrix x
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
        from torch_sparse_solve_cpp import forward

        x = forward(A, b)
        ctx.save_for_backward(A, b, x)
        return x

    @staticmethod
    def backward(ctx, grad):
        A, b, x = ctx.saved_tensors
        from torch_sparse_solve_cpp import backward

        gradA, gradb = backward(grad, A, b, x)
        return gradA, gradb


if __name__ == "__main__":
    A = torch.randn(2, 3, 3, requires_grad=True)
    b = torch.randn(2, 3, 2)
    test = torch.autograd.gradcheck(
        Solve.apply, [A.double(), b.double()]
    )  # gradcheck requires double precision
    print(A @ solve(A, b))
    print(b)
    print((torch.abs(A @ solve(A, b) - b) < 1e-5).all().item())
