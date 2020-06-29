""" tests for torch_sparse_solve """

import numpy
import torch
import ctypes
import pytest
import scipy.sparse
import torch_sparse_solve
import torch_sparse_solve_cpp


@pytest.fixture
def gen():
    """ default pytorch random number generator """
    return torch.Generator(device="cpu").manual_seed(42)


@pytest.fixture
def A(gen):
    """ default batched sparse tensor """
    mask = torch.tensor([[[1, 0, 0], [1, 1, 0], [0, 0, 1]]], dtype=torch.float64)
    A = mask * torch.randn(4, 3, 3, dtype=torch.float64, generator=gen)
    Asp = A.to_sparse()
    Asp.requires_grad_()
    return Asp


@pytest.fixture
def b(gen):
    """ default target tensor """
    return torch.randn(4, 3, 2, dtype=torch.float64, generator=gen)


def test_gradcheck(A, b):
    """ check if backward grads are correct """
    torch.autograd.gradcheck(torch_sparse_solve.solve, [A, b], check_sparse_nnz=True)


def test_result(A, b):
    """ confirm result """
    x = torch_sparse_solve.solve(A, b)
    b2 = torch.bmm(A.to_dense(), x)
    numpy.testing.assert_almost_equal(
        b2.data.cpu().numpy(), b.data.cpu().numpy(),
    )


def test_comparison_with_torch_solve(A, b):
    """ compare with dense torch.solve """
    x_sparse = torch_sparse_solve.solve(A, b)
    x_dense = torch.solve(b, A.to_dense())[0]
    numpy.testing.assert_almost_equal(
        x_sparse.data.cpu().numpy(), x_dense.data.cpu().numpy()
    )


def test_coo_to_csc():
    A = torch.tensor(
        [[0, 0, 0, 0], [5, 8, 0, 0], [0, 0, 3, 0], [0, 6, 0, 0],],
        dtype=torch.float64,
        device="cpu",
    )
    Asp = A.to_sparse()
    Ai = Asp._indices()[0].int()
    Aj = Asp._indices()[1].int()
    Ax = Asp._values()

    Ap, Aj, Ax = torch_sparse_solve_cpp._coo_to_csc(A.size(1), Ai, Aj, Ax)
    Ap = Ap.data.cpu().numpy()
    Aj = Aj.data.cpu().numpy()
    Ax = Ax.data.cpu().numpy()

    a = scipy.sparse.csc_matrix(A.detach().cpu().numpy())
    ap = numpy.asarray(a.indptr, dtype=numpy.int64)
    aj = numpy.asarray(a.indices, dtype=numpy.int64)
    ax = numpy.asarray(a.data, dtype=numpy.float64)

    assert (Ap == ap).all()
    assert (Aj == aj).all()
    assert (Ax == ax).all()


def test_sparse_solver(A, b):
    target = torch.solve(b, A.to_dense())[0][0, :, 0]
    result = b[0, :, 0].clone()
    Ai = A[0]._indices()[0].int()
    Aj = A[0]._indices()[1].int()
    Ax = A[0]._values()
    Ap, Ai, Ax = torch_sparse_solve_cpp._coo_to_csc(A[0].size(1), Ai, Aj, Ax)
    torch_sparse_solve_cpp._klu_solve(Ap, Ai, Ax, result)
    assert (target - result < 1e-5).all()


if __name__ == "__main__":
    pytest.main([__file__])
