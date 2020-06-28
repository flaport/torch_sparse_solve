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
def W(gen):
    """ default weight tensor """
    return torch.randn(4, 3, 3, dtype=torch.float64, requires_grad=True, generator=gen)


@pytest.fixture
def b(gen):
    """ default target tensor """
    return torch.randn(4, 3, 2, dtype=torch.float64, generator=gen)


def test_gradcheck(W, b):
    """ check if backward grads are correct """
    torch.autograd.gradcheck(torch_sparse_solve.solve, [W.double(), b.double()])


def test_result(W, b):
    """ confirm result """
    x = torch_sparse_solve.solve(W, b)
    b2 = torch.bmm(W, x)
    numpy.testing.assert_almost_equal(
        b2.data.cpu().numpy(), b.data.cpu().numpy(),
    )


def test_cuda_result(W, b):
    """ confirm result """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    test_result(W.cuda(), b.cuda())


def test_comparison_with_torch_solve(W, b):
    """ compare with dense torch.solve """
    x_sparse = torch_sparse_solve.solve(W, b)
    x_dense = torch.solve(b, W)[0]
    numpy.testing.assert_almost_equal(
        x_sparse.data.cpu().numpy(), x_dense.data.cpu().numpy()
    )


def test_cuda_comparison_with_torch_solve(W, b):
    """ compare with dense torch.solve """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    test_comparison_with_torch_solve(W.cuda(), b.cuda())


def test_coo_to_csc():
    A = torch.tensor(
        [[0, 0, 0, 0], [5, 8, 0, 0], [0, 0, 3, 0], [0, 6, 0, 0],],
        dtype=torch.float64,
        device="cpu",
    )

    Ap, Aj, Ax = torch_sparse_solve_cpp._coo_to_csc(A.to_sparse())
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


def test_sparse_solver(W, b):
    target = torch.solve(b, W)[0][0, :, 0]
    result = b[0, :, 0].clone()
    Wp, Wi, Wx = torch_sparse_solve_cpp._coo_to_csc(W[0].to_sparse())
    torch_sparse_solve_cpp._sparse_solve(Wp, Wi, Wx, result)
    assert (target - result < 1e-5).all()


if __name__ == "__main__":
    pytest.main([__file__])
