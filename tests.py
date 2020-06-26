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
        b2.data.cpu().numpy(),
        b.data.cpu().numpy(),
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
    numpy.testing.assert_almost_equal(x_sparse.data.cpu().numpy(), x_dense.data.cpu().numpy())

def test_cuda_comparison_with_torch_solve(W, b):
    """ compare with dense torch.solve """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    test_comparison_with_torch_solve(W.cuda(), b.cuda())

def test_sparse_solver(W, b): # <- failing
    W_np = W[0].data.cpu().numpy()
    b_np = W[0, :, 0].data.cpu().numpy()
    x_np = b_np.copy()
    W_sp = scipy.sparse.csc_matrix(W_np)
    Wp = W_sp.indptr
    Wi = W_sp.indices
    Wx = W_sp.data
    n = Wp.size - 1
    c_Wp = numpy.ctypeslib.as_ctypes(Wp)
    c_Wi = numpy.ctypeslib.as_ctypes(Wi)
    c_Wx = numpy.ctypeslib.as_ctypes(Wx)
    c_b = numpy.ctypeslib.as_ctypes(x_np)
    torch_sparse_solve_cpp._sparse_solve(n, c_Wp, c_Wi, c_Wx, c_b)
    assert (W_np@x_np - b_np < 1e-5).all()




