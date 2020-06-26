""" tests for torch_sparse_solve """

import torch
import pytest
import numpy as np
from torch_sparse_solve import solve as sparse_solve

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
    torch.autograd.gradcheck(sparse_solve, [W.double(), b.double()])

def test_result(W, b):
    """ confirm result """
    x = sparse_solve(W, b)
    b2 = torch.bmm(W, x)
    np.testing.assert_almost_equal(
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
    x_sparse = sparse_solve(W, b)
    x_dense = torch.solve(b, W)[0]
    np.testing.assert_almost_equal(x_sparse.data.cpu().numpy(), x_dense.data.cpu().numpy())

def test_cuda_comparison_with_torch_solve(W, b):
    """ compare with dense torch.solve """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA available")
    test_comparison_with_torch_solve(W.cuda(), b.cuda())

