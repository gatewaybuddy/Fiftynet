import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_fourier_operator import NeuralFourierOperator


def test_forward_shape_cpu():
    op = NeuralFourierOperator(dim=4)
    x = torch.randn(2, 3, 4, dtype=torch.cfloat)
    out = op(x)
    assert out.shape == (2, 3, 4)
    assert out.dtype == torch.cfloat


def test_forward_gpu():
    if not torch.cuda.is_available():
        return
    op = NeuralFourierOperator(dim=4).cuda()
    x = torch.randn(2, 3, 4, dtype=torch.cfloat, device='cuda')
    out = op(x)
    assert out.shape == (2, 3, 4)
    assert out.is_cuda
