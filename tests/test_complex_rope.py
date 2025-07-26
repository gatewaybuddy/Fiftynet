import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from complex_rope import ComplexRoPE


def test_forward_shape_cpu():
    rope = ComplexRoPE(dim=4)
    x = torch.randn(2, 3, 4)
    out = rope(x)
    assert out.shape == (2, 3, 2)
    assert out.is_complex()


def test_forward_gpu():
    if not torch.cuda.is_available():
        return
    rope = ComplexRoPE(dim=4).cuda()
    x = torch.randn(2, 3, 4, device='cuda')
    out = rope(x)
    assert out.shape == (2, 3, 2)
    assert out.is_complex()

