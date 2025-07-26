import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fftnet_block import FFTNetBlock


def test_block_forward_cpu():
    block = FFTNetBlock(dim=4)
    x = torch.randn(2, 6, 4)
    out = block(x)
    assert out.shape == (2, 6, 4)
    assert not out.is_complex()


def test_block_forward_gpu():
    if not torch.cuda.is_available():
        return
    block = FFTNetBlock(dim=4).cuda()
    x = torch.randn(2, 6, 4, device='cuda')
    out = block(x)
    assert out.shape == (2, 6, 4)
    assert out.device.type == 'cuda'
