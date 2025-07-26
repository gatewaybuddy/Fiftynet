import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import FFTNet


def test_forward_cpu():
    model = FFTNet(vocab_size=10, dim=4, num_blocks=2)
    input_ids = torch.randint(0, 10, (2, 5))
    out = model(input_ids)
    assert out.shape == (2, 5, 10)


def test_forward_gpu():
    if not torch.cuda.is_available():
        return
    model = FFTNet(vocab_size=10, dim=4, num_blocks=2).cuda()
    input_ids = torch.randint(0, 10, (2, 5), device='cuda')
    out = model(input_ids)
    assert out.shape == (2, 5, 10)
    assert out.device.type == 'cuda'
