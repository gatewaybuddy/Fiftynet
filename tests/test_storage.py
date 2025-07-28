import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from fftnet.utils.storage import save_model, load_model
import manage_models
from model import FFTNet


def test_save_and_load(tmp_path):
    path = tmp_path / 'weights' / 'ver'
    config = {'vocab_size': 10, 'dim': 16, 'num_blocks': 1}
    model = FFTNet(**config)
    save_model(model, str(path), config)
    loaded_model, loaded_cfg = load_model(str(path))
    assert loaded_cfg == config
    assert set(model.state_dict().keys()) == set(loaded_model.state_dict().keys())


def test_list_and_delete(tmp_path, monkeypatch):
    weights_dir = tmp_path / 'weights'
    monkeypatch.setattr(manage_models, 'WEIGHTS_DIR', weights_dir)
    config = {'vocab_size': 10, 'dim': 16, 'num_blocks': 1}
    model = FFTNet(**config)
    save_model(model, str(weights_dir / 'v1'), config)
    assert 'v1' in manage_models.list_models()
    manage_models.delete_model('v1')
    assert 'v1' not in manage_models.list_models()
