import json

from fftnet.utils.config import build_model, load_config


def test_build_model_from_yaml(tmp_path):
    cfg_json = tmp_path / "cfg.json"
    modules_yaml = tmp_path / "modules.yaml"
    cfg_json.write_text(json.dumps({"vocab_size": 10, "dim": 4}))
    modules_yaml.write_text(
        "blocks:\n" "  - type: FFTNetBlock\n" "    name: block0\n" "    base: 123\n" "  - type: FFTNetBlock\n" "    name: block1\n"
    )

    model, cfg = build_model(str(cfg_json), str(modules_yaml))
    assert cfg["num_blocks"] == 2
    assert len(model.blocks) == 2
    assert model.blocks[0].rope.base == 123

    loaded_cfg = load_config(str(cfg_json), str(modules_yaml))
    assert loaded_cfg["blocks"][1]["name"] == "block1"
