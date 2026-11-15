"""Unit tests for config loader."""
from src.utils.config import get_config

def test_config_loads():
    cfg = get_config()
    assert cfg is not None

def test_camera_config_has_required_keys():
    cfg = get_config()["camera"]
    assert "device_id" in cfg
    assert "width" in cfg
    assert "height" in cfg

def test_decision_weights_sum_to_one():
    weights = get_config()["decision"]["weights"]
    assert abs(sum(weights.values()) - 1.0) < 0.001


def test_config_yaml_is_ascii():
    """Korean Windows reads files as cp949 by default; keep config.yaml ASCII-only."""
    from pathlib import Path
    raw = (Path(__file__).resolve().parents[2] / "config.yaml").read_bytes()
    assert raw.isascii()
