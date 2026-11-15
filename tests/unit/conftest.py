"""Shared pytest fixtures for unit tests."""
import copy

import pytest

import src.utils.config as config_module


@pytest.fixture
def cfg():
    """
    A private, editable copy of config.yaml for one test.
    Changes (e.g. cfg['detectors']['roi'] = [...]) are undone after the test.
    """
    original = config_module.get_config()
    config_module._cfg = copy.deepcopy(original)
    yield config_module._cfg
    config_module._cfg = original
