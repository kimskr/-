"""Shared pytest fixtures for unit tests."""
import copy

import pytest

import src.utils.config as config_module


@pytest.fixture(autouse=True)
def cfg():
    """
    A private, editable copy of config.yaml for every test.

    Site-specific settings (detectors.roi, which depends on where the camera and
    part are on someone's desk) are reset so unit tests on synthetic images
    behave the same on every machine. Tests can still change values, e.g.
    cfg['detectors']['roi'] = [...]; everything is undone after the test.
    """
    original = config_module.get_config()
    config_module._cfg = copy.deepcopy(original)
    config_module._cfg["detectors"]["roi"] = None
    yield config_module._cfg
    config_module._cfg = original
