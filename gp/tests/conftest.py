import copy
import pytest
from floodnet.common import config


@pytest.fixture
def cfg():
    return copy.deepcopy(config())


@pytest.fixture
def project(tmp_path, monkeypatch, cfg):
    import floodnet.common as common
    import floodnet.training as training
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(training, "ROOT", tmp_path)
    (tmp_path / "src/floodnet").mkdir(parents=True)
    return tmp_path, cfg
