import numpy as np
import pytest
from PIL import Image
from floodnet.common import ProjectError
from floodnet.data import audit, prepare, Store
from floodnet.features import build_features
from floodnet.training import train_rf
from floodnet.evaluation import evaluate
from floodnet.export import export_web, validate_bundle, representatives


def test_fixture_pipeline_cache_reload_and_export(project):
    root, cfg = project
    cfg["expected_counts"] = {"train": 8, "val": 4, "test": 4}
    cfg["slic"]["n_segments"] = 20
    cfg["rf"]["n_estimators"] = 5
    for split, count in cfg["expected_counts"].items():
        for role in ("org", "label"):
            (root / f"data/raw/{split}/{split}-{role}-img").mkdir(parents=True)
        for i in range(count):
            truth = np.zeros((48, 64), dtype=np.uint8)
            truth[:, 20:42] = 1; truth[:, 42:] = 3
            rgb = np.empty((48, 64, 3), dtype=np.uint8)
            rgb[truth == 0] = [20+i, 130, 20]
            rgb[truth == 1] = [200, 20+i, 20]
            rgb[truth == 3] = [20, 20+i, 200]
            Image.fromarray(rgb).save(root / f"data/raw/{split}/{split}-org-img/{i}.jpg")
            Image.fromarray(truth).save(root / f"data/raw/{split}/{split}-label-img/{i}_lab.png")
    audited = audit(cfg)
    assert audited["counts"] == cfg["expected_counts"]
    first = prepare(cfg)
    second = prepare(cfg)
    assert first["identity"] == second["identity"]
    assert len(second["budgets"]["25"]["ids"]) == 2
    store = Store(cfg)
    with pytest.raises(ProjectError):
        store.select("test", 25)
    build_features(cfg)
    path = train_rf(cfg, 25, 42)
    evaluated = evaluate(cfg, path, "val")
    assert evaluated["image_count"] == 4
    assert evaluated["metrics"]["flooded_miou"] > .5
    assert train_rf(cfg, 25, 42, resume=True) == path
    # Changing an existing run's configuration must reject resume before fitting again.
    cfg["rf"]["max_depth"] = 1
    with pytest.raises(ProjectError, match="incompatible"):
        train_rf(cfg, 25, 42, resume=True)
    with pytest.raises(ProjectError, match="--final"):
        evaluate(cfg, path, "test")
    bundle = export_web(cfg)
    assert bundle["status"] == "incomplete" and not bundle["runs"]
    assert len(bundle["samples"]) == 4 and len(bundle["missing_runs"]) == 2 * 2 * len(cfg["train_seeds"])
    assert bundle["explorer_runs"] == [] and bundle["summary"] == []
    validate_bundle(bundle, root / "web")
    bundle["samples"][0]["rgb"] = "../escape.png"
    with pytest.raises(ProjectError):
        validate_bundle(bundle, root / "web")


def test_unpaired_and_unknown_masks_block_audit(project):
    root, cfg = project
    folder = root / "data/raw/train/train-org-img"
    folder.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(folder / "1.jpg")
    with pytest.raises(ProjectError, match="Unpaired"):
        audit(cfg)


def test_representatives_independent_of_model_scores():
    samples = [{"id": str(i), "presence": [bool(i % 2), bool(i % 3)], "target_fraction": i/100} for i in range(20)]
    chosen = representatives(samples)
    for sample in samples:
        sample["metrics"] = {"made_up": 1}
    assert len(chosen) == 6 and representatives(samples[::-1]) == chosen
