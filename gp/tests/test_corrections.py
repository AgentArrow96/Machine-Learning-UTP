"""Regression tests for the v2 corrections: RF sampling, overlap, seeds and clean-subset scores."""
import numpy as np
import pytest

from floodnet.common import ProjectError, write_json
from floodnet.metrics import confusion, score


class FakeStore:
    def __init__(self, records, identity="fake-processed"):
        self.records = records
        self.manifest = {"identity": identity, "budgets": {"25": {"identity": "b25", "ids": []}, "100": {"identity": "b100", "ids": []}}}

    def select(self, split, budget=None):
        return [r for r in self.records if r["split"] == split]


def write_feature_cache(cfg, store, labels_per_image):
    from floodnet.features import feature_key, feature_path
    key = feature_key(cfg, store)
    for record, labels in zip(store.records, labels_per_image):
        path = feature_path(cfg, store, record)
        path.parent.mkdir(parents=True, exist_ok=True)
        y = np.asarray(labels, dtype=np.uint8)
        x = np.random.default_rng(len(y)).random((len(y), 22), dtype=np.float32)
        np.savez_compressed(path, x=x, y=y, regions=np.zeros((4, 4), np.int32), identity=key)


def test_keep_targets_keeps_every_target_region_within_cap(project):
    from floodnet.features import training_table
    _, cfg = project
    records = [{"id": f"train_{i}", "split": "train"} for i in range(3)]
    store = FakeStore(records)
    # 30 regions per image: 4 building, 3 road, 23 other -> 21 targets, 69 other in total.
    write_feature_cache(cfg, store, [[1] * 4 + [2] * 3 + [0] * 23] * 3)
    cfg["rf"]["max_regions"] = 50
    cfg["rf"]["region_sampling"] = "keep_targets"
    x, y, references = training_table(cfg, store, 100, 42)
    assert len(y) == 50 and np.bincount(y, minlength=3).tolist() == [29, 12, 9]
    assert len({tuple(r) for r in references}) == 50
    again = training_table(cfg, store, 100, 42)
    assert again[2] == references and np.array_equal(again[0], x)
    # Other regions are drawn round-robin, so no image dominates the remainder.
    other_images = [image for (image, _), label in zip(references, y) if label == 0]
    assert max(other_images.count(r["id"]) for r in records) - min(other_images.count(r["id"]) for r in records) <= 1
    cfg["rf"]["max_regions"] = 20
    with pytest.raises(ProjectError, match="exceed"):
        training_table(cfg, store, 100, 42)


def test_uniform_sampling_still_available_and_seed_dependent(project):
    from floodnet.features import training_table
    _, cfg = project
    records = [{"id": f"train_{i}", "split": "train"} for i in range(3)]
    store = FakeStore(records)
    write_feature_cache(cfg, store, [[1] * 4 + [2] * 3 + [0] * 23] * 3)
    cfg["rf"].update(max_regions=50, region_sampling="uniform")
    _, y, first = training_table(cfg, store, 100, 42)
    _, _, second = training_table(cfg, store, 100, 43)
    assert len(y) == 50 and first != second


def test_overlap_verifies_shifted_crop_and_rejects_unrelated_scene(cfg):
    from skimage import data
    from floodnet.overlap import orb_features, verify_pair
    camera = data.camera().astype(np.float32) / 255
    settings = cfg["overlap"]
    def features(grey):
        points, descriptors, shape = orb_features(grey, settings["keypoints"])
        return {"points": points, "descriptors": descriptors, "shape": tuple(shape)}
    evaluated, train = features(camera[:, :384]), features(camera[:, 128:])
    result = verify_pair(evaluated, train, settings, 42)
    assert result["verified"] and result["overlap"] == pytest.approx(256 / 384, abs=.06)
    unrelated = features(data.coins().astype(np.float32) / 255)
    assert not verify_pair(evaluated, unrelated, settings, 42)["verified"]


def test_footprint_accepts_camera_motion_and_rejects_mirrored_homography():
    from skimage.transform import ProjectiveTransform
    from floodnet.overlap import footprint_overlap
    assert footprint_overlap(ProjectiveTransform(), (100, 200), (100, 200)) == pytest.approx(1)
    shifted = ProjectiveTransform(np.array([[1., 0, 100], [0, 1, 0], [0, 0, 1]]))
    assert footprint_overlap(shifted, (100, 200), (100, 200)) == pytest.approx(.5, abs=.02)
    # A mirror image cannot come from two views of the same ground plane.
    mirrored = ProjectiveTransform(np.array([[0., 1, 0], [1, 0, 0], [0, 0, 1]]))
    assert footprint_overlap(mirrored, (100, 100), (100, 100)) == 0


def test_subset_metrics_pool_saved_confusion_counts():
    from floodnet.evaluation import subset_metrics
    truth = [np.array([[0, 1], [2, 2]]), np.array([[1, 1], [0, 255]]), np.array([[2, 0], [0, 0]])]
    predicted = [np.array([[0, 1], [2, 0]]), np.array([[1, 0], [0, 0]]), np.array([[0, 0], [1, 0]])]
    per_image = [{"id": f"test_{i}", "metrics": score(confusion(t, p))} for i, (t, p) in enumerate(zip(truth, predicted))]
    pooled = score(sum(confusion(t, p) for t, p in zip(truth, predicted)))
    assert subset_metrics(per_image, ["test_0", "test_1", "test_2"]) == pooled
    assert subset_metrics(per_image, ["test_1"]) == per_image[1]["metrics"]
    with pytest.raises(ProjectError, match="missing"):
        subset_metrics(per_image, ["test_9"])


def test_freeze_requires_overlap_and_every_seed(project):
    from floodnet.evaluation import final_runs, freeze_experiment
    _, cfg = project
    store = FakeStore([])
    with pytest.raises(ProjectError, match="overlap"):
        freeze_experiment(cfg, store)
    write_json("data/manifests/overlap.json", {"identity": "o", "processed_identity": store.manifest["identity"]})
    with pytest.raises(ProjectError, match="missing rf_25_s42"):
        freeze_experiment(cfg, store)
    assert len(final_runs(cfg)) == 4 * len(cfg["train_seeds"])
    assert {seed for _, _, seed in final_runs(cfg)} == set(cfg["train_seeds"])


def test_rf_engine_predicts_single_threaded(tmp_path):
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from floodnet.evaluation import RFEngine
    rng = np.random.default_rng(0)
    x, y = rng.random((90, 22)), np.repeat([0, 1, 2], 30)
    model = RandomForestClassifier(n_estimators=8, random_state=0, n_jobs=4).fit(x, y)
    joblib.dump(model, tmp_path / "model.joblib")
    engine = RFEngine(tmp_path, {"config": {"slic": {}}, "checkpoint": "model.joblib"})
    assert engine.model.n_jobs == 1
    np.testing.assert_array_equal(engine.model.predict(x), model.predict(x))


def test_seed_summary_and_paired_budget_effect():
    from floodnet.export import stats, summarise
    assert stats([.5, .6, .7])["mean"] == pytest.approx(.6) and stats([.5, .6, .7])["sd"] == pytest.approx(.1)
    assert stats([.5, None])["mean"] is None and stats([.4])["sd"] is None
    def metrics(value):
        per_class = [{"iou": value, "dice": value, "recall": value}] * 3
        return {"flooded_miou": value, "miou": value, "per_class": per_class}
    runs = [{"id": f"unet_{b}_s{s}", "method": "unet", "budget": b, "seed": s, "metrics": metrics(base + s / 100),
             "clean_metrics": metrics(base), "training_seconds": 1., "timing": {"median_seconds": .01}, "epochs": 5, "best_epoch": 3}
            for b, base in ((25, .4), (100, .5)) for s in (1, 2, 3)]
    summary, effects = summarise(runs, [1, 2, 3])
    assert [(s["method"], s["budget"], s["n"]) for s in summary] == [("unet", 25, 3), ("unet", 100, 3)]
    assert summary[0]["full"]["flooded_miou"]["mean"] == pytest.approx(.42)
    assert effects[0]["full"]["values"] == pytest.approx([.1, .1, .1]) and effects[0]["clean"]["mean"] == pytest.approx(.1)
    # A missing seed leaves that group out instead of averaging an incomplete set.
    assert summarise(runs[:-1], [1, 2, 3])[1] == []


def test_target_strata_split_target_images_at_median_overlap():
    from floodnet.export import target_strata
    counts = {"test_0": [9, 1, 0], "test_1": [9, 0, 1], "test_2": [9, 1, 1], "test_3": [9, 2, 0], "test_4": [10, 0, 0]}
    store = FakeStore([{"id": i, "split": "test", "processed_counts": c} for i, c in counts.items()])
    overlap = {"per_image": {"test_0": {"max_train_overlap": .9}, "test_1": {"max_train_overlap": 0.},
                             "test_2": {"max_train_overlap": .5}, "test_3": {"max_train_overlap": .7},
                             "test_4": {"max_train_overlap": .95}}}
    strata = target_strata(store, overlap)
    # test_4 has no flooded target, so it is outside both halves regardless of its overlap.
    assert strata["threshold"] == pytest.approx(.6)
    assert strata["low"] == ["test_1", "test_2"] and strata["high"] == ["test_0", "test_3"]
