"""Hybrid U-Net + DenseCRF: refinement, validation-only selection, freezing and paired effects."""
import numpy as np
import pytest

from floodnet.common import ProjectError, write_json

PARAMS = {"iterations": 5, "gaussian": {"sxy": 3, "compat": 3}, "bilateral": {"sxy": 20, "srgb": 5, "compat": 10}}


def noisy_scene(seed=0):
    """Two colour halves (Other | flooded building) with 15% of pixels' probabilities flipped."""
    height, width = 40, 64
    rgb = np.zeros((height, width, 3), np.uint8)
    rgb[:, width // 2:] = [200, 40, 40]
    rgb[:, :width // 2] = [40, 160, 40]
    truth = np.zeros((height, width), np.uint8)
    truth[:, width // 2:] = 1
    probs = np.full((3, height, width), .1, np.float32)
    probs[1][truth == 1] = .8
    probs[0][truth == 0] = .8
    flip = np.random.default_rng(seed).random((height, width)) < .15
    probs[:, flip] = probs[:, flip][[1, 0, 2]]
    return probs / probs.sum(0, keepdims=True), rgb, truth


def test_refine_removes_label_noise_deterministically():
    from floodnet.hybrid import refine_valid
    probs, rgb, truth = noisy_scene()
    assert (probs.argmax(0) == truth).mean() < .9
    first = refine_valid(probs, rgb, PARAMS)
    assert (first == truth).mean() > .99
    np.testing.assert_array_equal(first, refine_valid(probs, rgb, PARAMS))


def test_refine_only_changes_the_valid_rectangle():
    from floodnet.hybrid import refine
    probs, rgb, _ = noisy_scene()
    padded_probs = np.zeros((3, 64, 64), np.float32)
    padded_probs[2] = 1  # padding predicts "road"; refinement must leave it alone
    padded_probs[:, 12:52] = probs
    padded_rgb = np.zeros((64, 64, 3), np.uint8)
    padded_rgb[12:52] = rgb
    labels = refine(padded_probs, padded_rgb, {"top": 12, "left": 0, "height": 40, "width": 64}, PARAMS)
    assert np.all(labels[:12] == 2) and np.all(labels[52:] == 2)
    assert set(np.unique(labels[12:52]).tolist()) <= {0, 1}


def test_selection_uses_mean_gain_across_runs_and_keeps_earlier_ties():
    from floodnet.hybrid import select
    settings = [{"bilateral": {"sxy": s}} for s in (20, 50, 80)]
    # Per-run bests differ (setting 1 wins run a); the rule averages across runs, so setting 0 wins.
    rounds = [{"settings": settings, "per_run": {"a": {"gain": [.03, .05, .04]}, "b": {"gain": [.03, -.01, 0.]}}}]
    best = select(rounds)
    assert best["params"] == settings[0] and best["mean_gain"] == pytest.approx(.03)
    tie = [{"settings": settings, "per_run": {"a": {"gain": [.02, .02, .01]}}}]
    assert select(tie)["params"] == settings[0]


def fake_tuning(monkeypatch, project, gains):
    import floodnet.hybrid as hybrid
    _, cfg = project
    calls = []

    def evaluate_settings(cfg, store, settings, hcfg):
        calls.append(settings)
        values = gains[len(calls) - 1]
        return {"unet_25_s42": {"budget": 25, "seed": 42, "unet_flooded_miou": .5, "flooded_miou": [.5 + g for g in values],
                                "gain": values, "seconds": 0.}}

    monkeypatch.setattr(hybrid, "Store", lambda cfg: type("S", (), {"manifest": {"identity": "p"}})())
    monkeypatch.setattr(hybrid, "selection_key", lambda cfg, hcfg, store: "key")
    monkeypatch.setattr(hybrid, "evaluate_settings", evaluate_settings)
    write_json("runs/final_experiment.json", {"identity": "base"})
    hcfg = {"backend": "densecrf", "iterations": 5, "gaussian": {"sxy": 3, "compat": 3},
            "bilateral_grid": {"sxy": [20, 50], "srgb": [5]}, "bilateral_compat": 10, "weaker_compat": [5, 3], "workers": 1}
    return hybrid.tune(cfg, hcfg), calls


def test_tuning_adopts_a_helpful_setting_without_a_second_round(project, monkeypatch):
    selection, calls = fake_tuning(monkeypatch, project, [[.02, .01]])
    assert len(calls) == 1 and selection["adopted"]
    assert selection["selected"]["bilateral"] == {"sxy": 20, "srgb": 5, "compat": 10}


def test_tuning_tries_weaker_smoothing_only_when_refinement_hurts(project, monkeypatch):
    selection, calls = fake_tuning(monkeypatch, project, [[-.03, -.01], [-.002, -.004]])
    assert len(calls) == 2
    assert [s["bilateral"]["compat"] for s in calls[1]] == [5, 3]
    assert calls[1][0]["bilateral"]["sxy"] == 50  # weaker strength at the best first-round setting
    assert selection["selected"]["bilateral"] == {"sxy": 50, "srgb": 5, "compat": 5}
    assert not selection["adopted"]


def test_freeze_hybrid_requires_every_recorded_and_validated_run(project):
    from floodnet.hybrid import freeze_hybrid
    _, cfg = project
    store = type("S", (), {"select": lambda self, split: []})()
    with pytest.raises(FileNotFoundError):
        freeze_hybrid(cfg, store)
    write_json("runs/final_experiment.json", {"identity": "base", "runs": []})
    write_json("runs/hybrid_selection.json", {"identity": "sel", "selected": PARAMS, "adopted": True,
                                               "rounds": [{"settings": [PARAMS], "per_run": {}}]})
    with pytest.raises(ProjectError, match="missing hybrid_25_s42"):
        freeze_hybrid(cfg, store)


def test_completed_run_reuse_ignores_source_changes_but_not_configuration(project):
    from floodnet.common import sha256
    from floodnet.training import reusable_complete_run
    root, _ = project
    directory = root / "runs/unet_25_s42"
    directory.mkdir(parents=True)
    (directory / "best.pt").write_bytes(b"weights")
    identity = {"config_hash": "c", "method": "unet", "budget": 25, "seed": 42, "source_identity": "old"}
    old = {"status": "complete", "identity": identity, "checkpoint": "best.pt", "checkpoint_sha256": sha256(directory / "best.pt")}
    assert reusable_complete_run(directory, old, dict(identity, source_identity="new"))
    assert not reusable_complete_run(directory, old, dict(identity, config_hash="changed"))
    assert not reusable_complete_run(directory, dict(old, status="running"), identity)
    (directory / "best.pt").write_bytes(b"tampered")
    with pytest.raises(ProjectError, match="checkpoint changed"):
        reusable_complete_run(directory, old, identity)


def test_crf_effects_pair_each_hybrid_with_the_unet_run_it_refines():
    from floodnet.export import crf_effects

    def run(method, budget, seed, value):
        per_class = [{"iou": value, "dice": value, "recall": value}] * 3
        return {"method": method, "budget": budget, "seed": seed,
                "metrics": {"flooded_miou": value, "miou": value, "per_class": per_class}}
    runs = [run("unet", 25, s, .5 + s / 100) for s in (1, 2)] + [run("hybrid", 25, s, .52 + s / 100) for s in (1, 2)]
    effects = crf_effects(runs, [1, 2])
    assert [e["budget"] for e in effects] == [25]
    assert effects[0]["full"]["flooded_miou"]["values"] == pytest.approx([.02, .02])
    assert crf_effects(runs[:-1], [1, 2]) == []
