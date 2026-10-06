"""The generator, the anomaly model, the thresholds and the review selection, without a database."""
import numpy as np

from qi import engine as E, world as W


def test_frames_are_reproducible_and_defects_have_masks():
    a, _ = W.frame("pcb", 3, 7)
    b, _ = W.frame("pcb", 3, 7)
    assert np.array_equal(a, b) and a.shape == (W.SIZE, W.SIZE) and 0 <= a.min() and a.max() <= 1
    for prod, kinds in W.DEFECTS.items():
        for k in kinds:
            img, m = W.frame(prod, 3, 11, k)
            assert m is not None and m.sum() > 0, (prod, k)
    plan = W.stream_plan(5, 2000, rate=0.05, supplier_from=1000)
    assert 0.03 < np.mean([p["defect"] is not None for p in plan]) < 0.07
    assert not any(p["supplier_b"] for p in plan[:1000]) and 0.6 < np.mean([p["supplier_b"] for p in plan[1000:]]) < 0.8


def test_lighting_is_normalised_away():
    img, _ = W.frame("pcb", 3, 1)
    assert np.allclose(E.normalise(img), E.normalise(img * 1.15 + 0.04), atol=1e-5)


def test_coreset_keeps_the_far_points():
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(0, 0.01, (1, 500, 3)), np.full((1, 1, 3), 10.0)], axis=1)
    assert (np.abs(E.coreset(x, 5, rng)[0]) > 5).any()      # the outlier is always a centre


def test_model_beats_the_template_and_holds_its_false_reject_target():
    ts = W.training_set("pcb", 14)
    m = E.Model().fit(ts["good"], ts["defects"], ts["defect_types"], ts["calib"], seed=1)
    plan = W.stream_plan(9, 800, rate=0.15)
    imgs, _ = W.render_plan("pcb", 9, plan)
    y = np.array([p["defect"] is not None for p in plan])
    s, b = m.scores(imgs), m.baseline_scores(imgs)
    assert E.auroc(s[~y], s[y]) > E.auroc(b[~y], b[y]) + 0.1
    assert (s[~y] > m.threshold).mean() < 0.04
    kinds = np.array([p["defect"] or "" for p in plan])
    assert (s[kinds == "missing_component"] > m.threshold).all()
    o = m.inspect(imgs[np.nonzero(kinds == "missing_component")[0][0]])
    assert o["reject"] and o["type"] == "missing_component" and 0.5 < o["confidence"] <= 1


def test_adaptation_is_local_and_fixes_the_new_part():
    ts = W.training_set("pcb", 14)
    m1 = E.Model().fit(ts["good"], ts["defects"], ts["defect_types"], ts["calib"], seed=1)
    plan = W.stream_plan(4, 120, rate=0.0, supplier_from=0)
    imgs, _ = W.render_plan("pcb", 4, plan)
    b = imgs[np.array([p["supplier_b"] for p in plan])]
    m2 = E.Model().fit(ts["good"], ts["defects"], ts["defect_types"], ts["calib"], seed=1, extra_good=b[:30])
    held = b[30:]
    assert (m1.scores(held) > m1.threshold).mean() > 0.9 and (m2.scores(held) > m2.threshold).mean() < 0.6
    assert 0 < m2.adapted_positions < E.GRID * E.GRID / 3


def test_metrics_and_review_selection():
    r = E.rates([1, 0, 0, 1], [1, 1, 0, 0])
    assert r["false_rejects"] == 1 and r["escapes"] == 1 and r["yield"] == 0.5
    lo, hi = E.wilson(0, 30)
    assert lo == 0 and 0.05 < hi < 0.12
    assert E.auroc(np.zeros(5), np.ones(5)) == 1.0
    rng = np.random.default_rng(1)
    feats = np.concatenate([rng.normal(0, 1, (90, 4)), rng.normal(8, 1, (10, 4))])
    pick = E.select_for_review(feats, rng.random(100), 0.5, 20, k=2)
    assert len(set(pick)) == 20 and any(i >= 90 for i in pick)     # the small cluster is not crowded out
