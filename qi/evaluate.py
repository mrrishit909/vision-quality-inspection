"""Held-out evaluation: lines the demo never uses (seeds 21, 31, 41), all three products. Each line trains the way the demo
does (200 good parts, 8 labelled defects per type, 150 good parts for the threshold) and is scored on a fresh 3,000-frame
stream with 10% defects against the simulation truth, beside the golden-template baseline. Then the supplier change on the
board line: what 40 reviewed rejects buy, by how they were chosen.

    python -m qi.evaluate > docs/evaluation.md
"""
import time

import numpy as np

from . import engine as E, world as W

SEEDS = (21, 31, 41)


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    t0 = time.time()
    out = ["# Evaluation", "", f"Lines the demo never uses (seeds {', '.join(map(str, SEEDS))}). Training as in the demo; each line scored on a fresh",
           "3,000-frame stream with 10% defects. Thresholds set for 1% false rejects on held-back good parts. Reproduce with `python -m qi.evaluate`.", "",
           "## Detection, against the simulation truth", "",
           "| Product | Seed | AUROC model / template | False rejects model / template | Escapes model / template | Escapes by type (model) | Type right, of caught | Box on the defect |",
           "|---|---|---|---|---|---|---|---|"]
    lat = []
    for prod in W.PRODUCTS:
        for seed in SEEDS:
            ts = W.training_set(prod, seed)
            m = E.Model().fit(ts["good"], ts["defects"], ts["defect_types"], ts["calib"], seed=1)
            plan = W.stream_plan(seed + 500, 3000, prod, rate=0.1)
            imgs, masks = W.render_plan(prod, seed + 500, plan)
            imgs = np.round(imgs * 255).astype(np.uint8).astype(np.float32) / 255
            y = np.array([p["defect"] is not None for p in plan])
            kinds = np.array([p["defect"] or "" for p in plan])
            s, b = m.scores(imgs), m.baseline_scores(imgs)
            r, rb = E.rates(s > m.threshold, y), E.rates(b > m.base_threshold, y)
            caught = np.nonzero(y & (s > m.threshold))[0]
            right = hit = 0
            for i in caught:
                o = m.inspect(imgs[i])
                right += o["type"] == kinds[i]
                x, yy, w, h = o["bbox"]
                hit += bool(masks[i][yy:yy + h, x:x + w].any())
            esc = ", ".join(f"{k.replace('_', ' ')} {int(((s <= m.threshold) & (kinds == k)).sum())}/{int((kinds == k).sum())}" for k in W.DEFECTS[prod])
            out.append(f"| {prod} | {seed} | {E.auroc(s[~y], s[y]):.3f} / {E.auroc(b[~y], b[y]):.3f} | {pct(r['false_reject_rate'])} / {pct(rb['false_reject_rate'])} | {pct(r['escape_rate'])} / {pct(rb['escape_rate'])} | {esc} | {right}/{len(caught)} | {hit}/{len(caught)} |")
            for i in range(300):
                t = time.perf_counter()
                m.inspect(imgs[i])
                lat.append((time.perf_counter() - t) * 1000)
    lat = np.array(lat)
    out += ["", f"Single-frame inspection on one core, {len(lat):,} frames: p50 {np.percentile(lat, 50):.2f} ms, p95 {np.percentile(lat, 95):.2f} ms, p99 {np.percentile(lat, 99):.2f} ms against the 33.3 ms a 30 FPS frame allows.",
            "", "## The supplier change: what 40 reviewed rejects buy", "",
            "Board lines. A 3,000-frame window with the second supplier's capacitors from frame 1,500 is inspected by version 1; 40 of its rejects are reviewed",
            "(labels from the truth) and version 2 adds the reviewed good boards where they looked wrong. Both versions then inspect a fresh 3,000 frames with the",
            "new part on 70% of boards. Escapes are split by supplier because version 1 rejects every new-supplier board, which catches their defects for the wrong reason.", "",
            "| Seed | Chosen by | Good boards reviewed | False rejects, new-supplier boards | False rejects, old-supplier boards | Escapes, old-supplier boards | Escapes, new-supplier boards |",
            "|---|---|---|---|---|---|---|"]
    for seed in SEEDS:
        ts = W.training_set("pcb", seed)
        m1 = E.Model().fit(ts["good"], ts["defects"], ts["defect_types"], ts["calib"], seed=1)
        plan = W.stream_plan(seed + 600, 3000, "pcb", rate=0.03, supplier_from=1500)
        imgs, _ = W.render_plan("pcb", seed + 600, plan)
        y = np.array([p["defect"] is not None for p in plan])
        s1 = m1.scores(imgs)
        ri = np.nonzero(s1 > m1.threshold)[0]
        feats = m1.type_features(imgs[ri])
        tplan = W.stream_plan(seed + 700, 3000, "pcb", rate=0.06, supplier_from=0)
        timgs, _ = W.render_plan("pcb", seed + 700, tplan)
        ty = np.array([p["defect"] is not None for p in tplan])
        tb = np.array([p["supplier_b"] for p in tplan])

        def row(label, mm, n_good):
            rj = mm.scores(timgs) > mm.threshold
            f = lambda msk: f"{int((~rj[msk & ty]).sum())}/{int((msk & ty).sum())}"     # noqa: E731
            out.append(f"| {seed} | {label} | {n_good} | {pct(rj[tb & ~ty].mean())} | {pct(rj[~tb & ~ty].mean())} | {f(~tb)} | {f(tb)} |")
        row("nobody (version 1)", m1, 0)
        for label, pick in (("cluster representatives (the product)", E.select_for_review(feats, s1[ri], m1.threshold, 40)),
                            ("nearest the threshold", E.select_for_review(feats, s1[ri], m1.threshold, 40, strategy="near_threshold")),
                            ("at random", list(np.random.default_rng(seed).choice(len(ri), 40, replace=False))),
                            ("highest score", list(np.argsort(-s1[ri])[:40]))):
            sel = ri[pick]
            goods = sel[~y[sel]]
            m2 = E.Model().fit(ts["good"], ts["defects"], ts["defect_types"], ts["calib"], seed=1, extra_good=imgs[goods] if len(goods) else None)
            row(label, m2, len(goods))
    out += ["", f"Run time {time.time() - t0:.0f} s."]
    print("\n".join(out))


if __name__ == "__main__":
    main()
