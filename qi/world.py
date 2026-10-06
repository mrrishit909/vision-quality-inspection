"""The simulated production lines. Every frame is rendered from a seed: a part (circuit board, sealed pouch or machined bracket),
the nuisance a real camera sees (lighting gain and offset, a brightness gradient, sensor noise, the part a pixel or two off
centre), and sometimes a defect with its true pixel mask. A benign supplier change (a different capacitor body on the board)
can be switched on to see what a model does with a good part it has never seen.

Images are 64 x 64, grayscale, floats in [0, 1].
"""
import functools

import numpy as np
from scipy import ndimage

SIZE = 64
DEFECTS = {"pcb": ["missing_component", "solder_bridge", "scratch"],
           "pouch": ["seal_gap", "contamination", "wrinkle"],
           "bracket": ["scratch", "dent", "missing_hole"]}
PRODUCTS = {"pcb": "Controller board CB-7 (electronics)", "pouch": "Sterile pouch SP-2 (packaging)", "bracket": "Mounting bracket MB-40 (metal)"}
COMPONENTS = [(14, 12, 10, 6), (14, 38, 10, 6), (40, 12, 6, 12), (40, 40, 12, 8)]       # row, col, height, width of each part on the board
HOLES = [(16, 16, 4.5), (16, 48, 4.5), (46, 32, 5.5)]


def _texture(r, sigma, amp):
    return ndimage.gaussian_filter(r.normal(0, 1, (SIZE, SIZE)), sigma) * amp / max(1e-9, 0.6 / sigma)


def _pcb(r, supplier_b=False):
    img = 0.36 + _texture(r, 1.0, 0.03)
    for row in (6, 30, 56):
        img[row:row + 2, 4:60] = 0.55                         # traces
    for col in (30,):
        img[4:60, col:col + 2] = 0.55
    body, extra = (0.28, 1) if supplier_b else (0.17, 0)
    for k, (y, x, h, w) in enumerate(COMPONENTS):
        if k in (0, 1):                                       # the two capacitors are the parts the second supplier changes
            img[y:y + h, x:x + w + extra] = body
        else:
            img[y:y + h, x:x + w] = 0.17
        img[y + h // 2 - 1:y + h // 2 + 1, x - 2:x] = 0.85    # pads either side
        img[y + h // 2 - 1:y + h // 2 + 1, x + w + extra:x + w + extra + 2] = 0.85
    return img


def _pouch(r, **_):
    img = 0.6 + _texture(r, 2.5, 0.05) + 0.02 * np.sin(np.linspace(0, r.uniform(4, 9), SIZE))[None, :]
    seal = 0.8 + 0.04 * ((np.add.outer(np.arange(SIZE), np.arange(SIZE)) % 4) < 2)
    img[5:12] = seal[5:12]
    img[22:42, 18:46] = 0.42                                  # printed label
    img[26:28, 22:42] = 0.75
    img[32:34, 22:38] = 0.75
    return img


def _bracket(r, **_):
    streak = ndimage.gaussian_filter1d(r.normal(0, 1, (SIZE, SIZE)), 6, axis=1) * 0.06
    img = 0.55 + streak + _texture(r, 0.8, 0.01)
    yy, xx = np.mgrid[:SIZE, :SIZE]
    for y, x, rad in HOLES:
        img[(yy - y) ** 2 + (xx - x) ** 2 <= rad ** 2] = 0.08
    img[:, :3] = img[:, -3:] = 0.3                            # edges
    return img


RENDER = {"pcb": _pcb, "pouch": _pouch, "bracket": _bracket}


def _line_mask(r, length, angle, cy, cx):
    m = np.zeros((SIZE, SIZE), bool)
    t = np.linspace(-length / 2, length / 2, int(length * 2))
    ys, xs = np.round(cy + t * np.sin(angle)).astype(int), np.round(cx + t * np.cos(angle)).astype(int)
    ok = (ys >= 0) & (ys < SIZE) & (xs >= 0) & (xs < SIZE)
    m[ys[ok], xs[ok]] = True
    return m


def _blob(cy, cx, rad):
    yy, xx = np.mgrid[:SIZE, :SIZE]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= rad ** 2


def _defect(product, kind, img, r):
    """Apply a defect in place; return its true mask. Contrast is drawn wide, so some defects are faint."""
    if kind == "missing_component":
        y, x, h, w = COMPONENTS[r.integers(4)]
        m = np.zeros_like(img, bool)
        m[y:y + h, x:x + w] = True
        img[m] = 0.36 + r.normal(0, 0.01, m.sum())
    elif kind == "solder_bridge":
        y, x, h, w = COMPONENTS[r.integers(4)]
        m = _blob(y + h // 2, x + w + 3, r.uniform(1.2, 2.6))
        img[m] += r.uniform(0.15, 0.45)
    elif kind == "scratch":
        m = _line_mask(r, r.uniform(10, 30), r.uniform(0, np.pi), r.uniform(12, 52), r.uniform(12, 52))
        img[m] += r.uniform(0.05, 0.25) * r.choice([1, -1])
    elif kind == "seal_gap":
        x0, w = r.integers(6, 52), r.integers(2, 8)
        m = np.zeros_like(img, bool)
        m[5:12, x0:x0 + w] = True
        img[m] = 0.6 + r.normal(0, 0.02, m.sum())
    elif kind == "contamination":
        m = _blob(r.uniform(14, 58), r.uniform(6, 58), r.uniform(1.0, 2.5))
        img[m] -= r.uniform(0.12, 0.35)
    elif kind == "wrinkle":
        yy, xx = np.mgrid[:SIZE, :SIZE]
        a, c = r.uniform(-0.6, 0.6), r.uniform(15, 55)
        d = np.abs(yy - (c + a * (xx - 32)))
        m = d < 1.5
        img -= r.uniform(0.04, 0.12) * np.exp(-d ** 2 / 2)
    elif kind == "dent":
        cy, cx, rad = r.uniform(10, 54), r.uniform(10, 54), r.uniform(3, 6)
        yy, xx = np.mgrid[:SIZE, :SIZE]
        shade = r.uniform(0.06, 0.18) * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (rad / 1.5) ** 2)) * np.sign(xx - cx + 0.1)
        img += shade
        m = _blob(cy, cx, rad)
    elif kind == "missing_hole":
        y, x, rad = HOLES[r.integers(3)]
        m = _blob(y, x, rad)
        img[m] = 0.55 + r.normal(0, 0.02, m.sum())
    else:
        raise ValueError(kind)
    return m


def frame(product, seed, idx, defect=None, supplier_b=False, drift=0.0):
    """One camera frame. -> (image, mask or None). drift moves the lighting mean (a lamp ageing)."""
    r = np.random.default_rng([seed, idx])
    img = RENDER[product](r, supplier_b=supplier_b)
    mask = _defect(product, defect, img, r) if defect else None
    dy, dx = r.uniform(-1.5, 1.5, 2)
    img = ndimage.shift(img, (dy, dx), order=1, mode="nearest")
    if mask is not None:
        mask = ndimage.shift(mask.astype(float), (dy, dx), order=1) > 0.3
    gy = np.linspace(-1, 1, SIZE)[:, None] * r.normal(0, 0.03) + np.linspace(-1, 1, SIZE)[None, :] * r.normal(0, 0.03)
    img = img * r.uniform(0.88, 1.12) + r.uniform(-0.05, 0.05) + drift + gy + r.normal(0, 0.012, img.shape)
    return np.clip(img, 0, 1).astype(np.float32), mask


def stream_plan(seed, n, product="pcb", rate=0.03, supplier_from=None, drift_per_frame=0.0):
    """Which frames carry which defect, whether the second supplier's part is on the board, and the lighting drift."""
    r = np.random.default_rng([seed, 999])
    kinds = DEFECTS[product]
    out = []
    for i in range(n):
        d = kinds[r.integers(len(kinds))] if r.random() < rate else None
        sb = supplier_from is not None and i >= supplier_from and r.random() < 0.7
        out.append({"idx": i, "defect": d, "supplier_b": bool(sb), "drift": drift_per_frame * i})
    return out


def render_plan(product, seed, plan):
    imgs, masks = [], []
    for p in plan:
        im, m = frame(product, seed, p["idx"], p["defect"], p["supplier_b"], p["drift"])
        imgs.append(im)
        masks.append(m)
    return np.stack(imgs), masks


@functools.lru_cache(maxsize=16)
def training_set(product="pcb", seed=14, goods=200, per_defect=8, calib=150):
    """What a new line has on day one: good parts from the first shift, a handful of labelled defects per type, and a
    separate set of good parts to set the threshold on."""
    plan = [{"idx": i, "defect": None, "supplier_b": False, "drift": 0.0} for i in range(goods)]
    good, _ = render_plan(product, seed + 1, plan)
    cplan = [{"idx": i, "defect": None, "supplier_b": False, "drift": 0.0} for i in range(calib)]
    cal, _ = render_plan(product, seed + 2, cplan)
    dplan = [{"idx": i, "defect": k, "supplier_b": False, "drift": 0.0} for i, k in enumerate([k for k in DEFECTS[product] for _ in range(per_defect)])]
    bad, bmask = render_plan(product, seed + 3, dplan)
    return {"good": good, "calib": cal, "defects": bad, "defect_types": [p["defect"] for p in dplan], "defect_masks": bmask}
