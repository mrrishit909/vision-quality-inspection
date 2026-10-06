"""Models. The anomaly model is PatchCore in miniature: every 8 x 8 patch of a normalised image becomes a 24-number vector
(PCA), the vectors from good parts are kept per position (and from a pixel or two either side, so a part slightly off centre is
still normal), thinned to a coreset by greedy k-centre, and a new image scores each patch by its distance to the nearest
normal one. The image score is the worst patch. A small classifier names the defect type from where and how the patches
are wrong, trained on the few labelled defects. The baseline is a golden template: the mean good image, and the largest
smoothed difference from it.
"""
import io
import math

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from . import world as W

P, STRIDE, DIMS, BANK = 8, 4, 24, 320
EXTRA_BANK = 128
RESID_W = 2.0                                 # weight of the off-subspace residual in the distance
GRID = (W.SIZE - P) // STRIDE + 1              # 15 x 15 patch positions
SHIFTS = [(dy, dx) for dy in (-2, 0, 2) for dx in (-2, 0, 2)]


def normalise(img):
    """Per-image standardisation: removes the lamp's gain and offset."""
    m, s = img.mean(axis=(-2, -1), keepdims=True), img.std(axis=(-2, -1), keepdims=True)
    return (img - m) / np.maximum(s, 1e-6)


def _windows(imgs):
    return sliding_window_view(imgs, (P, P), axis=(-2, -1))       # [..., 57, 57, 8, 8]


def patches(imgs, shift=(0, 0)):
    """[n, GRID*GRID, 64] patch vectors at the grid, displaced by `shift` pixels (clipped at the border)."""
    w = _windows(normalise(imgs))
    ys = np.clip(np.arange(GRID) * STRIDE + shift[0], 0, W.SIZE - P)
    xs = np.clip(np.arange(GRID) * STRIDE + shift[1], 0, W.SIZE - P)
    return w[:, ys][:, :, xs].reshape(len(imgs), GRID * GRID, P * P)


def coreset(x, k, rng):
    """Greedy k-centre per position. x: [positions, n, d] -> [positions, k, d]."""
    npos, n, _ = x.shape
    if n <= k:
        return x
    chosen = np.zeros((npos, k), dtype=int)
    chosen[:, 0] = rng.integers(0, n, npos)
    d = np.full((npos, n), np.inf, dtype=np.float32)
    ar = np.arange(npos)
    for i in range(1, k):
        last = x[ar, chosen[:, i - 1]][:, None, :]
        d = np.minimum(d, ((x - last) ** 2).sum(-1))
        chosen[:, i] = d.argmax(1)
    return x[ar[:, None], chosen]


class Model:
    """The deployed artifact: PCA, memory bank, template baseline, type classifier, calibrated threshold."""

    def fit(self, good, defects=(), types=(), calib=None, target_fr=0.01, seed=0, extra_good=None):
        rng = np.random.default_rng(seed)
        sample = patches(good[rng.choice(len(good), min(60, len(good)), replace=False)]).reshape(-1, P * P)
        self.pca = PCA(DIMS, random_state=seed).fit(sample)
        bank = lambda imgs, k: coreset(np.ascontiguousarray(np.concatenate([self._proj(patches(imgs, s)) for s in SHIFTS], axis=0).transpose(1, 0, 2)), k, rng)   # noqa: E731
        self.bank = bank(good, BANK)
        self.bank = self.bank.astype(np.float32)
        self.bank_sq = (self.bank ** 2).sum(-1)
        self.adapted_positions = 0
        calib = good[-40:] if calib is None else calib
        if extra_good is not None and len(extra_good):
            self._thr_hint = float(np.quantile(self.scores(calib), 1 - target_fr))
            self.adapt(extra_good, rng)
        self.template = normalise(good).mean(0)
        self.calib_scores = self.scores(calib)
        self.threshold = float(np.quantile(self.calib_scores, 1 - target_fr))
        self.base_calib = self.baseline_scores(calib)
        self.base_threshold = float(np.quantile(self.base_calib, 1 - target_fr))
        self.target_fr = target_fr
        self.n_good, self.n_extra = len(good), 0 if extra_good is None else len(extra_good)
        self.types = sorted(set(types))
        self.clf = None
        if len(defects):
            X = self.type_features(defects)
            self.clf = RandomForestClassifier(200, min_samples_leaf=1, random_state=seed).fit(X, list(types))
            ds = self.scores(defects)
            z = np.r_[np.log(self.calib_scores), np.log(ds)]
            y = np.r_[np.zeros(len(self.calib_scores)), np.ones(len(ds))]
            self.platt = LogisticRegression(C=10).fit(z[:, None], y)
            self.labelled_recall = float((ds > self.threshold).mean())
            self.n_defects = len(ds)
        return self

    def adapt(self, extra_good, rng):
        """Few-shot adaptation from operator-confirmed good parts. Only the positions where those parts looked anomalous take
        their patches: adding them everywhere makes every nearest neighbour closer and hides faint defects all over the part."""
        m = self.maps(extra_good)                                                       # [n, GRID, GRID] under the current bank
        hot = m.reshape(len(extra_good), -1) > 0.8 * self._thr_hint
        feats = np.concatenate([self._proj(patches(extra_good, s)) for s in SHIFTS], axis=0).transpose(1, 0, 2)      # [pos, n*9, d]
        hot9 = np.tile(hot.T, (1, len(SHIFTS)))                                          # [pos, n*9]
        extra = np.repeat(self.bank[:, :1], EXTRA_BANK, axis=1).copy()                   # padding: a duplicate never changes a nearest neighbour
        for p in np.nonzero(hot9.any(1))[0]:
            cand = feats[p][hot9[p]][None]
            extra[p, :min(EXTRA_BANK, cand.shape[1])] = coreset(cand, EXTRA_BANK, rng)[0][:EXTRA_BANK]
        self.adapted_positions = int(hot9.any(1).sum())
        self.bank = np.concatenate([self.bank, extra.astype(np.float32)], axis=1)
        self.bank_sq = (self.bank ** 2).sum(-1)

    def _proj(self, pt):
        """PCA coordinates plus the distance from the PCA subspace: what PCA cannot express (a thin scratch) is kept as one number."""
        n, npos, _ = pt.shape
        x = pt.reshape(-1, P * P)
        z = self.pca.transform(x)
        resid = np.linalg.norm(x - self.pca.inverse_transform(z), axis=1, keepdims=True)
        return np.hstack([z, RESID_W * resid]).reshape(n, npos, DIMS + 1).astype(np.float32)

    def maps(self, imgs):
        """Anomaly map per image: distance from each patch to its nearest normal patch at that position. [n, GRID, GRID]"""
        f = self._proj(patches(imgs)).transpose(1, 0, 2)                                  # [pos, n, d+1]
        d2 = (f ** 2).sum(-1)[:, :, None] + self.bank_sq[:, None, :] - 2 * np.einsum("pnd,pkd->pnk", f, self.bank)
        return np.sqrt(np.maximum(d2.min(-1), 0)).T.reshape(len(imgs), GRID, GRID)

    def scores(self, imgs, chunk=256):
        return np.concatenate([self.maps(imgs[i:i + chunk]).max(axis=(1, 2)) for i in range(0, len(imgs), chunk)])

    def baseline_scores(self, imgs):
        """Golden template with a pixel of tolerance: each pixel's difference is the smallest over the template shifted by up to 1 px."""
        x = normalise(imgs)
        diff = np.min([np.abs(x - np.roll(self.template, (dy, dx), (0, 1))) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)
        return ndimage.uniform_filter(diff, size=(1, 3, 3))[:, 2:-2, 2:-2].max(axis=(1, 2))

    def type_features(self, imgs, maps=None):
        maps = self.maps(imgs) if maps is None else maps
        res = normalise(imgs) - self.template
        out = []
        for m, rr in zip(maps, res):
            i, j = np.unravel_index(m.argmax(), m.shape)
            hot = m > 0.6 * m.max()
            ys, xs = np.nonzero(hot)
            cov = np.cov(np.vstack([ys, xs])) if len(ys) > 2 else np.eye(2) * 0.1
            ev = np.sort(np.linalg.eigvalsh(cov + np.eye(2) * 1e-3))
            y0, x0 = i * STRIDE, j * STRIDE
            win = rr[y0:y0 + P, x0:x0 + P]
            out.append([m.max(), hot.sum(), ev[1] / ev[0], i, j, win.mean(), win.min(), win.max(), (np.abs(win) > 1).mean()])
        return np.array(out)

    def confidence(self, s):
        """P(defect | score) with the line's 3% defect rate as the prior (the labelled set is not 3% defects)."""
        p = self.platt.predict_proba(np.log(np.maximum(np.atleast_1d(s), 1e-9))[:, None])[:, 1]
        lab_prior = self.n_defects / (self.n_defects + len(self.calib_scores))
        odds = p / (1 - p + 1e-12) * (0.03 / 0.97) / (lab_prior / (1 - lab_prior))
        return odds / (1 + odds)

    def inspect(self, img):
        """One frame on the edge: score, decision, defect type, box, confidence."""
        m = self.maps(img[None])[0]
        s = float(m.max())
        out = {"score": s, "reject": s > self.threshold}
        if out["reject"]:
            hot = np.argwhere(m > max(self.threshold, 0.7 * s))
            (i0, j0), (i1, j1) = hot.min(0), hot.max(0)
            out["bbox"] = [int(j0 * STRIDE), int(i0 * STRIDE), int((j1 - j0) * STRIDE + P), int((i1 - i0) * STRIDE + P)]     # x, y, w, h
            out["type"] = str(self.clf.predict(self.type_features(img[None], m[None]))[0]) if self.clf is not None else "anomaly"
            out["confidence"] = float(self.confidence(s)[0]) if self.clf is not None else None
        out["map"] = m
        return out

    def dumps(self):
        buf = io.BytesIO()
        import pickle
        pickle.dump(self, buf, protocol=5)
        return buf.getvalue()


def loads(b):
    import pickle
    return pickle.loads(b)


# ---------------------------------------------------------------- metrics

def wilson(k, n, z=1.645):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, c - h), min(1.0, c + h))


def rates(reject, is_defect):
    reject, is_defect = np.asarray(reject, bool), np.asarray(is_defect, bool)
    good, bad = ~is_defect, is_defect
    fr, esc = int((reject & good).sum()), int((~reject & bad).sum())
    return {"frames": int(len(reject)), "rejected": int(reject.sum()), "good": int(good.sum()), "defective": int(bad.sum()),
            "false_rejects": fr, "escapes": esc, "false_reject_rate": fr / max(1, good.sum()), "escape_rate": esc / max(1, bad.sum()),
            "false_reject_90": wilson(fr, int(good.sum())), "escape_90": wilson(esc, int(bad.sum())), "yield": float((~reject).mean())}


def auroc(neg, pos):
    s = np.r_[neg, pos]
    rk = s.argsort().argsort() + 1
    return float((rk[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def select_for_review(features, scores, threshold, budget, seed=0, k=8, strategy="representative"):
    """Active learning: cluster the rejected frames by what is wrong with them, then take from each cluster (largest first)
    its most typical frames (or, with strategy='near_threshold', those nearest the threshold), round-robin, until the budget
    is spent. -> indices into the arrays given."""
    n = len(scores)
    if n <= budget:
        return list(range(n))
    X = (features - features.mean(0)) / (features.std(0) + 1e-9)
    km = KMeans(min(k, n), n_init=4, random_state=seed).fit(X)
    lab, dist = km.labels_, ((X - km.cluster_centers_[km.labels_]) ** 2).sum(1)
    if strategy == "near_threshold":
        key = lambda i: scores[i] - threshold          # noqa: E731
    else:
        key = lambda i: dist[i]                         # noqa: E731
    sizes = np.bincount(lab)
    queues = [sorted(np.nonzero(lab == c)[0].tolist(), key=key) for c in np.argsort(-sizes)]
    out = []
    while len(out) < budget:
        for q in queues:
            if q and len(out) < budget:
                out.append(q.pop(0))
    return out
