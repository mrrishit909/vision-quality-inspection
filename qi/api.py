"""Public API (modular monolith). Blueprint services map to: camera gateway and inspection runtime (the edge simulator: a
30 FPS stream through the deployed model, as a background job), defect model (anomaly memory bank + type classifier), labeling
service (annotations, operator feedback), active learning (the review queue), model registry (versions with hashes and
metrics), edge deployer (full and canary deployments, promotion behind a gate) and quality analytics.

    uvicorn qi.api:app          python -m core.jobs qi.api      # the worker
"""
import base64
import hashlib
import io
import time
import uuid

import numpy as np
from fastapi import Depends, Header
from fastapi.encoders import jsonable_encoder
from PIL import Image
from psycopg.types.json import Jsonb
from pydantic import BaseModel, Field

from core import audit, db, jobs
from core.app import Ctx, Problem, create_app, run

from . import engine as E, world as W

READ = {"line:read", "jobs:read"}
OPERATE = READ | {"feedback:write"}
ENGINEER = OPERATE | {"models:train", "models:deploy", "inspect:run"}
PERMISSIONS = {"viewer": READ, "operator": OPERATE, "quality_engineer": ENGINEER, "quality_manager": ENGINEER | {"line:load", "models:promote", "audit:read"}}
app = create_app("vision-quality-inspection", PERMISSIONS)
auth = app.state.auth
IdemKey = Header(None, alias="Idempotency-Key")
AUDIT_SHARE = 0.05                    # a quality auditor pulls 5% of passed parts for a manual check
FPS_BUDGET_MS = 1000 / 30
_models = {}                          # the edge's loaded artifacts, by model_version id


def worker_ctx(job):
    return Ctx(job["tenant_id"], uuid.UUID(job["payload"]["actor_id"]), "worker", "system")


def to_png(img):
    buf = io.BytesIO()
    Image.fromarray(np.round(img * 255).astype(np.uint8)).save(buf, "PNG", optimize=True)
    return buf.getvalue()


def from_png(b):
    return np.asarray(Image.open(io.BytesIO(bytes(b))), dtype=np.float32) / 255


def quantise(img):
    """The camera delivers 8-bit pixels."""
    return np.round(img * 255).astype(np.uint8).astype(np.float32) / 255


def data_uri(png):
    return "data:image/png;base64," + base64.b64encode(bytes(png)).decode()


def line(c):
    ln = c.execute("SELECT l.*, p.code AS product_code, cam.id AS camera_id FROM line l JOIN product p ON p.id = l.product_id JOIN camera cam ON cam.line_id = l.id LIMIT 1").fetchone()
    if not ln:
        raise Problem(409, "no_line", "load the line first: POST /v1/lines:load")
    return ln


def model(c, mv_id):
    """Load a model version on the edge, checking the artifact's hash first."""
    if mv_id not in _models:
        r = c.execute("SELECT artifact, sha256 FROM model_version WHERE id = %s", [mv_id]).fetchone()
        if hashlib.sha256(bytes(r["artifact"])).hexdigest() != r["sha256"]:
            raise Problem(409, "artifact_tampered", "the model's artifact does not match its registered hash")
        _models[mv_id] = E.loads(r["artifact"])
    return _models[mv_id]


# ---------------------------------------------------------------- line and training set

class LoadIn(BaseModel):
    seed: int = Field(14, ge=0, le=10 ** 6)
    product: str = Field("pcb", pattern="^(pcb|pouch|bracket)$")


@app.post("/v1/lines:load", status_code=202, tags=["line"], summary="(+) Set up the simulated line (factory, product, camera) with its day-one training set: good parts from the first shift, a few labelled defects per type, and good parts held back to set the threshold")
def load_line(body: LoadIn, ctx: Ctx = Depends(auth("line:load")), idem: str | None = IdemKey):
    def work(c):
        if c.execute("SELECT 1 FROM line").fetchone():
            raise Problem(409, "line_exists")
        return 202, jobs.enqueue(c, ctx, "line.load", body.model_dump())
    return run(ctx, idem, body, work)


@jobs.handler("line.load")
def load_job(c, job):
    t, p = job["tenant_id"], job["payload"]
    fid, pid, lid, cid = (uuid.uuid5(t, k) for k in ("factory", "product", "line", "camera"))
    c.execute("INSERT INTO factory VALUES (%s,%s,'Plant 3','Riverside industrial park')", [fid, t])
    c.execute("INSERT INTO product VALUES (%s,%s,%s,%s,%s)", [pid, t, p["product"].upper(), W.PRODUCTS[p["product"]], p["product"]])
    c.execute("INSERT INTO line (id, tenant_id, factory_id, product_id, name, seed) VALUES (%s,%s,%s,%s,'Line 2 end-of-line inspection',%s)", [lid, t, fid, pid, p["seed"]])
    c.execute("INSERT INTO camera VALUES (%s,%s,%s,'Cam 2A (top, diffuse light)',30,%s,%s)", [cid, t, lid, W.SIZE, W.SIZE])
    ts = W.training_set(p["product"], p["seed"])
    imgs, anns = [], []
    groups = [("train", ts["good"], [None] * len(ts["good"])), ("calibration", ts["calib"], [None] * len(ts["calib"])), ("train", ts["defects"], ts["defect_types"])]
    n = 0
    for split, arr, types in groups:
        for im, ty in zip(arr, types):
            png = to_png(im)
            iid = uuid.uuid7()
            n -= 1
            imgs.append((iid, t, cid, n, f"synthetic://{p['product']}/{p['seed']}/setup/{-n}", hashlib.sha256(png).hexdigest(), png, Jsonb({"defect": ty})))
            anns.append((uuid.uuid7(), t, iid, "defect" if ty else "good", ty, split, "day-one labelling", uuid.UUID(job["payload"]["actor_id"])))
    db.load(c, "image", ["id", "tenant_id", "camera_id", "frame_no", "uri", "sha256", "png", "sim_truth"], imgs)
    db.load(c, "annotation", ["id", "tenant_id", "image_id", "label", "defect_type", "split", "source", "created_by"], anns)
    audit.record(c, worker_ctx(job), "line.loaded", "line", lid, {"product": p["product"], "images": len(imgs)})
    ex = [data_uri(to_png(ts["good"][0]))] + [data_uri(to_png(ts["defects"][k])) for k in range(0, len(ts["defects"]), len(ts["defects"]) // len(W.DEFECTS[p["product"]]))]
    return {"line_id": lid, "product": W.PRODUCTS[p["product"]], "camera": {"fps": 30, "pixels": f"{W.SIZE} x {W.SIZE}"},
            "training_set": {"good": len(ts["good"]), "calibration_good": len(ts["calib"]), "labelled_defects": len(ts["defects"]), "defect_types": W.DEFECTS[p["product"]]},
            "examples": [{"label": lab, "png": u} for lab, u in zip(["good"] + W.DEFECTS[p["product"]], ex)]}


# ---------------------------------------------------------------- models

class TrainIn(BaseModel):
    target_false_reject: float = Field(0.01, ge=0.001, le=0.1)
    include_feedback: bool = True


@app.post("/v1/models/train", status_code=202, tags=["models"], summary="Train a model version from the annotations: the anomaly memory bank from good parts (and operator-confirmed good parts, where they looked wrong), the type classifier from labelled defects, the threshold from held-back good parts")
def train(body: TrainIn, ctx: Ctx = Depends(auth("models:train")), idem: str | None = IdemKey):
    def work(c):
        line(c)
        return 202, jobs.enqueue(c, ctx, "model.train", body.model_dump())
    return run(ctx, idem, body, work)


def annotated(c, split, label):
    rows = c.execute("SELECT i.png, a.defect_type FROM annotation a JOIN image i ON i.id = a.image_id WHERE a.split = %s AND a.label = %s ORDER BY i.frame_no", [split, label]).fetchall()
    return (np.stack([from_png(r["png"]) for r in rows]) if rows else np.zeros((0, W.SIZE, W.SIZE), np.float32)), [r["defect_type"] for r in rows]


@jobs.handler("model.train")
def train_job(c, job):
    t, p = job["tenant_id"], job["payload"]
    ln = line(c)
    good, _ = annotated(c, "train", "good")
    calib, _ = annotated(c, "calibration", "good")
    bad, types = annotated(c, "train", "defect")
    fb_good, _ = annotated(c, "feedback", "good") if p["include_feedback"] else (np.zeros((0, W.SIZE, W.SIZE), np.float32), [])
    fb_bad, fb_types = annotated(c, "feedback", "defect") if p["include_feedback"] else (np.zeros((0, W.SIZE, W.SIZE), np.float32), [])
    if len(fb_bad):
        bad, types = np.concatenate([bad, fb_bad]), types + fb_types
    t0 = time.perf_counter()
    m = E.Model().fit(good, bad, types, calib, target_fr=p["target_false_reject"], seed=1, extra_good=fb_good if len(fb_good) else None)
    secs = time.perf_counter() - t0
    from sklearn.model_selection import cross_val_score
    cv = float(cross_val_score(type(m.clf)(200, random_state=1), m.type_features(bad), types, cv=4).mean())
    ds, bs = m.scores(bad), m.baseline_scores(bad)
    metrics = {"threshold": m.threshold, "target_false_reject": p["target_false_reject"], "calibration_false_reject": float((m.calib_scores > m.threshold).mean()),
               "labelled_defects_caught": int((ds > m.threshold).sum()), "labelled_defects": len(ds), "labelled_recall_90": E.wilson(int((ds > m.threshold).sum()), len(ds)),
               "baseline_labelled_caught": int((bs > m.base_threshold).sum()), "type_accuracy_cv": cv, "adapted_positions": m.adapted_positions, "train_seconds": round(secs, 1),
               "by_type_caught": {k: [int((ds[np.array(types) == k] > m.threshold).sum()), int((np.array(types) == k).sum())] for k in sorted(set(types))}}
    art = m.dumps()
    ver = (c.execute("SELECT coalesce(max(version), 0) AS v FROM model_version").fetchone()["v"]) + 1
    mid = uuid.uuid7()
    c.execute("INSERT INTO model_version (id, tenant_id, line_id, version, artifact, sha256, metrics, training, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
              [mid, t, ln["id"], ver, art, hashlib.sha256(art).hexdigest(), Jsonb(jsonable_encoder(metrics)),
               Jsonb({"good": len(good), "calibration_good": len(calib), "labelled_defects": len(bad), "feedback_good": len(fb_good), "feedback_defects": len(fb_bad), "patch": E.P, "stride": E.STRIDE, "pca_dims": E.DIMS, "bank_per_position": int(m.bank.shape[1])}),
               uuid.UUID(p["actor_id"])])
    audit.record(c, worker_ctx(job), "model.trained", "model_version", mid, {"version": ver, "feedback_good": len(fb_good), "threshold": round(m.threshold, 3)})
    return jsonable_encoder({"model_version_id": mid, "version": ver, "sha256": hashlib.sha256(art).hexdigest()[:16], "artifact_kb": round(len(art) / 1024), "training": {"good": len(good), "labelled_defects": len(bad), "feedback_good": len(fb_good), "feedback_defects": len(fb_bad)}, **metrics})


class DeployIn(BaseModel):
    model_version_id: uuid.UUID
    mode: str = Field("full", pattern="^(full|canary)$")
    share: float = Field(1.0, gt=0, le=1)


@app.post("/v1/deployments", status_code=201, tags=["models"], summary="Deploy a model version to the line's edge: full, or as a canary on a share of frames beside the current full deployment")
def deploy(body: DeployIn, ctx: Ctx = Depends(auth("models:deploy")), idem: str | None = IdemKey):
    def work(c):
        ln = line(c)
        mv = c.execute("SELECT id, version FROM model_version WHERE id = %s", [body.model_version_id]).fetchone()
        if not mv:
            raise Problem(404, "model_version_not_found")
        model(c, mv["id"])                                     # hash checked before anything goes live
        share = 1.0 if body.mode == "full" else body.share
        if body.mode == "canary":
            if not c.execute("SELECT 1 FROM deployment WHERE status = 'active' AND mode = 'full'").fetchone():
                raise Problem(409, "no_control", "a canary needs a full deployment to compare against")
            if share > 0.5:
                raise Problem(422, "canary_too_large", "a canary takes at most half the frames")
        c.execute("UPDATE deployment SET status = 'retired' WHERE status = 'active' AND mode = %s", [body.mode])
        did = uuid.uuid7()
        c.execute("INSERT INTO deployment (id, tenant_id, line_id, model_version_id, mode, share, status, created_by) VALUES (%s,%s,%s,%s,%s,%s,'active',%s)", [did, ctx.tenant_id, ln["id"], mv["id"], body.mode, share, ctx.actor_id])
        audit.record(c, ctx, "model.deployed", "deployment", did, {"version": mv["version"], "mode": body.mode, "share": share})
        return 201, {"deployment_id": did, "version": mv["version"], "mode": body.mode, "share": share, "status": "active"}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- the edge: a stream through the deployed models

class StreamIn(BaseModel):
    frames: int = Field(3000, ge=30, le=20000)
    defect_rate: float = Field(0.03, ge=0, le=0.5)
    supplier_change_at: int | None = Field(None, ge=0, description="frame from which 70% of boards carry the second supplier's capacitors")
    stream_seed: int = Field(101, ge=0, le=10 ** 6)


@app.post("/v1/inspections", status_code=202, tags=["inspection"], summary="Run a stretch of the line through the edge simulator at 30 FPS: every frame inspected by the deployed model (or the canary), each frame's latency measured; 5% of passed parts audited by hand")
def inspect_stream(body: StreamIn, ctx: Ctx = Depends(auth("inspect:run")), idem: str | None = IdemKey):
    def work(c):
        line(c)
        if not c.execute("SELECT 1 FROM deployment WHERE status = 'active' AND mode = 'full'").fetchone():
            raise Problem(409, "nothing_deployed", "deploy a model first: POST /v1/deployments")
        rid = uuid.uuid7()
        c.execute("INSERT INTO inspection_run (id, tenant_id, line_id, frames, fps, params, created_by) VALUES (%s,%s,(SELECT id FROM line LIMIT 1),%s,30,%s,%s)", [rid, ctx.tenant_id, body.frames, Jsonb(body.model_dump()), ctx.actor_id])
        return 202, {**jobs.enqueue(c, ctx, "inspection.stream", {"run_id": str(rid), **body.model_dump()}), "run_id": rid}
    return run(ctx, idem, body, work)


@jobs.handler("inspection.stream")
def stream_job(c, job):
    t, p = job["tenant_id"], job["payload"]
    rid = uuid.UUID(p["run_id"])
    ln = line(c)
    deps = {r["mode"]: r for r in c.execute("SELECT d.*, mv.version FROM deployment d JOIN model_version mv ON mv.id = d.model_version_id WHERE d.status = 'active'")}
    full, canary = deps["full"], deps.get("canary")
    models = {"control": model(c, full["model_version_id"])}
    if canary:
        models["canary"] = model(c, canary["model_version_id"])
    plan = W.stream_plan(p["stream_seed"], p["frames"], ln["product_code"].lower(), p["defect_rate"], p["supplier_change_at"])
    r = np.random.default_rng([p["stream_seed"], 5])
    start = ln["frames_seen"]
    images, insp, defects, lat, rows = [], [], [], [], []
    for k, fp in enumerate(plan):
        img, _ = W.frame(ln["product_code"].lower(), p["stream_seed"], fp["idx"], fp["defect"], fp["supplier_b"], fp["drift"])
        img = quantise(img)
        arm = "canary" if canary and r.random() < float(canary["share"]) else "control"
        t0 = time.perf_counter()
        out = models[arm].inspect(img)
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)
        reject = bool(out["reject"])
        audited = (not reject) and r.random() < AUDIT_SHARE
        iid, sid = uuid.uuid7(), uuid.uuid7()
        png = to_png(img) if (reject or audited) else None
        raw = np.round(img * 255).astype(np.uint8).tobytes()
        images.append((iid, t, ln["camera_id"], start + k, f"synthetic://{ln['product_code'].lower()}/{p['stream_seed']}/{fp['idx']}", hashlib.sha256(raw).hexdigest(), png, Jsonb({"defect": fp["defect"], "supplier_b": fp["supplier_b"]})))
        dep = canary if arm == "canary" else full
        insp.append((sid, t, iid, rid, dep["model_version_id"], dep["id"], arm, out["score"], "reject" if reject else "pass", ms, audited, (fp["defect"] is not None) if audited else None,
                     np.round(out["map"].ravel(), 2).tolist() if reject else None))
        if reject:
            defects.append((uuid.uuid7(), t, sid, out["type"], out["bbox"], out["confidence"]))
        rows.append({"arm": arm, "reject": reject, "defect": fp["defect"], "supplier_b": fp["supplier_b"], "audited": audited})
    db.load(c, "image", ["id", "tenant_id", "camera_id", "frame_no", "uri", "sha256", "png", "sim_truth"], images)
    db.load(c, "inspection", ["id", "tenant_id", "image_id", "run_id", "model_version_id", "deployment_id", "arm", "score", "decision", "latency_ms", "audited", "audit_found_defect", "anomaly_map"], insp)
    db.load(c, "defect", ["id", "tenant_id", "inspection_id", "defect_type", "bbox", "confidence"], defects)
    c.execute("UPDATE line SET frames_seen = frames_seen + %s WHERE id = %s", [len(plan), ln["id"]])
    summary = summarise(rows, np.array(lat), {"control": full["version"], **({"canary": canary["version"]} if canary else {})})
    c.execute("UPDATE inspection_run SET summary = %s WHERE id = %s", [Jsonb(jsonable_encoder(summary)), rid])
    audit.record(c, worker_ctx(job), "inspection.run", "inspection_run", rid, {"frames": len(plan), "rejected": summary["rejected"], "p99_ms": summary["latency_ms"]["p99"]})
    return jsonable_encoder({"run_id": rid, **summary})


def summarise(rows, lat, versions):
    """Observed metrics (what the line can see) and simulation truth (what only the generator knows)."""
    arr = lambda k: np.array([x[k] for x in rows])     # noqa: E731
    arm, rej, aud, sb = arr("arm"), arr("reject"), arr("audited"), arr("supplier_b")
    bad = np.array([x["defect"] is not None for x in rows])
    by_arm = {}
    for a, v in versions.items():
        m = arm == a
        tru = E.rates(rej[m], bad[m])
        n_aud, found = int((aud & m).sum()), int((aud & m & bad).sum())
        by_arm[a] = {"version": v, "frames": int(m.sum()), "reject_rate": float(rej[m].mean()), "audited": n_aud, "audit_found": found, "audit_escape_90": E.wilson(found, n_aud),
                     "simulation_truth": {**tru, "false_reject_rate_new_supplier": float(rej[m & sb & ~bad].mean()) if (m & sb & ~bad).any() else None,
                                          "escapes_new_supplier": [int((~rej[m & sb & bad]).sum()), int((m & sb & bad).sum())], "escapes_by_type": {k: [int((~rej[m & (np.array([x["defect"] for x in rows]) == k)]).sum()), int((m & (np.array([x["defect"] for x in rows]) == k)).sum())] for k in sorted({x["defect"] for x in rows if x["defect"]})}}}
    win = 300                                              # 10 seconds at 30 FPS
    timeline = [{"second": int(i / 30), **{f"{a}_reject_rate": round(float(rej[i:i + win][arm[i:i + win] == a].mean()), 4) if (arm[i:i + win] == a).any() else None for a in versions},
                 "new_supplier_share": round(float(sb[i:i + win].mean()), 2)} for i in range(0, len(rows), win)]
    busy = lat.sum() / 1000
    return {"frames": len(rows), "seconds_of_line": round(len(rows) / 30, 1), "rejected": int(rej.sum()), "yield": float(1 - rej.mean()),
            "latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2), "p95": round(float(np.percentile(lat, 95)), 2), "p99": round(float(np.percentile(lat, 99)), 2), "max": round(float(lat.max()), 2)},
            "budget_ms": round(FPS_BUDGET_MS, 1), "frames_over_budget": int((lat > FPS_BUDGET_MS).sum()), "edge_utilisation_at_30fps": round(busy / (len(rows) / 30), 3), "capacity_fps": round(len(rows) / busy),
            "by_arm": by_arm, "timeline": timeline}


@app.get("/v1/defects", tags=["inspection"], summary="Rejected frames with their image, anomaly map, box, defect type and confidence")
def list_defects(run_id: uuid.UUID, limit: int = 12, ctx: Ctx = Depends(auth("line:read"))):
    with db.tx(ctx.tenant_id) as c:
        rows = c.execute("""SELECT s.id AS inspection_id, i.frame_no, i.png, s.score, s.arm, s.anomaly_map, d.defect_type, d.bbox, d.confidence, mv.version
                              FROM inspection s JOIN image i ON i.id = s.image_id JOIN defect d ON d.inspection_id = s.id JOIN model_version mv ON mv.id = s.model_version_id
                             WHERE s.run_id = %s ORDER BY s.score DESC LIMIT %s""", [run_id, min(limit, 50)]).fetchall()
        counts = c.execute("SELECT d.defect_type, count(*) AS n FROM defect d JOIN inspection s ON s.id = d.inspection_id WHERE s.run_id = %s GROUP BY 1 ORDER BY 2 DESC", [run_id]).fetchall()
    return jsonable_encoder({"run_id": run_id, "pareto": counts, "items": [{**{k: r[k] for k in ("inspection_id", "frame_no", "arm", "defect_type", "bbox", "version")}, "score": round(r["score"], 2),
                                                                            "confidence": round(r["confidence"], 3) if r["confidence"] is not None else None, "png": data_uri(r["png"]), "map": r["anomaly_map"]} for r in rows]})


# ---------------------------------------------------------------- review, feedback, active learning

@app.get("/v1/review-queue", tags=["feedback"], summary="(+) Active learning: which rejected frames an operator should look at, by clustering the rejects and taking each cluster's most typical frames")
def review_queue(run_id: uuid.UUID, budget: int = 40, images: bool = False, ctx: Ctx = Depends(auth("line:read"))):
    with db.tx(ctx.tenant_id) as c:
        items, feats = queue(c, run_id, budget)
    return jsonable_encoder({"run_id": run_id, "budget": budget, "candidates": feats, "items": [{k: v for k, v in it.items() if images or k != "png"} for it in items]})


def queue(c, run_id, budget):
    rows = c.execute("""SELECT s.id AS inspection_id, s.score, s.model_version_id, i.png, i.frame_no, d.defect_type FROM inspection s JOIN image i ON i.id = s.image_id JOIN defect d ON d.inspection_id = s.id
                         LEFT JOIN operator_feedback f ON f.inspection_id = s.id WHERE s.run_id = %s AND f.id IS NULL ORDER BY i.frame_no""", [run_id]).fetchall()
    if not rows:
        return [], 0
    m = model(c, rows[0]["model_version_id"])
    imgs = np.stack([from_png(r["png"]) for r in rows])
    pick = E.select_for_review(m.type_features(imgs), np.array([r["score"] for r in rows]), m.threshold, budget)
    return [{"inspection_id": rows[i]["inspection_id"], "frame_no": rows[i]["frame_no"], "score": round(rows[i]["score"], 2), "model_says": rows[i]["defect_type"], "png": data_uri(rows[i]["png"])} for i in pick], len(rows)


class Verdict(BaseModel):
    inspection_id: uuid.UUID
    verdict: str = Field(pattern="^(false_positive|confirmed_defect)$")
    defect_type: str | None = Field(None, max_length=40)


class FeedbackIn(BaseModel):
    operator: str = Field(min_length=2, max_length=60)
    items: list[Verdict] = Field(min_length=1, max_length=500)


@app.post("/v1/annotations", status_code=201, tags=["feedback"], summary="An operator's verdicts on rejected frames: a false positive becomes a good-part label for the next model, a confirmed defect a defect label")
def annotate(body: FeedbackIn, ctx: Ctx = Depends(auth("feedback:write")), idem: str | None = IdemKey):
    def work(c):
        return 201, record_feedback(c, ctx, body.operator, [v.model_dump() for v in body.items], "operator")
    return run(ctx, idem, body, work)


def record_feedback(c, ctx, operator, items, source):
    done, refused = [], []
    for v in items:
        s = c.execute("SELECT s.id, s.image_id, s.decision, d.defect_type FROM inspection s LEFT JOIN defect d ON d.inspection_id = s.id WHERE s.id = %s", [v["inspection_id"]]).fetchone()
        if not s:
            refused.append({"inspection_id": v["inspection_id"], "why": "unknown inspection"})
            continue
        if s["decision"] != "reject":
            refused.append({"inspection_id": v["inspection_id"], "why": "only rejected frames are reviewed"})
            continue
        if c.execute("SELECT 1 FROM operator_feedback WHERE inspection_id = %s", [s["id"]]).fetchone():
            refused.append({"inspection_id": v["inspection_id"], "why": "already reviewed"})
            continue
        dtype = (v.get("defect_type") or s["defect_type"]) if v["verdict"] == "confirmed_defect" else None
        if dtype and dtype not in sum(W.DEFECTS.values(), []):
            refused.append({"inspection_id": v["inspection_id"], "why": f"unknown defect type '{dtype}'"})
            continue
        c.execute("INSERT INTO operator_feedback (id, tenant_id, inspection_id, verdict, defect_type, operator, created_by) VALUES (%s,%s,%s,%s,%s,%s,%s)", [uuid.uuid7(), ctx.tenant_id, s["id"], v["verdict"], dtype, operator, ctx.actor_id])
        c.execute("INSERT INTO annotation (id, tenant_id, image_id, label, defect_type, split, source, created_by) VALUES (%s,%s,%s,%s,%s,'feedback',%s,%s)",
                  [uuid.uuid7(), ctx.tenant_id, s["image_id"], "good" if v["verdict"] == "false_positive" else "defect", dtype, source, ctx.actor_id])
        done.append({"inspection_id": s["id"], "verdict": v["verdict"]})
    audit.record(c, ctx, "feedback.recorded", "inspection_run", "-", {"operator": operator, "source": source, "recorded": len(done), "refused": len(refused)})
    return {"recorded": len(done), "false_positives": sum(d["verdict"] == "false_positive" for d in done), "confirmed_defects": sum(d["verdict"] == "confirmed_defect" for d in done), "refused": refused}


class SimOperatorIn(BaseModel):
    run_id: uuid.UUID
    budget: int = Field(40, ge=1, le=500)
    error_rate: float = Field(0.02, ge=0, le=0.5)


@app.post("/v1/review-queue:simulate-operator", status_code=201, tags=["feedback"], summary="(+) A stand-in for the person at the review station: labels the queued frames from the simulation truth, wrong on error_rate of them")
def simulate_operator(body: SimOperatorIn, ctx: Ctx = Depends(auth("feedback:write")), idem: str | None = IdemKey):
    def work(c):
        items, n = queue(c, body.run_id, body.budget)
        r = np.random.default_rng(7)
        verdicts, wrong = [], 0
        for it in items:
            truth = c.execute("SELECT i.sim_truth FROM inspection s JOIN image i ON i.id = s.image_id WHERE s.id = %s", [it["inspection_id"]]).fetchone()["sim_truth"]
            is_bad = truth["defect"] is not None
            if r.random() < body.error_rate:
                is_bad, wrong = not is_bad, wrong + 1
            verdicts.append({"inspection_id": it["inspection_id"], "verdict": "confirmed_defect" if is_bad else "false_positive", "defect_type": truth["defect"] if truth["defect"] and is_bad else None})
        out = record_feedback(c, ctx, "simulated operator", verdicts, "simulated operator")
        return 201, {**out, "reviewed_of_rejects": [len(items), n], "operator_mistakes": wrong,
                     "items": [{**{k: v for k, v in it.items()}, "verdict": vv["verdict"]} for it, vv in zip(items[:16], verdicts[:16])]}
    return run(ctx, idem, body, work)


# ---------------------------------------------------------------- promotion and analytics

class PromoteIn(BaseModel):
    run_id: uuid.UUID = Field(description="the inspection run the canary and the control shared")
    note: str = Field(min_length=3, max_length=500)


@app.post("/v1/deployments/{deployment_id}/promote", tags=["models"], summary="(+) Promote a canary to full, behind a gate on what the line can observe: its reject rate must be lower and the audit sample must not show more escapes than the control's")
def promote(deployment_id: uuid.UUID, body: PromoteIn, ctx: Ctx = Depends(auth("models:promote")), idem: str | None = IdemKey):
    def work(c):
        d = c.execute("SELECT d.*, mv.version FROM deployment d JOIN model_version mv ON mv.id = d.model_version_id WHERE d.id = %s", [deployment_id]).fetchone()
        if not d:
            raise Problem(404, "deployment_not_found")
        if d["mode"] != "canary" or d["status"] != "active":
            raise Problem(409, "not_an_active_canary")
        if d["created_by"] == ctx.actor_id:
            raise Problem(409, "four_eyes", "the person who deployed the canary cannot promote it")
        run_ = c.execute("SELECT summary FROM inspection_run WHERE id = %s", [body.run_id]).fetchone()
        if not run_ or "canary" not in run_["summary"].get("by_arm", {}):
            raise Problem(409, "no_canary_evidence", "that run had no canary arm")
        ca, co = run_["summary"]["by_arm"]["canary"], run_["summary"]["by_arm"]["control"]
        gate = {"reject_rate_lower": ca["reject_rate"] < co["reject_rate"],
                "audit_escapes_not_worse": ca["audit_escape_90"][0] <= co["audit_escape_90"][1],
                "canary": {"reject_rate": ca["reject_rate"], "audited": ca["audited"], "audit_found": ca["audit_found"]},
                "control": {"reject_rate": co["reject_rate"], "audited": co["audited"], "audit_found": co["audit_found"]}}
        if not (gate["reject_rate_lower"] and gate["audit_escapes_not_worse"]):
            c.execute("UPDATE deployment SET status = 'rolled_back', decided_by = %s, decision = %s WHERE id = %s", [ctx.actor_id, Jsonb(gate), deployment_id])
            audit.record(c, ctx, "model.rolled_back", "deployment", deployment_id, {"version": d["version"]})
            return 409, {"code": "gate_failed", "gate": gate, "status": "rolled_back"}
        c.execute("UPDATE deployment SET status = 'retired' WHERE status = 'active' AND mode = 'full'")
        c.execute("UPDATE deployment SET status = 'promoted', decided_by = %s, decision = %s WHERE id = %s", [ctx.actor_id, Jsonb(gate), deployment_id])
        did = uuid.uuid7()
        c.execute("INSERT INTO deployment (id, tenant_id, line_id, model_version_id, mode, share, status, created_by) VALUES (%s,%s,%s,%s,'full',1,'active',%s)", [did, ctx.tenant_id, d["line_id"], d["model_version_id"], ctx.actor_id])
        audit.record(c, ctx, "model.promoted", "deployment", did, {"version": d["version"], "note": body.note})
        return 200, {"status": "promoted", "version": d["version"], "full_deployment_id": did, "gate": gate}
    return run(ctx, idem, body, work)


@app.get("/v1/lines/{line_id}/quality", tags=["analytics"], summary="Yield, reject rate, audit findings and latency for each stretch of the line, by model version; the simulation truth alongside")
def quality(line_id: uuid.UUID, ctx: Ctx = Depends(auth("line:read"))):
    with db.tx(ctx.tenant_id) as c:
        if not c.execute("SELECT 1 FROM line WHERE id = %s", [line_id]).fetchone():
            raise Problem(404, "line_not_found")
        runs = c.execute("SELECT id, frames, params, summary, created_at FROM inspection_run WHERE line_id = %s ORDER BY created_at", [line_id]).fetchall()
        versions = c.execute("SELECT version, metrics, training, created_at FROM model_version ORDER BY version").fetchall()
        deps = c.execute("SELECT d.mode, d.share, d.status, mv.version, d.decision FROM deployment d JOIN model_version mv ON mv.id = d.model_version_id ORDER BY d.created_at").fetchall()
    return jsonable_encoder({"line_id": line_id, "runs": [{"run_id": r["id"], "frames": r["frames"], "yield": r["summary"].get("yield"), "by_arm": {a: {k: v[k] for k in ("version", "frames", "reject_rate", "audited", "audit_found")} | {"truth_false_reject_rate": v["simulation_truth"]["false_reject_rate"], "truth_escape_rate": v["simulation_truth"]["escape_rate"]} for a, v in r["summary"].get("by_arm", {}).items()},
                                                            "p99_ms": r["summary"].get("latency_ms", {}).get("p99")} for r in runs],
                             "model_versions": [{"version": v["version"], "threshold": v["metrics"]["threshold"], "feedback_good": v["training"]["feedback_good"], "labelled_caught": [v["metrics"]["labelled_defects_caught"], v["metrics"]["labelled_defects"]]} for v in versions],
                             "deployments": deps})
