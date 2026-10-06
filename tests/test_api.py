"""Integration, end-to-end and security tests against a real Postgres."""
import psycopg
import pytest

from core import db, jobs, scenario

from .conftest import bearer

MANAGER, ENGINEER, OPERATOR, VIEWER = (bearer("plant", r) for r in ("quality_manager", "quality_engineer", "operator", "viewer"))
OTHER = bearer("other", "quality_manager")


def test_nothing_runs_before_a_line_and_a_model(client):
    assert client.post("/v1/models/train", headers=ENGINEER, json={}).json()["code"] == "no_line"
    assert client.post("/v1/lines:load", headers=ENGINEER, json={}).status_code == 403


@pytest.fixture(scope="module")
def demo(client):
    results, _ = scenario.run(client, drain=jobs.drain)
    return results


def one(sql, params=(), tenant=None):
    with db.tx(tenant) as c:
        return c.execute(sql, params).fetchone()


def test_demo_journey(demo):
    assert demo["load"]["result"]["training_set"]["labelled_defects"] == 24 and one("SELECT count(*) AS n FROM annotation")["n"] == 374 + 40
    v1 = demo["v1"]["result"]
    assert v1["version"] == 1 and v1["calibration_false_reject"] <= 0.02 and v1["labelled_defects_caught"] > v1["baseline_labelled_caught"]
    s1 = demo["s1"]["result"]
    c1 = s1["by_arm"]["control"]
    assert s1["frames"] == 3000 and one("SELECT count(*) AS n FROM inspection WHERE run_id = %s", [s1["run_id"]])["n"] == 3000
    assert s1["timeline"][0]["control_reject_rate"] < 0.15 and s1["timeline"][-1]["control_reject_rate"] > 0.5     # the supplier change
    assert c1["simulation_truth"]["false_reject_rate_new_supplier"] > 0.9 and s1["latency_ms"]["p50"] < 33
    assert demo["defects"]["items"][0]["png"].startswith("data:image/png;base64,") and len(demo["defects"]["items"][0]["map"]) == 225
    assert len(demo["queue"]["items"]) == 40 and demo["review"]["recorded"] == 40 and demo["review"]["false_positives"] > 20
    v2 = demo["v2"]["result"]
    assert v2["version"] == 2 and v2["training"]["feedback_good"] == demo["review"]["false_positives"] and v2["adapted_positions"] > 0
    s2 = demo["s2"]["result"]["by_arm"]
    assert 0.1 < s2["canary"]["frames"] / 3000 < 0.3 and s2["canary"]["reject_rate"] < s2["control"]["reject_rate"] / 2
    assert demo["promote"]["status"] == "promoted" and one("SELECT count(*) AS n FROM deployment WHERE status = 'active' AND mode = 'full'")["n"] == 1
    assert len(demo["quality"]["runs"]) == 2 and [v["version"] for v in demo["quality"]["model_versions"]] == [1, 2]
    assert demo["audit"]["chain_valid"] and {"line.loaded", "model.trained", "model.deployed", "inspection.run", "feedback.recorded", "model.promoted"} <= {e["action"] for e in demo["audit"]["events"]}


def test_feedback_rules_and_deployment_rules(client, demo):
    run1 = demo["s1"]["result"]["run_id"]
    reviewed = demo["queue"]["items"][0]["inspection_id"]
    passed = one("SELECT id FROM inspection WHERE run_id = %s AND decision = 'pass' LIMIT 1", [run1])["id"]
    out = client.post("/v1/annotations", headers=OPERATOR, json={"operator": "J. Ortiz", "items": [{"inspection_id": reviewed, "verdict": "false_positive"}, {"inspection_id": str(passed), "verdict": "false_positive"}]}).json()
    assert out["recorded"] == 0 and {r["why"] for r in out["refused"]} == {"already reviewed", "only rejected frames are reviewed"}
    assert client.post("/v1/annotations", headers=VIEWER, json={"operator": "x", "items": [{"inspection_id": reviewed, "verdict": "false_positive"}]}).status_code == 403
    v1 = demo["v1"]["result"]["model_version_id"]
    assert client.post("/v1/deployments", headers=ENGINEER, json={"model_version_id": v1, "mode": "canary", "share": 0.8}).json()["code"] == "canary_too_large"
    assert client.post(f"/v1/deployments/{demo['dep2']['deployment_id']}/promote", headers=MANAGER, json={"run_id": run1, "note": "again"}).json()["code"] == "not_an_active_canary"
    h = {**ENGINEER, "Idempotency-Key": "dep-replay"}
    a, b = client.post("/v1/deployments", headers=h, json={"model_version_id": v1, "mode": "canary", "share": 0.1}), client.post("/v1/deployments", headers=h, json={"model_version_id": v1, "mode": "canary", "share": 0.1})
    assert a.json() == b.json() and b.headers.get("Idempotent-Replay") == "true"
    assert client.post(f"/v1/deployments/{a.json()['deployment_id']}/promote", headers=MANAGER, json={"run_id": run1, "note": "no canary arm in that run"}).json()["code"] == "no_canary_evidence"


def test_tampered_artifact_is_refused(client, demo):
    from qi import api
    v2 = demo["v2"]["result"]["model_version_id"]
    api._models.clear()
    with db.tx() as c:
        c.execute("UPDATE model_version SET sha256 = 'x' WHERE id = %s", [v2])
    try:
        assert client.post("/v1/deployments", headers=ENGINEER, json={"model_version_id": v2, "mode": "full"}).json()["code"] == "artifact_tampered"
    finally:
        with db.tx() as c:
            c.execute("UPDATE model_version SET sha256 = encode(sha256(artifact), 'hex') WHERE id = %s", [v2])


def test_tenant_isolation_and_append_only_audit(client, demo):
    line_id, run1 = demo["load"]["result"]["line_id"], demo["s1"]["result"]["run_id"]
    assert client.get(f"/v1/lines/{line_id}/quality", headers=OTHER).status_code == 404
    assert client.get(f"/v1/defects?run_id={run1}", headers=OTHER).json()["items"] == []
    with db.tx() as c:
        other = c.execute("SELECT DISTINCT tenant_id FROM api_tokens WHERE actor_name LIKE '%%Halvorsen%%'").fetchone()["tenant_id"]
    assert one("SELECT count(*) AS n FROM inspection", tenant=other)["n"] == 0 and one("SELECT count(*) AS n FROM image", tenant=other)["n"] == 0
    with pytest.raises(psycopg.errors.RaiseException), db.tx() as c:
        c.execute("UPDATE audit_events SET action = 'edited'")
