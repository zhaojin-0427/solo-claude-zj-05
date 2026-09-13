"""API 行为测试：校验拒绝、版本化、冻结/剔除、对比、修正搜索。"""
import math

import pytest
from fastapi.testclient import TestClient

from foucault.app import create_app

ZONES = [(0, 22), (22, 45), (45, 65), (65, 85), (85, 100)]


def la_readings(bump=0.0, n=3):
    """接近理想抛物面的读数，边缘区附加 bump(mm) 模拟面形误差。"""
    out = []
    for i, (ri, ro) in enumerate(ZONES):
        rm = math.sqrt((ri**2 + ro**2) / 2.0)
        base = 0.5 + rm**2 / 1600.0
        if i == len(ZONES) - 1:
            base += bump
        out.append([round(base + 0.005 * j, 6) for j in range(n)])
    return out


def make_payload(bump=0.0, **kw):
    payload = {
        "name": "200mm f/4 主镜",
        "diameter": 200.0,
        "radius_of_curvature": 1600.0,
        "conic_constant": -1.0,
        "wavelength_nm": 550.0,
        "source_mode": "fixed",
        "unit": "mm",
        "zones": [
            {"inner_radius": ri, "outer_radius": ro, "readings": rd}
            for (ri, ro), rd in zip(ZONES, la_readings(bump))
        ],
    }
    payload.update(kw)
    return payload


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def create_ok(client, **kw):
    resp = client.post("/api/tests", json=make_payload(**kw))
    assert resp.status_code == 201, resp.json()
    return resp.json()


# ---------------- 创建与校验 ----------------

def test_create_and_auto_version(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    assert body["version"]["version_no"] == 1
    assert body["version"]["input_hash"]
    summary = body["version"]["summary"]
    assert summary["wavefront_rms_waves"] < 0.01  # 近理想抛物面
    assert summary["strehl"] > 0.99
    # 详情接口可拿到分区与读数 ID
    detail = client.get(f"/api/tests/{tid}").json()
    assert len(detail["test"]["zones"]) == 5
    assert detail["test"]["zones"][0]["readings"][0]["id"] > 0
    assert detail["versions"][0]["version_no"] == 1


def test_reject_overlapping_zones(client):
    p = make_payload()
    p["zones"][1] = {"inner_radius": 20, "outer_radius": 45, "readings": [0.7, 0.71]}
    resp = client.post("/api/tests", json=p)
    assert resp.status_code == 422
    assert "重叠" in str(resp.json()["detail"])


def test_reject_zone_beyond_mirror(client):
    p = make_payload()
    p["zones"][-1] = {"inner_radius": 85, "outer_radius": 105, "readings": [5.4, 5.41]}
    resp = client.post("/api/tests", json=p)
    assert resp.status_code == 422
    assert "越出镜面" in str(resp.json()["detail"])


def test_reject_unknown_unit(client):
    resp = client.post("/api/tests", json=make_payload(unit="cm"))
    assert resp.status_code == 422


def test_reject_insufficient_readings(client):
    p = make_payload()
    p["zones"][2]["readings"] = [1.95]  # 只有 1 次，默认最少 2 次
    resp = client.post("/api/tests", json=p)
    assert resp.status_code == 422
    assert "有效读数不足" in str(resp.json()["detail"])


def test_reject_bad_constants(client):
    resp = client.post("/api/tests", json=make_payload(diameter=-200))
    assert resp.status_code == 422
    resp = client.post("/api/tests", json=make_payload(source_mode="laser"))
    assert resp.status_code == 422


def test_unsorted_zones_accepted_and_sorted(client):
    p = make_payload()
    p["zones"] = list(reversed(p["zones"]))
    body = create_ok(client, **{})
    resp = client.post("/api/tests", json=p)
    assert resp.status_code == 201
    zones = resp.json()["test"]["zones"]
    inners = [z["inner_radius"] for z in zones]
    assert inners == sorted(inners)


def test_inch_unit_equivalence(client):
    IN = 25.4
    p_mm = make_payload()
    p_in = make_payload(
        unit="in",
        diameter=200.0 / IN,
        radius_of_curvature=1600.0 / IN,
    )
    for z, rd in zip(p_in["zones"], la_readings()):
        z["inner_radius"] /= IN
        z["outer_radius"] /= IN
        z["readings"] = [v / IN for v in rd]
    r1 = client.post("/api/tests", json=p_mm).json()
    r2 = client.post("/api/tests", json=p_in).json()
    assert r1["version"]["summary"]["wavefront_rms_nm"] == pytest.approx(
        r2["version"]["summary"]["wavefront_rms_nm"], rel=1e-6
    )


# ---------------- 剔除 / 冻结 / 恢复 ----------------

def test_exclude_creates_new_version(client):
    body = create_ok(client, bump=0.4)
    tid = body["test"]["id"]
    v1_rms = body["version"]["summary"]["wavefront_rms_nm"]
    detail = client.get(f"/api/tests/{tid}").json()
    rid = detail["test"]["zones"][4]["readings"][0]["id"]
    resp = client.post(
        f"/api/tests/{tid}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": "回程读数记错行"}]},
    )
    assert resp.status_code == 200
    assert resp.json()["version"]["version_no"] == 2
    assert resp.json()["version"]["summary"]["wavefront_rms_nm"] != v1_rms
    # 剔除记录带原因
    detail = client.get(f"/api/tests/{tid}").json()
    r = detail["test"]["zones"][4]["readings"][0]
    assert r["excluded"] == 1 and r["exclude_reason"] == "回程读数记错行"


def test_exclude_requires_reason(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    detail = client.get(f"/api/tests/{tid}").json()
    rid = detail["test"]["zones"][0]["readings"][0]["id"]
    resp = client.post(
        f"/api/tests/{tid}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": ""}]},
    )
    assert resp.status_code == 422


def test_frozen_reading_cannot_be_excluded(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    detail = client.get(f"/api/tests/{tid}").json()
    rid = detail["test"]["zones"][1]["readings"][0]["id"]
    resp = client.post(f"/api/tests/{tid}/readings/freeze", json={"reading_ids": [rid]})
    assert resp.json()["frozen"] == 1
    resp = client.post(
        f"/api/tests/{tid}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": "试图剔除冻结读数"}]},
    )
    assert resp.status_code == 409
    assert "冻结" in str(resp.json()["detail"])


def test_exclude_below_minimum_rejected(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    detail = client.get(f"/api/tests/{tid}").json()
    rids = [r["id"] for r in detail["test"]["zones"][2]["readings"][:2]]
    resp = client.post(
        f"/api/tests/{tid}/readings/exclude",
        json={"items": [{"reading_id": r, "reason": "x"} for r in rids]},
    )
    assert resp.status_code == 409
    assert "不足" in str(resp.json()["detail"])


def test_restore_reading(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    detail = client.get(f"/api/tests/{tid}").json()
    rid = detail["test"]["zones"][0]["readings"][0]["id"]
    client.post(
        f"/api/tests/{tid}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": "复测后确认可疑"}]},
    )
    resp = client.post(
        f"/api/tests/{tid}/readings/restore", json={"reading_ids": [rid]}
    )
    assert resp.status_code == 200
    detail = client.get(f"/api/tests/{tid}").json()
    assert detail["test"]["zones"][0]["readings"][0]["excluded"] == 0


def test_freeze_all_valid(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    resp = client.post(f"/api/tests/{tid}/readings/freeze", json={"all_valid": True})
    assert resp.json()["frozen"] == 15  # 5 区 × 3 读数


# ---------------- 版本 ----------------

def test_version_immutable_and_deterministic(client):
    body = create_ok(client, bump=0.3)
    tid = body["test"]["id"]
    v1a = client.get(f"/api/tests/{tid}/versions/1").json()
    # 剔除一个读数生成 v2
    detail = client.get(f"/api/tests/{tid}").json()
    rid = detail["test"]["zones"][4]["readings"][0]["id"]
    orig_val = detail["test"]["zones"][4]["readings"][0]["value_original"]
    client.post(
        f"/api/tests/{tid}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": "离群"}]},
    )
    v1b = client.get(f"/api/tests/{tid}/versions/1").json()
    assert v1a == v1b  # 历史版本不受后续操作影响
    v2 = client.get(f"/api/tests/{tid}/versions/2").json()
    assert v2["input_hash"] != v1a["input_hash"]
    # 快照冻结全部原始读数（含被剔除者）及其状态与原因，可完整还原输入
    z4 = v2["snapshot"]["zones"][4]
    assert len(z4["readings"]) == 3  # 被剔除的读数仍在快照中
    exc = [r for r in z4["readings"] if r["excluded"]]
    assert len(exc) == 1
    assert exc[0]["id"] == rid
    assert exc[0]["exclude_reason"] == "离群"
    assert exc[0]["value_original"] == orig_val
    valid = [r for r in z4["readings"] if not r["excluded"]]
    assert len(valid) == 2
    # v1 快照中该区 3 条全部有效
    assert all(not r["excluded"] for r in v1a["snapshot"]["zones"][4]["readings"])


def test_analyze_idempotent_when_unchanged(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    resp = client.post(f"/api/tests/{tid}/analyze", json={})
    assert resp.json()["version"]["reused"] is True
    assert resp.json()["version"]["version_no"] == 1
    # 换算法选项 → 新版本
    resp = client.post(
        f"/api/tests/{tid}/analyze", json={"options": {"error_band_waves": 0.1}}
    )
    assert resp.json()["version"]["version_no"] == 2
    assert resp.json()["version"]["reused"] is False


def test_versions_listing(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    client.post(f"/api/tests/{tid}/analyze", json={"options": {"grid_points": 200}})
    versions = client.get(f"/api/tests/{tid}/versions").json()["versions"]
    assert [v["version_no"] for v in versions] == [1, 2]
    assert all("summary" in v for v in versions)


# ---------------- 批次对比 ----------------

def test_compare_batches(client):
    b1 = create_ok(client, bump=0.0, name="批次A")
    b2 = create_ok(client, bump=0.2, name="批次B")
    resp = client.post(
        "/api/compare",
        json={
            "entries": [
                {"test_id": b1["test"]["id"]},
                {"test_id": b2["test"]["id"]},
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mask_aligned"] is True
    assert len(data["entries"]) == 2
    assert len(data["zonal_comparison"]) == 5
    # 第二批边缘区波前应更负/更正（有 bump）
    last = data["zonal_comparison"][-1]
    assert abs(last["delta_vs_first_nm"][1]) > 0


def test_compare_specific_versions(client):
    body = create_ok(client, bump=0.1)
    tid = body["test"]["id"]
    client.post(f"/api/tests/{tid}/analyze", json={"options": {"grid_points": 300}})
    resp = client.post(
        "/api/compare",
        json={"entries": [{"test_id": tid, "version_no": 1}, {"test_id": tid, "version_no": 2}]},
    )
    assert resp.status_code == 200
    assert resp.json()["entries"][0]["version_no"] == 1


# ---------------- 修正量搜索 ----------------

def test_correction_search(client):
    body = create_ok(client, bump=0.6)  # 边缘区明显误差
    tid = body["test"]["id"]
    resp = client.post(
        f"/api/tests/{tid}/corrections/search",
        json={"max_removal_nm": 200.0, "preserve_edge": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["based_on"]["version_no"] == 1
    cands = data["candidates"]
    assert len(cands) >= 2
    # 约束：边缘保留 + 每区上限
    for c in cands:
        assert c["corrections_nm"][-1] == 0.0
        assert all(0 <= x <= 200.0 + 1e-9 for x in c["corrections_nm"])
    # 排序：rank 0 的剩余波前 RMS 最小，且优于不修正
    orig_rms = body["version"]["summary"]["wavefront_rms_waves"]
    best = cands[0]
    assert best["rank"] == 0
    assert best["metrics"]["residual_wavefront_rms_waves"] < orig_rms
    rms_seq = [c["metrics"]["residual_wavefront_rms_waves"] for c in cands]
    assert rms_seq == sorted(rms_seq)


def test_correction_search_edge_limit(client):
    body = create_ok(client, bump=0.6)
    tid = body["test"]["id"]
    resp = client.post(
        f"/api/tests/{tid}/corrections/search",
        json={"max_removal_nm": 300.0, "edge_zone_max_removal_nm": 20.0},
    )
    assert resp.status_code == 200
    for c in resp.json()["candidates"]:
        assert c["corrections_nm"][-1] <= 20.0 + 1e-9


def test_correction_search_invalid(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    resp = client.post(
        f"/api/tests/{tid}/corrections/search", json={"max_removal_nm": -5}
    )
    assert resp.status_code == 422


# ---------------- 404 ----------------

def test_not_found(client):
    assert client.get("/api/tests/999").status_code == 404
    assert client.get("/api/tests/999/versions/1").status_code == 404
