"""子午线复测研究 API 行为测试。

用 signal_field=la_error（逐批次圆锥/离焦拟合之前的 LA 误差）注入随角度
变化的合成二次像散，使端到端可核对主轴/幅值；覆盖一致性拒绝、欠秩缺口、
来源排除与留一、不可变版本、定稿冻结、复制追加。
"""
import math

import pytest
from fastapi.testclient import TestClient

from foucault.app import create_app

ZONES = [(0, 22), (22, 45), (45, 65), (65, 85), (85, 100)]


def _rmean(ri, ro):
    return math.sqrt((ri * ri + ro * ro) / 2)


RMEAN = [_rmean(*z) for z in ZONES]
PSI = [0.0, 45.0, 90.0, 30.0, 60.0, 120.0, 150.0, 20.0,
       75.0, 100.0, 10.0, 130.0, 55.0, 85.0, 160.0, 25.0]
ALPHA = [0.0, 0.0, 45.0, 90.0, 135.0, 45.0, 90.0, 0.0,
         60.0, 120.0, 30.0, 75.0, 150.0, 15.0, 100.0, 50.0]
AM, PHIM = 0.05, math.radians(30.0)
AL, PHIL = 0.02, math.radians(110.0)


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "study.db"))
    with TestClient(app) as c:
        yield c


def _batch_payload(i, client, psi=PSI, alpha=ALPHA, noise=2e-4,
                   diameter=200.0, roc=1600.0, zones=ZONES, rmean=RMEAN):
    p, a = math.radians(psi[i]), math.radians(alpha[i])
    zone_readings = []
    for j, (ri, ro) in enumerate(zones):
        prof = 0.8 + 0.4 * rmean[j] / 100.0
        g = AM * prof * math.cos(2 * (p - PHIM)) + AL * prof * math.cos(2 * (a - PHIL))
        base = 0.5 + rmean[j] ** 2 / roc + g
        zone_readings.append([round(base + noise * k, 6) for k in range(3)])
    return {
        "diameter": diameter,
        "radius_of_curvature": roc,
        "conic_constant": -1.0,
        "wavelength_nm": 550.0,
        "source_mode": "fixed",
        "unit": "mm",
        "zones": [
            {"inner_radius": ri, "outer_radius": ro, "readings": rd}
            for (ri, ro), rd in zip(zones, zone_readings)
        ],
    }


def _make_batches(client, n=8, **kw):
    ids = []
    for i in range(n):
        r = client.post("/api/tests", json=_batch_payload(i, client, **kw))
        assert r.status_code == 201, r.text
        ids.append(r.json()["test"]["id"])
    return ids


def _sources(ids, psi=PSI, alpha=ALPHA):
    return [
        {
            "test_id": t,
            "mirror_rotation_deg": psi[i],
            "knife_diameter_azimuth_deg": alpha[i],
            "sampled_at": f"2026-01-{i + 1:02d}T10:00:00+00:00",
        }
        for i, t in enumerate(ids)
    ]


def _create_study(client, ids=None, signal_field="la_error", **extra):
    ids = ids if ids is not None else _make_batches(client)
    payload = {"name": "子午线复测", "signal_field": signal_field,
               "sources": _sources(ids)}
    payload.update(extra)
    r = client.post("/api/meridian-studies", json=payload)
    return r, ids


# ---------------- 创建与求解 ----------------

def test_create_study_recovers_components(client):
    r, ids = _create_study(client)
    assert r.status_code == 201, r.text
    body = r.json()
    sid = body["study"]["id"]
    assert body["version"]["version_no"] == 1
    s = body["version"]["summary"]
    assert abs(s["mirror_axis_deg"] - 30.0) < 4.0
    assert abs(s["lab_axis_deg"] - 110.0) < 5.0
    assert s["mirror_rim_amplitude_nm"] > 5.0
    assert s["n_sources"] == 8 and s["n_zones"] == 5
    assert s["rank"] == 5
    assert s["loo_stable"] in (True, False)
    # 详情含来源与角度约定
    detail = client.get(f"/api/meridian-studies/{sid}").json()["study"]
    assert len(detail["sources"]) == 8
    assert detail["angle_convention"]["unit"] == "degree"
    assert detail["signal_field"] == "la_error"
    assert detail["sources"][0]["mirror_rotation_deg"] == 0.0
    assert detail["sources"][0]["sampled_at"].startswith("2026-01-01")


def test_version_detail_has_per_radius_and_ci(client):
    r, _ = _create_study(client)
    sid = r.json()["study"]["id"]
    v = client.get(f"/api/meridian-studies/{sid}/versions/1").json()
    res = v["result"]
    z = res["zones"][-1]
    assert set(["amplitude_mm", "amplitude_ci_mm", "axis_deg", "axis_ci_deg"]) \
        <= set(z["mirror"].keys())
    assert z["mirror"]["amplitude_ci_mm"][0] <= z["mirror"]["amplitude_mm"] \
        <= z["mirror"]["amplitude_ci_mm"][1]
    assert len(z["per_source"]) == 8
    # 整体像散波前与最可疑区段
    assert res["astigmatism"]["mirror"]["rim_wavefront_amplitude_nm"] > 0
    assert 0 <= res["most_suspicious_zone"]["index"] < 5
    # 留一报告
    loo = client.get(f"/api/meridian-studies/{sid}/loo").json()["loo"]
    assert len(loo["leave_one_out"]) == 8


def test_reject_missing_sampled_at(client):
    ids = _make_batches(client, n=4)
    src = _sources(ids)
    for s in src:
        s.pop("sampled_at")
    r = client.post("/api/meridian-studies", json={"sources": src})
    assert r.status_code == 422
    assert "sampled_at" in r.text


def test_four_separable_sources_solve_without_intercept(client):
    # 4 份 ψ/α 独立变化的来源：4 个谐波系数满秩，应正常创建并求解（非 422）
    psi4 = [0.0, 15.0, 30.0, 45.0]
    alpha4 = [0.0, 15.0, 45.0, 60.0]
    ids = _make_batches(client, n=4)
    src = _sources(ids, psi=psi4, alpha=alpha4)
    r = client.post("/api/meridian-studies", json={"sources": src, "loo": False})
    assert r.status_code == 201, r.text
    assert r.json()["version"]["summary"]["rank"] == 4


def test_historical_loo_uses_version_snapshot_not_current_state(client):
    # 建研究（v1，8 份）→ 排除来源 0（v2）；查询 v1 的留一报告仍应基于 8 份来源
    r, _ = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 0, "reason": "异常", "loo": False},
    )
    # v1 未存留一（loo=False）→ 按 v1 冻结结果按需复算，仍是 8 份来源
    resp1 = client.get(f"/api/meridian-studies/{sid}/loo?version_no=1").json()
    assert resp1["computed_on_demand"] is True
    assert len(resp1["loo"]["leave_one_out"]) == 8
    # v2 的留一为 7 份
    resp2 = client.get(f"/api/meridian-studies/{sid}/loo?version_no=2").json()
    assert len(resp2["loo"]["leave_one_out"]) == 7
    # 最新（默认）即 v2
    latest = client.get(f"/api/meridian-studies/{sid}/loo").json()
    assert latest["version_no"] == 2
    assert len(latest["loo"]["leave_one_out"]) == 7


def test_finalize_selected_version_freezes_its_source_set(client):
    # v1（8 份）→ 排除来源 0（v2）→ 指定定稿 v1：冻结包必须按 v1 的 8 份来源，
    # 且每份 active_in_frozen_version=True，不混入当前排除状态。
    r, _ = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 0, "reason": "异常", "loo": False},
    )
    fin = client.post(
        f"/api/meridian-studies/{sid}/finalize", json={"version_no": 1}
    )
    assert fin.status_code == 200, fin.text
    fz = client.get(f"/api/meridian-studies/{sid}/freeze").json()["freeze"]
    assert fz["frozen_version_no"] == 1
    assert fz["n_active_sources"] == 8
    assert len(fz["sources"]) == 8
    assert all(s["active_in_frozen_version"] for s in fz["sources"])
    assert all(not s["excluded"] for s in fz["sources"])


def test_reject_fewer_than_four_sources(client):
    ids = _make_batches(client, n=4)
    payload = {"sources": _sources(ids)[:3]}
    r = client.post("/api/meridian-studies", json=payload)
    assert r.status_code == 422
    assert "at least 4" in r.text or "4" in r.text


def test_reject_diameter_mismatch(client):
    ids = _make_batches(client, n=8)
    # 用不同口径再造一份替换最后来源
    other = _make_batches(client, n=1, diameter=210.0)
    src = _sources(ids[:7] + other)
    r = client.post("/api/meridian-studies", json={"sources": src})
    assert r.status_code == 422
    assert "口径" in str(r.json()["detail"])


def test_reject_roc_mismatch(client):
    ids = _make_batches(client, n=8)
    other = _make_batches(client, n=1, roc=1700.0)
    src = _sources(ids[:7] + other)
    r = client.post("/api/meridian-studies", json={"sources": src})
    assert r.status_code == 422
    assert "曲率半径" in str(r.json()["detail"])


def test_reject_zone_mask_mismatch(client):
    # 分区边界不同（遮罩分区不一致）：追加时先校验不落库
    r, ids = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    other_zones = [(0, 20), (20, 45), (45, 65), (65, 85), (85, 100)]
    other_rmean = [_rmean(*z) for z in other_zones]
    other = _make_batches(client, n=1, zones=other_zones, rmean=other_rmean)
    src = _sources(other, psi=PSI[:1], alpha=ALPHA[:1])
    resp = client.post(f"/api/meridian-studies/{sid}/sources", json={"sources": src})
    assert resp.status_code in (409, 422)
    assert "分区" in str(resp.json()["detail"])
    # 没有半提交：来源数仍为 8
    detail = client.get(f"/api/meridian-studies/{sid}").json()["study"]
    assert len(detail["sources"]) == 8


def test_append_diameter_mismatch_rejected_without_persist(client):
    r, _ = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    other = _make_batches(client, n=1, diameter=210.0)
    src = _sources(other, psi=PSI[:1], alpha=ALPHA[:1])
    resp = client.post(f"/api/meridian-studies/{sid}/sources", json={"sources": src})
    assert resp.status_code == 422
    assert "口径" in str(resp.json()["detail"])
    detail = client.get(f"/api/meridian-studies/{sid}").json()["study"]
    assert len(detail["sources"]) == 8


def test_reject_duplicate_source(client):
    ids = _make_batches(client, n=4)
    src = _sources(ids)
    src = src[:3] + [src[0]]  # 同一冻结版本重复
    r = client.post("/api/meridian-studies", json={"sources": src})
    assert r.status_code == 422
    assert "重复" in str(r.json()["detail"])


def test_underrank_keeps_archive_and_reports_gap(client):
    ids = _make_batches(client)
    # 刀口方位全相同：架位分量不可识别
    src = _sources(ids, alpha=[0.0] * 8)
    r = client.post("/api/meridian-studies", json={"sources": src, "loo": False})
    assert r.status_code == 422
    body = r.json()
    assert body["solve_error"]["errors"]
    assert "架位固定分量无法识别" in body["solve_error"]["errors"][0]
    # 档案已建立，处于开放状态，可补角度后重解
    sid = body["study"]["id"]
    assert body["study"]["status"] == "open"


def test_append_source_and_solve(client):
    r, ids = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    n_before = len(client.get(f"/api/meridian-studies/{sid}").json()["study"]["sources"])
    # 新批次（第 9 份）——再造一个角度
    extra_id = _make_batches(client, n=9)[-1]
    src = {"test_id": extra_id, "mirror_rotation_deg": 75.0,
           "knife_diameter_azimuth_deg": 60.0,
           "sampled_at": "2026-02-01T10:00:00+00:00"}
    resp = client.post(f"/api/meridian-studies/{sid}/sources", json={"sources": [src]})
    assert resp.status_code == 201, resp.text
    assert len(resp.json()["study"]["sources"]) == n_before + 1
    assert resp.json()["version"]["version_no"] == 2


def test_append_over_16_rejected(client):
    # 8 份建研究，再追加到 17 份
    r, _ = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    more = _make_batches(client, n=16)  # 新建 16 个批次
    sources = [
        {"test_id": t, "mirror_rotation_deg": float(10 + i),
         "knife_diameter_azimuth_deg": float(5 + 3 * i),
         "sampled_at": f"2026-02-{i + 1:02d}T10:00:00+00:00"}
        for i, t in enumerate(more)
    ]
    # 分批追加，前 8 份（到 16）成功
    ok = client.post(
        f"/api/meridian-studies/{sid}/sources",
        json={"sources": sources[:8], "solve": False},
    )
    assert ok.status_code == 201
    bad = client.post(
        f"/api/meridian-studies/{sid}/sources",
        json={"sources": sources[8:], "solve": False},
    )
    assert bad.status_code == 422
    assert "16" in str(bad.json()["detail"])


# ---------------- 排除与留一 ----------------

def test_exclude_source_with_reason_creates_version(client):
    r, _ = _create_study(client)
    sid = r.json()["study"]["id"]
    resp = client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 0, "reason": "该次上机疑似碰动刀口"},
    )
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["excluded"]["reason"] == "该次上机疑似碰动刀口"
    assert out["study"]["sources"][0]["excluded"] is True
    assert out["study"]["sources"][0]["exclude_reason"] == "该次上机疑似碰动刀口"
    assert out["version"]["version_no"] == 2
    assert out["version"]["summary"]["n_sources"] == 7


def test_exclude_below_four_blocked(client):
    r, _ = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    for idx in range(4):  # 排除到剩 4 份时再排一次
        rr = client.post(
            f"/api/meridian-studies/{sid}/sources/exclude",
            json={"source_index": idx, "reason": "x", "solve": False},
        )
        assert rr.status_code == 200
    blocked = client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 4, "reason": "x"},
    )
    assert blocked.status_code == 409
    assert "4 份" in str(blocked.json()["detail"])


def test_exclude_then_idempotent_solve(client):
    r, _ = _create_study(client)
    sid = r.json()["study"]["id"]
    client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 0, "reason": "异常"},
    )
    # 输入未变，重复 solve 复用同一版本
    again = client.post(f"/api/meridian-studies/{sid}/solve", json={"loo": True})
    assert again.status_code == 201
    assert again.json()["version"]["reused"] is True


# ---------------- 不可变版本 ----------------

def test_versions_immutable_and_listed(client):
    r, _ = _create_study(client)
    sid = r.json()["study"]["id"]
    client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 1, "reason": "异常"},
    )
    versions = client.get(f"/api/meridian-studies/{sid}").json()["study"]["versions"]
    assert [v["version_no"] for v in versions] == [1, 2]
    v1 = client.get(f"/api/meridian-studies/{sid}/versions/1").json()
    v2 = client.get(f"/api/meridian-studies/{sid}/versions/2").json()
    assert v1["input_hash"] != v2["input_hash"]
    assert len(v1["active_source_ids"]) == 8
    assert len(v2["active_source_ids"]) == 7
    assert v2["excluded_snapshot"][0]["source_index"] == 1
    # 旧版本结果仍可重复读取且不变
    assert client.get(f"/api/meridian-studies/{sid}/versions/1").json() == v1


# ---------------- 定稿冻结 ----------------

def test_finalize_freezes_and_blocks_mutation(client):
    r, _ = _create_study(client)
    sid = r.json()["study"]["id"]
    fin = client.post(f"/api/meridian-studies/{sid}/finalize", json={})
    assert fin.status_code == 200
    assert fin.json()["reused"] is False
    assert fin.json()["input_hash"]
    # 重复定稿幂等
    again = client.post(f"/api/meridian-studies/{sid}/finalize", json={})
    assert again.json()["reused"] is True
    # 冻结后追加 / 排除 / 重解均 409
    ids = _make_batches(client, n=9)
    add = client.post(
        f"/api/meridian-studies/{sid}/sources",
        json={"sources": [{"test_id": ids[-1], "mirror_rotation_deg": 10.0,
                           "knife_diameter_azimuth_deg": 10.0,
                           "sampled_at": "2026-02-01T10:00:00+00:00"}]},
    )
    assert add.status_code == 409
    ex = client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 0, "reason": "x"},
    )
    assert ex.status_code == 409
    solve = client.post(f"/api/meridian-studies/{sid}/solve", json={})
    assert solve.status_code == 409
    # 冻结包
    fz = client.get(f"/api/meridian-studies/{sid}/freeze").json()
    assert fz["input_hash"] == fin.json()["input_hash"]
    assert len(fz["freeze"]["sources"]) == 8
    assert fz["freeze"]["fit_params"]["signal_field"] == "la_error"
    assert fz["freeze"]["angle_convention"]["unit"] == "degree"


def test_finalize_requires_version(client):
    # 欠秩研究无求解版本，定稿 404
    ids = _make_batches(client)
    src = _sources(ids, alpha=[0.0] * 8)
    r = client.post("/api/meridian-studies", json={"sources": src, "loo": False})
    sid = r.json()["study"]["id"]
    fin = client.post(f"/api/meridian-studies/{sid}/finalize", json={})
    assert fin.status_code in (404, 409, 422)


# ---------------- 复制与追加新批次 ----------------

def test_copy_study_resets_exclusion_and_allows_append(client):
    r, _ = _create_study(client, loo=False)
    sid = r.json()["study"]["id"]
    client.post(
        f"/api/meridian-studies/{sid}/sources/exclude",
        json={"source_index": 0, "reason": "异常", "solve": False},
    )
    client.post(f"/api/meridian-studies/{sid}/finalize", json={})
    cp = client.post(f"/api/meridian-studies/{sid}/copy", json={})
    assert cp.status_code == 201, cp.text
    body = cp.json()
    new_id = body["study"]["id"]
    assert new_id != sid
    assert body["study"]["status"] == "open"
    assert len(body["study"]["sources"]) == 8
    assert all(not s["excluded"] for s in body["study"]["sources"])
    # 副本可追加新批次并重解
    more = _make_batches(client, n=9)
    resp = client.post(
        f"/api/meridian-studies/{new_id}/sources",
        json={"sources": [{"test_id": more[-1], "mirror_rotation_deg": 80.0,
                           "knife_diameter_azimuth_deg": 50.0,
                           "sampled_at": "2026-02-01T10:00:00+00:00"}]},
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["version"]["summary"]["n_sources"] == 9
    # 原研究仍冻结
    assert client.get(f"/api/meridian-studies/{sid}").json()["study"]["status"] == "finalized"
