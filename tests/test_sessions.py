"""往返测量会话：计划生成、逐笔校验、漂移/回程校正、锁定/补测/剔除、
状态流转、定稿冻结与幂等批次生成。"""
from __future__ import annotations

import datetime as dt

import pytest
from fastapi.testclient import TestClient

from foucault.app import create_app

MASK = {
    "name": "200mm f/4 遮罩",
    "diameter": 200.0,
    "radius_of_curvature": 1600.0,
    "conic_constant": -1.0,
    "source_mode": "fixed",
    "unit": "mm",
    "zone_count": 5,
    "weighting": "equal_area",
    "center_exclusion_radius": 25.0,
    "min_zone_width": 5.0,
    "bridge_width": 6.0,
    "knife_resolution": 0.5,
    "print_scale": 1.0,
}


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def scheme(client):
    resp = client.post("/api/mask-schemes", json=MASK)
    assert resp.status_code == 201, resp.json()
    body = resp.json()
    return body["scheme"]["id"], body["version"]["version_no"]


def mask_zones(client, scheme_id, version_no=1):
    resp = client.get(f"/api/mask-schemes/{scheme_id}/versions/{version_no}")
    assert resp.status_code == 200
    return resp.json()["layout"]["zones"]


def make_session(client, scheme, **kw):
    sid, vno = scheme
    payload = {
        "mask_scheme_id": sid,
        "mask_version_no": vno,
        "repeats_per_zone": 2,
        "start_direction": "forward",
        "reference_refresh": 2,
        "wavelength_nm": 550.0,
        "collect_deadline": (
            dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
        ).isoformat(),
    }
    payload.update(kw)
    resp = client.post("/api/sessions", json=payload)
    assert resp.status_code == 201, resp.json()
    return resp.json()["session"]


def ideal_knife(zones, zi, t_offset, *, drift=0.0, backlash=0.0, direction="forward",
                reference=False, noise=0.0, bump=None):
    """合成刀口读数：理想抛物面 + 线性漂移 + 回程间隙（反向加）。"""
    if reference:
        v = 0.5 + drift * t_offset
    else:
        rm = zones[zi]["r_mean"]
        v = 0.5 + rm**2 / 1600.0 + drift * t_offset
        if bump is not None and zi == len(zones) - 1:
            v += bump
        if direction == "reverse":
            v += backlash
    return round(v + noise, 6)


def submit_plan(client, session_id, plan, zones, *, t0, interval=30, **kw):
    """按计划顺序提交合成读数，返回 seq → reading id。"""
    ids = {}
    for i, slot in enumerate(plan):
        v = ideal_knife(
            zones,
            slot["zone_index"],
            i * interval,
            direction=slot["direction"],
            reference=slot["kind"] == "reference",
            **kw,
        )
        r = client.post(
            f"/api/sessions/{session_id}/readings",
            json={
                "seq": slot["seq"],
                "knife_position": v,
                "zone_index": slot["zone_index"],
                "direction": slot["direction"],
                "collected_at": (t0 + dt.timedelta(seconds=interval * i)).isoformat(),
            },
        )
        assert r.status_code == 201, r.json()
        ids[slot["seq"]] = r.json()["reading_id"]
    return ids


@pytest.fixture()
def t0():
    return dt.datetime.now(dt.timezone.utc) + dt.timedelta(minutes=1)


# ---------------- 计划与创建 ----------------


def test_create_session_plan_references_immutable_mask(client, scheme):
    sid, vno = scheme
    s = make_session(client, scheme)
    plan = s["plan"]
    # 锚点参考 + 2 重复 × 2 方向 × 5 常规 + 穿插复测
    assert plan[0] == {
        "seq": 1, "kind": "reference", "zone_index": 0,
        "trial": 0, "sweep": 0, "direction": "forward",
    }
    readings = [p for p in plan if p["kind"] == "reading"]
    assert len(readings) == 20  # 5 区 × 2 方向 × 2 重复
    for zi in range(5):
        zslots = [p for p in readings if p["zone_index"] == zi]
        assert len([p for p in zslots if p["direction"] == "forward"]) == 2
        assert len([p for p in zslots if p["direction"] == "reverse"]) == 2
    refs = [p for p in plan if p["kind"] == "reference"]
    assert len(refs) >= 3  # 锚点 + 穿插
    # 起始方向反向：首扫掠从最外区开始
    s2 = make_session(client, scheme, start_direction="reverse")
    first_reading = next(p for p in s2["plan"] if p["kind"] == "reading")
    assert first_reading["direction"] == "reverse"
    assert first_reading["zone_index"] == 4
    # 遮罩另建版本不影响既有会话（冻结引用 v1）
    client.post(
        f"/api/mask-schemes/{sid}/versions",
        json={**MASK, "knife_resolution": 0.25},
    )
    detail = client.get(f"/api/sessions/{s['id']}").json()["session"]
    assert detail["mask_version_no"] == 1
    assert len(detail["plan"]) == len(plan)


def test_session_requires_existing_mask(client):
    resp = client.post(
        "/api/sessions",
        json={
            "mask_scheme_id": 999, "wavelength_nm": 550.0,
            "repeats_per_zone": 2, "reference_refresh": 2,
        },
    )
    assert resp.status_code == 404


def test_reference_zone_out_of_range(client, scheme):
    resp = client.post(
        "/api/sessions",
        json={
            "mask_scheme_id": scheme[0], "wavelength_nm": 550.0,
            "reference_zone_index": 9,
        },
    )
    assert resp.status_code == 422


# ---------------- 逐笔提交校验 ----------------


def test_skip_step_rejected_with_expected_seq(client, scheme, t0):
    s = make_session(client, scheme)
    body = {
        "seq": 2, "knife_position": 1.2, "zone_index": 0,
        "direction": "forward", "collected_at": t0.isoformat(),
    }
    r = client.post(f"/api/sessions/{s['id']}/readings", json=body)
    assert r.status_code == 409
    msg = r.json()["detail"]["message"]
    assert "跳步" in msg and "#1" in msg and "#2" in msg


def test_duplicate_plan_slot_rejected(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(client, scheme)
    slot = s["plan"][0]
    body = {
        "seq": 1, "knife_position": 0.5, "zone_index": 0,
        "direction": "forward", "collected_at": t0.isoformat(),
    }
    assert client.post(f"/api/sessions/{s['id']}/readings", json=body).status_code == 201
    r = client.post(f"/api/sessions/{s['id']}/readings", json=body)
    assert r.status_code == 409
    assert "测次 #1" in r.json()["detail"]["message"]
    assert "重复提交" in r.json()["detail"]["message"]


def test_zone_and_direction_mismatch_rejected(client, scheme, t0):
    s = make_session(client, scheme)
    # seq1 是 zone0/forward 锚点
    for bad in (
        {"zone_index": 1, "direction": "forward"},
        {"zone_index": 0, "direction": "reverse"},
    ):
        body = {
            "seq": 1, "knife_position": 0.5,
            "collected_at": t0.isoformat(), **bad,
        }
        r = client.post(f"/api/sessions/{s['id']}/readings", json=body)
        assert r.status_code == 422
        assert "测次 #1" in str(r.json()["detail"])


def test_collected_after_deadline_rejected(client, scheme):
    deadline = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    s = make_session(client, scheme, collect_deadline=deadline.isoformat())
    body = {
        "seq": 1, "knife_position": 0.5, "zone_index": 0,
        "direction": "forward",
        "collected_at": (deadline + dt.timedelta(minutes=1)).isoformat(),
    }
    r = client.post(f"/api/sessions/{s['id']}/readings", json=body)
    assert r.status_code == 422
    assert "采集时限" in str(r.json()["detail"])


def test_pydantic_rejects_bad_enum_and_nonfinite(client, scheme, t0):
    s = make_session(client, scheme)
    body = {
        "seq": 1, "knife_position": 0.5, "zone_index": 0,
        "direction": "sideways", "collected_at": t0.isoformat(),
    }
    assert client.post(f"/api/sessions/{s['id']}/readings", json=body).status_code == 422
    body["direction"] = "forward"
    body["collected_at"] = "not-a-time"
    assert client.post(f"/api/sessions/{s['id']}/readings", json=body).status_code == 422


# ---------------- 校正：漂移 + 回程间隙 ----------------


def test_drift_and_backlash_correction(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 5.0, "max_direction_diff": 5.0,
                    "max_drift_residual": 5.0},
    )
    submit_plan(
        client, s["id"], s["plan"], zones, t0=t0,
        drift=0.0001, backlash=0.04,
    )
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    assert q["drift"]["slope_mm_per_s"] == pytest.approx(0.0001, abs=1e-9)
    assert q["drift"]["fit_observations"] == "start_direction"
    assert q["drift"]["residual_rms_mm"] == pytest.approx(0.0, abs=1e-9)
    assert q["backlash"]["global_mm"] == pytest.approx(0.04, abs=1e-6)
    # 校正后正反向均值对齐、每区离散度趋零
    for z in q["zones"]:
        assert z["direction_diff_mm"] == pytest.approx(0.0, abs=1e-6)
        assert z["range_mm"] == pytest.approx(0.0, abs=1e-6)
    assert q["can_finalize"] is True


def test_missing_slots_block_finalize_with_seqs(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(client, scheme)
    # 只提交前 3 个测次
    for slot in s["plan"][:3]:
        r = client.post(
            f"/api/sessions/{s['id']}/readings",
            json={
                "seq": slot["seq"], "knife_position": 0.5,
                "zone_index": slot["zone_index"], "direction": slot["direction"],
                "collected_at": (t0 + dt.timedelta(seconds=slot["seq"])).isoformat(),
            },
        )
        assert r.status_code == 201
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    missing = q["missing_seqs"]
    assert missing[0] == 4 and missing[-1] == len(s["plan"])
    r = client.post(f"/api/sessions/{s['id']}/finalize")
    assert r.status_code == 409
    assert "漏测" in str(r.json()["detail"])
    assert "#4" in str(r.json()["detail"])


def test_dispersion_violation_blocks_finalize(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    # 严格阈值：合成读数两次重复一致，但加 bump 到同一区只影响单次 → 极差超标
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 0.01, "max_direction_diff": 5.0,
                    "max_drift_residual": 5.0},
    )
    ids = submit_plan(
        client, s["id"], s["plan"], zones, t0=t0,
        drift=0.0, backlash=0.0,
    )
    # 对最外区某一测次补测一个明显偏离值
    edge_seqs = [
        p["seq"] for p in s["plan"]
        if p["kind"] == "reading" and p["zone_index"] == 4
    ]
    seq = edge_seqs[0]
    # 邻近时间补测异常值：只污染该测位，不改变漂移拟合
    r = client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": seq, "knife_position": 9.0, "zone_index": 4,
            "direction": "forward",
            "collected_at": (t0 + dt.timedelta(seconds=30 * (seq - 1) + 5)).isoformat(),
        },
    )
    assert r.status_code == 201 and r.json()["attempt"] == 2
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    types = {(v["type"], v.get("zone_index")) for v in q["blocking_violations"]}
    assert ("dispersion", 4) in types
    r = client.post(f"/api/sessions/{s['id']}/finalize")
    assert r.status_code == 409
    detail = str(r.json()["detail"])
    assert "离散度" in detail and f"#{seq}" in detail


def test_direction_diff_violation_blocks_finalize(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 5.0, "max_direction_diff": 0.01,
                    "max_drift_residual": 5.0},
    )
    # 正常提交（无回程间隙）后，仅对一个区的反向测次补测偏移，制造非均匀方向差
    submit_plan(client, s["id"], s["plan"], zones, t0=t0)
    rev_seq = next(
        p["seq"] for p in s["plan"]
        if p["kind"] == "reading" and p["zone_index"] == 2
        and p["direction"] == "reverse"
    )
    rm = zones[2]["r_mean"]
    client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": rev_seq, "knife_position": 0.5 + rm**2 / 1600.0 + 0.05,
            "zone_index": 2, "direction": "reverse",
            "collected_at": (t0 + dt.timedelta(seconds=30 * (rev_seq - 1) + 5)).isoformat(),
        },
    )
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    assert any(
        v["type"] == "direction_diff" and v.get("zone_index") == 2
        for v in q["blocking_violations"]
    )
    r = client.post(f"/api/sessions/{s['id']}/finalize")
    assert r.status_code == 409
    assert "回程间隙" in str(r.json()["detail"])


def test_drift_residual_violation_blocks_finalize(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 5.0, "max_direction_diff": 5.0,
                    "max_drift_residual": 0.005},
    )
    # 参考复测线性良好（残差≈0），然后对其中一个参考测次补测异常值
    submit_plan(client, s["id"], s["plan"], zones, t0=t0, drift=0.0001)
    ref_seqs = [p["seq"] for p in s["plan"] if p["kind"] == "reference"]
    seq = ref_seqs[2]
    client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": seq, "knife_position": 5.0, "zone_index": 0,
            "direction": "forward",
            "collected_at": (t0 + dt.timedelta(seconds=30 * (seq - 1) + 5)).isoformat(),
        },
    )
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    assert any(v["type"] == "drift_residual" for v in q["blocking_violations"])
    r = client.post(f"/api/sessions/{s['id']}/finalize")
    assert r.status_code == 409
    assert f"#{seq}" in str(r.json()["detail"])


# ---------------- 锁定 / 补测 / 剔除 ----------------


def test_lock_suppresses_blocking_violation(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 0.01, "max_direction_diff": 5.0,
                    "max_drift_residual": 5.0},
    )
    ids = submit_plan(client, s["id"], s["plan"], zones, t0=t0)
    edge_seqs = [
        p["seq"] for p in s["plan"]
        if p["kind"] == "reading" and p["zone_index"] == 4
    ]
    client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": edge_seqs[0], "knife_position": 9.0, "zone_index": 4,
            "direction": "forward",
            "collected_at": (
                t0 + dt.timedelta(seconds=30 * (edge_seqs[0] - 1) + 5)
            ).isoformat(),
        },
    )
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    assert not q["can_finalize"]
    # 锁定该区全部测次 → 违例仍列出但被抑制
    r = client.post(
        f"/api/sessions/{s['id']}/lock", json={"seqs": edge_seqs}
    )
    assert r.status_code == 200
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    assert q["can_finalize"] is True
    # 被锁的最外区违例仍列出但已抑制
    assert all(v["locked_suppressed"] for v in q["violations"])
    assert {v["zone_index"] for v in q["violations"]} == {4}
    # 锁定读数不能剔除
    rid = next(
        r["id"] for r in client.get(f"/api/sessions/{s['id']}").json()["session"]["readings"]
        if r["seq"] == edge_seqs[0] and not r["excluded"]
    )
    r = client.post(
        f"/api/sessions/{s['id']}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": "想剔除锁定值"}]},
    )
    assert r.status_code == 409
    # 锁定测次不能补测
    r = client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": edge_seqs[0], "knife_position": 8.0, "zone_index": 4,
            "direction": "forward",
            "collected_at": (t0 + dt.timedelta(hours=3)).isoformat(),
        },
    )
    assert r.status_code == 409


def test_retest_keeps_history_and_uses_latest(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(client, scheme)
    slot = s["plan"][0]
    for k in range(3):
        r = client.post(
            f"/api/sessions/{s['id']}/readings/retest",
            json={
                "seq": 1, "knife_position": 0.5 + 0.01 * k, "zone_index": 0,
                "direction": "forward",
                "collected_at": (t0 + dt.timedelta(minutes=k)).isoformat(),
            },
        )
        assert r.status_code == 201 and r.json()["attempt"] == k + 1
    detail = client.get(f"/api/sessions/{s['id']}").json()["session"]
    attempts = [r for r in detail["readings"] if r["seq"] == 1]
    assert [r["attempt"] for r in attempts] == [1, 2, 3]


def test_exclude_with_reason_then_retest_then_finalize(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 0.01, "max_direction_diff": 5.0,
                    "max_drift_residual": 5.0},
    )
    ids = submit_plan(client, s["id"], s["plan"], zones, t0=t0)
    edge_seqs = [
        p["seq"] for p in s["plan"]
        if p["kind"] == "reading" and p["zone_index"] == 4
    ]
    bad_id = None
    client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": edge_seqs[0], "knife_position": 9.0, "zone_index": 4,
            "direction": "forward",
            "collected_at": (
                t0 + dt.timedelta(seconds=30 * (edge_seqs[0] - 1) + 5)
            ).isoformat(),
        },
    )
    detail = client.get(f"/api/sessions/{s['id']}").json()["session"]
    bad_id = max(
        (r["id"] for r in detail["readings"] if r["seq"] == edge_seqs[0]),
    )
    # 剔除必须注明原因；空原因 422
    r = client.post(
        f"/api/sessions/{s['id']}/readings/exclude",
        json={"items": [{"reading_id": bad_id, "reason": ""}]},
    )
    assert r.status_code == 422
    r = client.post(
        f"/api/sessions/{s['id']}/readings/exclude",
        json={"items": [{"reading_id": bad_id, "reason": "读数时受震动干扰"}]},
    )
    assert r.status_code == 200
    # 剔除后该测位回到 attempt1 有效值（attempt 历史保留），可定稿
    q = client.get(f"/api/sessions/{s['id']}/quality").json()
    assert q["can_finalize"] is True
    detail = client.get(f"/api/sessions/{s['id']}").json()["session"]
    rows = [r for r in detail["readings"] if r["seq"] == edge_seqs[0]]
    assert rows[-1]["excluded"] == 1
    assert rows[-1]["exclude_reason"] == "读数时受震动干扰"


# ---------------- 状态流转与定稿 ----------------


def test_confirm_flow_and_revert_on_new_data(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 5.0, "max_direction_diff": 5.0,
                    "max_drift_residual": 5.0},
    )
    # 未采完不能确认
    r = client.post(f"/api/sessions/{s['id']}/confirm")
    assert r.status_code == 409 and "漏测" in str(r.json()["detail"])
    submit_plan(client, s["id"], s["plan"], zones, t0=t0)
    r = client.post(f"/api/sessions/{s['id']}/confirm")
    assert r.status_code == 200
    assert r.json()["session"]["status"] == "confirmed"
    # 待确认状态补测 → 回到采集中
    seq2 = next(p["seq"] for p in s["plan"] if p["kind"] == "reading")
    client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": seq2, "knife_position": 1.23, "zone_index": 0,
            "direction": "forward",
            "collected_at": (t0 + dt.timedelta(hours=2)).isoformat(),
        },
    )
    detail = client.get(f"/api/sessions/{s['id']}").json()["session"]
    assert detail["status"] == "collecting"


def test_finalize_generates_batch_and_is_idempotent(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme, name="上机会话 A",
        thresholds={"max_dispersion": 5.0, "max_direction_diff": 5.0,
                    "max_drift_residual": 5.0},
    )
    submit_plan(client, s["id"], s["plan"], zones, t0=t0, drift=0.00005, backlash=0.02)
    r = client.post(f"/api/sessions/{s['id']}/finalize")
    assert r.status_code == 200, r.json()
    body = r.json()
    assert body["reused"] is False
    test_id = body["test_id"]
    assert body["version"]["version_no"] == 1
    # 重复定稿返回同一批次
    r2 = client.post(f"/api/sessions/{s['id']}/finalize")
    assert r2.status_code == 200
    assert r2.json()["reused"] is True
    assert r2.json()["test_id"] == test_id
    # 现有分析接口可直接读取
    t = client.get(f"/api/tests/{test_id}").json()["test"]
    assert t["mask_scheme_id"] == scheme[0]
    assert t["mask_version_no"] == scheme[1]
    assert [len(z["readings"]) for z in t["zones"]] == [4, 4, 4, 4, 4]
    versions = client.get(f"/api/tests/{test_id}/versions").json()["versions"]
    assert len(versions) == 1
    # 近理想数据：波前 RMS 很低
    v = client.get(f"/api/tests/{test_id}/versions/1").json()
    assert v["result"]["summary"]["wavefront_rms_waves"] < 0.01


def test_finalize_freezes_raw_and_input_hash(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(
        client, scheme,
        thresholds={"max_dispersion": 5.0, "max_direction_diff": 5.0,
                    "max_drift_residual": 5.0},
    )
    submit_plan(client, s["id"], s["plan"], zones, t0=t0)
    body = client.post(f"/api/sessions/{s['id']}/finalize").json()
    freeze = client.get(f"/api/sessions/{s['id']}/freeze").json()
    assert freeze["input_hash"] == body["input_hash"]
    bundle = freeze["freeze"]
    assert bundle["mask"]["params_hash"]
    assert abs(bundle["correction"]["drift"]["t0_epoch"]) > 0
    assert len(bundle["raw_readings"]) == len(s["plan"])
    # 全部 attempt 已冻结
    detail = client.get(f"/api/sessions/{s['id']}").json()["session"]
    assert all(r["frozen"] for r in detail["readings"])
    # 定稿后追加数据：409 且指明测次
    r = client.post(
        f"/api/sessions/{s['id']}/readings/retest",
        json={
            "seq": 2, "knife_position": 1.0, "zone_index": 0,
            "direction": "forward",
            "collected_at": t0.isoformat(),
        },
    )
    assert r.status_code == 409
    msg = r.json()["detail"]["message"]
    assert "已定稿" in msg and f"#{2}" in msg
    # 定稿后剔除同样拒绝
    rid = detail["readings"][0]["id"]
    r = client.post(
        f"/api/sessions/{s['id']}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": "定稿后剔除"}]},
    )
    assert r.status_code == 409


def test_lock_all_collected(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s = make_session(client, scheme)
    submit_plan(client, s["id"], s["plan"][:5], zones, t0=t0)
    r = client.post(f"/api/sessions/{s['id']}/lock", json={"all_collected": True})
    assert r.status_code == 200 and r.json()["locked"] == 5
    # 未采集测次不能锁定
    r = client.post(f"/api/sessions/{s['id']}/lock", json={"seqs": [999]})
    assert r.status_code == 422


def test_exclude_reading_from_other_session_rejected(client, scheme, t0):
    zones = mask_zones(client, scheme[0])
    s1 = make_session(client, scheme)
    s2 = make_session(client, scheme)
    slot = s1["plan"][0]
    r = client.post(
        f"/api/sessions/{s1['id']}/readings",
        json={
            "seq": 1, "knife_position": 0.5, "zone_index": 0,
            "direction": "forward", "collected_at": t0.isoformat(),
        },
    )
    rid = r.json()["reading_id"]
    r = client.post(
        f"/api/sessions/{s2['id']}/readings/exclude",
        json={"items": [{"reading_id": rid, "reason": "跨会话"}]},
    )
    assert r.status_code == 409


def test_session_not_found(client):
    assert client.get("/api/sessions/999").status_code == 404
    assert client.post("/api/sessions/999/confirm").status_code == 404
