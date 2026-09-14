"""往返会话校正核心：计划生成、漂移拟合、回程间隙、阈值抑制。"""
from __future__ import annotations

import pytest

from foucault.session import (
    FORWARD,
    REVERSE,
    build_plan,
    correct_session,
    fit_drift,
)

THRESHOLDS_OFF = {
    "max_dispersion_mm": 5.0,
    "max_direction_diff_mm": 5.0,
    "max_drift_residual_mm": 5.0,
}


def _obs(seq, t, v, direction=FORWARD):
    return {"seq": seq, "epoch": t, "value_mm": v, "direction": direction}


def _fill(plan, raw):
    """raw: {seq: (epoch, value, locked)} → current map。"""
    return {
        s["seq"]: {
            "epoch": raw[s["seq"]][0],
            "value_mm": raw[s["seq"]][1],
            "locked": raw[s["seq"]][2] if len(raw[s["seq"]]) > 2 else False,
        }
        for s in plan
    }


def test_fit_drift_linear_recovery():
    obs = [
        _obs(1, 0.0, 0.50),
        _obs(4, 10.0, 0.51),
        _obs(7, 20.0, 0.52),
        _obs(11, 30.0, 0.53),
    ]
    fit = fit_drift(obs)
    assert fit["status"] == "ok"
    assert fit["t0_epoch"] == 0.0
    assert fit["slope_mm_per_s"] == pytest.approx(0.001, abs=1e-12)
    assert fit["intercept_mm"] == pytest.approx(0.50, abs=1e-12)
    assert fit["residual_rms_mm"] == pytest.approx(0.0, abs=1e-12)


def test_fit_drift_insufficient_reference():
    fit = fit_drift([_obs(1, 5.0, 1.0)])
    assert fit["status"] == "insufficient_reference"
    assert fit["slope_mm_per_s"] == 0.0
    assert fit["n_points"] == 1
    assert fit_drift([])["status"] == "insufficient_reference"


def test_build_plan_counts_and_directions():
    # refresh=3、3 区：每扫掠末尾（第 3 个常规测位后）插一次复测
    plan = build_plan(3, 2, "forward", 0, 3)
    readings = [s for s in plan if s["kind"] == "reading"]
    refs = [s for s in plan if s["kind"] == "reference"]
    assert len(readings) == 12  # 3 区 × 2 方向 × 2 重复
    # 锚点 + 4 扫掠各 1 次复测
    assert len(refs) == 5
    assert refs[0]["direction"] == FORWARD
    sweeps = {}
    for s in readings:
        sweeps.setdefault(s["sweep"], s["direction"])
    assert [sweeps[k] for k in sorted(sweeps)] == [
        FORWARD, REVERSE, FORWARD, REVERSE
    ]
    assert [s["seq"] for s in plan] == list(range(1, len(plan) + 1))
    # 反向起始：首扫掠方向为 reverse，从最外区开始
    plan2 = build_plan(3, 1, "reverse", 2, 1)
    first = next(s for s in plan2 if s["kind"] == "reading")
    assert first["direction"] == REVERSE and first["zone_index"] == 2


def test_correct_session_backlash_only():
    # 3 区 × 1 重复，refresh=10：计划 = 锚点 + 正向3 + 正末参考 + 反向3 + 反末参考
    plan = build_plan(3, 1, "forward", 0, 10)
    assert len(plan) == 9
    # seq: 1锚点 2-4正向(区0,1,2) 5正末参考 6-8反向(区2,1,0) 9反末参考；
    # 反向统一 +0.2 间隙；起始方向有锚点 + seq5 两个参考点（无漂移）
    raw = {
        1: (0.0, 1.0), 2: (1.0, 1.0), 3: (2.0, 2.0), 4: (3.0, 3.0),
        5: (4.0, 1.0),
        6: (5.0, 3.2), 7: (6.0, 2.2), 8: (7.0, 1.2), 9: (8.0, 1.0),
    }
    rep = correct_session(plan, _fill(plan, raw), THRESHOLDS_OFF,
                          fit_direction=FORWARD)
    assert rep["drift"]["status"] == "ok"
    assert rep["drift"]["slope_mm_per_s"] == pytest.approx(0.0, abs=1e-12)
    assert rep["backlash"]["global_mm"] == pytest.approx(0.2, abs=1e-9)
    for z in rep["zones"]:
        assert z["direction_diff_mm"] == pytest.approx(0.0, abs=1e-9)
    # 间隙校正后每区正反向均值恢复 1/2/3
    means = [z["mean"] for z in rep["zones"]]
    assert means == pytest.approx([1.0, 2.0, 3.0], abs=1e-9)
    assert rep["missing_seqs"] == []
    assert rep["can_finalize"] is True


def test_tail_guarantee_gives_two_start_direction_references():
    # refresh 大于扫掠长度时，每遍扫掠末尾兜底参考点 → 起始方向 ≥2 点可拟合
    plan = build_plan(3, 1, "forward", 0, 100)
    start_refs = [
        s for s in plan
        if s["kind"] == "reference" and s["direction"] == FORWARD
    ]
    assert len(start_refs) == 2  # 锚点 + 正向扫掠末尾兜底
    cur = {
        s["seq"]: {"epoch": float(i), "value_mm": 1.0 + 0.01 * i,
                   "locked": False}
        for i, s in enumerate(plan)
    }
    rep = correct_session(plan, cur, THRESHOLDS_OFF, fit_direction=FORWARD)
    assert not any(
        v["type"] == "drift_underconstrained" for v in rep["violations"]
    )


def test_drift_correction_removes_linear_trend():
    # refresh=3：正向参考点 = 锚点 + 正向扫掠末尾复测，共 2 个，可拟合斜率
    plan = build_plan(3, 1, "forward", 0, 3)
    slope = 0.01
    raw = {}
    for i, s in enumerate(plan):
        t = float(i)
        true_v = 1.0 if s["kind"] == "reference" else float(s["zone_index"] + 1)
        raw[s["seq"]] = (t, true_v + slope * t)
    rep = correct_session(plan, _fill(plan, raw), THRESHOLDS_OFF,
                          fit_direction=FORWARD)
    assert rep["drift"]["status"] == "ok"
    assert rep["drift"]["slope_mm_per_s"] == pytest.approx(slope, abs=1e-9)
    for z in rep["zones"]:
        assert z["mean"] == pytest.approx(float(z["zone_index"] + 1), abs=1e-7)


def test_missing_slots_block_and_are_named():
    plan = build_plan(2, 1, "forward", 0, 10)
    current = {s["seq"]: None for s in plan}
    rep = correct_session(plan, current, THRESHOLDS_OFF, fit_direction=FORWARD)
    assert rep["missing_seqs"] == [s["seq"] for s in plan]
    assert not rep["can_finalize"]
    types = [v["type"] for v in rep["blocking_violations"]]
    assert types[0] == "missing"
    assert "drift_underconstrained" in types


def test_locked_violation_suppressed():
    # 3 区，refresh=10（每扫掠末尾兜底参考点）：
    # seq 1锚点 2-4正向(区0,1,2) 5正末参考 6-8反向(区2,1,0) 9反末参考。
    # 区0 反向偏到 1.9（该区全部锁定）；区1/2 正常 +0.5 回程，中位数取 0.5，
    # 校正后正常区无违例；区0 的离散度/方向差违例因全锁定被抑制。
    plan = build_plan(3, 1, "forward", 0, 10)
    raw = {
        1: (0.0, 1.0, True),
        2: (1.0, 1.0, True),
        3: (2.0, 2.0, False),
        4: (3.0, 3.0, False),
        5: (4.0, 1.0, False),
        6: (5.0, 3.5, False),
        7: (6.0, 2.5, False),
        8: (7.0, 1.9, True),
        9: (8.0, 1.0, False),
    }
    rep = correct_session(
        plan, _fill(plan, raw),
        {"max_dispersion_mm": 0.1, "max_direction_diff_mm": 0.1,
         "max_drift_residual_mm": 5.0},
        fit_direction=FORWARD,
    )
    assert rep["can_finalize"] is True
    assert rep["violations"]
    assert {v["zone_index"] for v in rep["violations"]} == {0}
    for v in rep["violations"]:
        assert v["locked_suppressed"] is True


def test_unlocked_violation_blocks():
    plan = build_plan(3, 1, "forward", 0, 10)
    # 同上但区0 未锁定 → 违例阻断定稿
    raw = {
        1: (0.0, 1.0),
        2: (1.0, 1.0),
        3: (2.0, 2.0),
        4: (3.0, 3.0),
        5: (4.0, 1.0),
        6: (5.0, 3.5),
        7: (6.0, 2.5),
        8: (7.0, 1.9),
        9: (8.0, 1.0),
    }
    rep = correct_session(
        plan, _fill(plan, raw),
        {"max_dispersion_mm": 0.1, "max_direction_diff_mm": 0.1,
         "max_drift_residual_mm": 5.0},
        fit_direction=FORWARD,
    )
    assert rep["can_finalize"] is False
    assert any(
        not v["locked_suppressed"] and v["zone_index"] == 0
        for v in rep["violations"]
    )
