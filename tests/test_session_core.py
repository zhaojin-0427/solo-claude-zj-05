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
    # 3 区 × 1 重复，refresh=10：计划 = 锚点 + 正向3 + 反向3（扫掠内不插复测）
    plan = build_plan(3, 1, "forward", 0, 10)
    assert len(plan) == 7
    # seq: 1锚点 2-4正向(区0,1,2) 5-7反向(区2,1,0)；反向统一 +0.2 间隙
    raw = {
        1: (0.0, 1.0), 2: (1.0, 1.0), 3: (2.0, 2.0), 4: (3.0, 3.0),
        5: (5.0, 3.2), 6: (6.0, 2.2), 7: (8.0, 1.2),
    }
    rep = correct_session(plan, _fill(plan, raw), THRESHOLDS_OFF,
                          fit_direction=FORWARD)
    # 仅一个参考观测 → 漂移不拟合（斜率 0），状态明确标注
    assert rep["drift"]["status"] == "insufficient_reference"
    assert rep["drift"]["slope_mm_per_s"] == 0.0
    assert rep["backlash"]["global_mm"] == pytest.approx(0.2, abs=1e-9)
    for z in rep["zones"]:
        assert z["direction_diff_mm"] == pytest.approx(0.0, abs=1e-9)
    # 间隙校正后每区正反向均值恢复 1/2/3
    means = [z["mean"] for z in rep["zones"]]
    assert means == pytest.approx([1.0, 2.0, 3.0], abs=1e-9)
    assert rep["missing_seqs"] == []
    assert rep["can_finalize"] is True


def test_drift_correction_removes_linear_trend():
    # refresh=3：正向参考点 = 锚点 + 各正向扫掠末尾复测，共 3 个，可拟合斜率
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
    assert rep["blocking_violations"][0]["type"] == "missing"


def test_locked_violation_suppressed():
    # 3 区，refresh=10：1锚点 2-4正向(区0,1,2) 5-7反向(区2,1,0)。
    # 区0 反向偏到 1.9（该区全部锁定）；区1/2 正常 +0.5 回程，中位数取 0.5，
    # 校正后正常区无违例；区0 的离散度/方向差违例因全锁定被抑制。
    plan = build_plan(3, 1, "forward", 0, 10)
    raw = {
        1: (0.0, 1.0, True),
        2: (1.0, 1.0, True),
        3: (2.0, 2.0, False),
        4: (3.0, 3.0, False),
        5: (4.0, 3.5, False),
        6: (5.0, 2.5, False),
        7: (6.0, 1.9, True),
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
        5: (4.0, 3.5),
        6: (5.0, 2.5),
        7: (6.0, 1.9),
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
