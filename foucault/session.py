"""往返测量会话的校正核心：计划生成、零点漂移拟合、回程间隙估计。

一次刀口仪上机测量 = 一份往返（forward / reverse 两个扫掠方向）测量会话。
计划测位（slot，对外称"测次"，序号从 1 开始）由不可变遮罩版本的环带与
会话参数（每区重复次数、起始方向、参考区复测频率）确定：

- 计划开头先测一次参考区作为漂移锚点；
- 每次重复包含起始方向扫掠与反向扫掠各一遍（forward = 从内向外，
  reverse = 从外向内）；
- 每个扫掠内每测量 reference_refresh 个常规测位，穿插一次参考区复测。

校正（全部内部使用 mm，时间使用 epoch 秒）：
1. 零点线性漂移：参考锚点与各次复测 (t, x) 拟合 x = a + b·(t − t0)，
   每条读数扣除 b·(t − t0)（常数 a 与仪器零位一样由后续零点参考扣除）。
   优先使用起始方向上的参考观测拟合（避免回程间隙污染），不足 2 个时
   退回全部方向并在结果中标注。
2. 回程间隙：逐区计算反向校正均值 − 正向校正均值，全局取各区均值；
   反向读数统一扣除该全局间隙，使正反向均值对齐。
3. 离散度：每区校正读数的均值 / 样本标准差 / 极差；方向差为该区反向均值
   减正向均值；漂移残差为参考观测相对拟合直线的偏差。
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np

FORWARD = "forward"
REVERSE = "reverse"
DIRECTIONS = (FORWARD, REVERSE)


class SessionError(Exception):
    """会话计划/校正前置条件不满足。"""


def opposite(direction: str) -> str:
    return REVERSE if direction == FORWARD else FORWARD


def _aware(dt: datetime) -> datetime:
    """ naive 时间按 UTC 处理；带时区时间统一换算到 UTC。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def epoch_seconds(dt: datetime) -> float:
    return _aware(dt).timestamp()


def iso(dt: datetime) -> str:
    return _aware(dt).isoformat()


# ---------------- 测量计划 ----------------


def build_plan(
    zone_count: int,
    repeats_per_zone: int,
    start_direction: str,
    reference_zone_index: int,
    reference_refresh: int,
) -> list[dict]:
    """生成往返测量计划测位列表（seq 从 1 开始连续编号）。

    每个 slot：{seq, kind('reading'|'reference'), zone_index, trial, sweep,
    direction}。锚点参考观测 trial/sweep 记为 0；常规测位每区每方向各
    repeats_per_zone 次（每区合计 2·repeats 次有效读数）。

    参考复测每 reference_refresh 个常规测位穿插一次；无论该频率多大，
    每遍扫掠末尾保证再补一次，使任一方向都有 ≥2 个参考点（锚点 + 复测），
    足以拟合该方向上的线性漂移。
    """
    if zone_count < 2:
        raise SessionError("遮罩至少需要两个分区才能建立往返测量计划")
    if not (0 <= reference_zone_index < zone_count):
        raise SessionError(
            f"参考区序号 {reference_zone_index} 超出分区范围 [0, {zone_count - 1}]"
        )
    if start_direction not in DIRECTIONS:
        raise SessionError(f"未知起始方向: {start_direction!r}")
    if repeats_per_zone < 1:
        raise SessionError("每区重复次数至少为 1")
    if reference_refresh < 1:
        raise SessionError("参考区复测频率至少为 1")

    slots: list[dict] = []
    seq = 1

    def add(kind: str, zone: int, trial: int, sweep: int, direction: str) -> None:
        nonlocal seq
        slots.append(
            {
                "seq": seq,
                "kind": kind,
                "zone_index": zone,
                "trial": trial,
                "sweep": sweep,
                "direction": direction,
            }
        )
        seq += 1

    # 漂移锚点：上机后首次参考区测量
    add("reference", reference_zone_index, 0, 0, start_direction)
    for trial in range(1, repeats_per_zone + 1):
        for si, direction in enumerate((start_direction, opposite(start_direction))):
            sweep = (trial - 1) * 2 + si + 1
            order = (
                range(zone_count)
                if direction == FORWARD
                else range(zone_count - 1, -1, -1)
            )
            since_ref = 0
            added_in_sweep = False
            for zi in order:
                add("reading", zi, trial, sweep, direction)
                since_ref += 1
                if since_ref % reference_refresh == 0:
                    add("reference", reference_zone_index, trial, sweep, direction)
                    since_ref = 0
                    added_in_sweep = True
            # 每遍扫掠至少一次参考复测（refresh 大于扫掠长度等情况下兜底）
            if not added_in_sweep:
                add("reference", reference_zone_index, trial, sweep, direction)
    return slots


# ---------------- 漂移拟合 ----------------


def _line_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """最小二乘 y = intercept + slope·x；x 无跨度时斜率取 0。"""
    if x.size >= 2 and float(x.max() - x.min()) > 0.0:
        slope, intercept = np.polyfit(x, y, 1)
        return float(slope), float(intercept)
    return 0.0, float(y.mean()) if y.size else 0.0


def fit_drift(observations: list[dict]) -> dict:
    """根据参考区观测拟合零点线性漂移。

    observations 元素需含 epoch（秒）、value_mm、seq、direction。
    优先使用起始方向的观测（含锚点）；不足 2 个时退回全部方向。
    返回 {status, t0, slope_mm_per_s, intercept_mm, n_points, residuals,
    residual_rms_mm}，residuals 元素含 seq 与 residual_mm。
    """
    obs = sorted(observations, key=lambda o: o["epoch"])

    def _fit(points: list[dict], status: str) -> dict:
        t0 = points[0]["epoch"] if points else 0.0
        x = np.array([p["epoch"] - t0 for p in points], dtype=float)
        y = np.array([p["value_mm"] for p in points], dtype=float)
        slope, intercept = _line_fit(x, y)
        resid = y - (intercept + slope * x)
        residuals = [
            {"seq": p["seq"], "residual_mm": float(r)}
            for p, r in zip(points, resid)
        ]
        rms = float(math.sqrt(float((resid**2).mean()))) if resid.size else 0.0
        return {
            "status": status,
            "t0_epoch": float(t0),
            "slope_mm_per_s": slope,
            "intercept_mm": intercept,
            "n_points": len(points),
            "residuals": residuals,
            "residual_rms_mm": rms,
        }

    if len(obs) < 2:
        return _fit(obs, "insufficient_reference")
    return _fit(obs, "ok")


# ---------------- 会话校正与质量报告 ----------------


def _stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    n = int(arr.size)
    return {
        "n": n,
        "mean": float(arr.mean()) if n else None,
        "std": float(arr.std(ddof=1)) if n > 1 else 0.0,
        "range": float(arr.max() - arr.min()) if n else 0.0,
        "min": float(arr.min()) if n else None,
        "max": float(arr.max()) if n else None,
    }


def correct_session(
    plan: list[dict],
    current: dict,
    thresholds: dict,
    fit_direction: str | None = None,
) -> dict:
    """执行漂移/回程校正并汇总质量指标。

    current: {seq: 当前有效读数 dict 或 None}；读数含 value_mm、epoch、
    direction、locked。None 或缺失表示该测位漏测。
    thresholds: max_dispersion_mm / max_direction_diff_mm /
    max_drift_residual_mm（mm）。
    fit_direction: 漂移拟合优先使用的方向（通常为起始方向）。
    """
    slots_by_seq = {s["seq"]: s for s in plan}
    zone_count = max(s["zone_index"] for s in plan) + 1

    # 1) 参考观测（仅当前有效读数）
    ref_all = []
    for slot in plan:
        if slot["kind"] != "reference":
            continue
        r = current.get(slot["seq"])
        if r is not None:
            ref_all.append(
                {
                    "seq": slot["seq"],
                    "epoch": r["epoch"],
                    "value_mm": r["value_mm"],
                    "direction": slot["direction"],
                    "locked": bool(r.get("locked")),
                }
            )
    preferred = [o for o in ref_all if o["direction"] == fit_direction]
    fit_obs = preferred if len(preferred) >= 2 else ref_all
    drift = fit_drift(fit_obs)
    if len(preferred) >= 2:
        drift["fit_observations"] = "start_direction"
    elif len(ref_all) >= 2:
        drift["fit_observations"] = "all_directions"
    elif ref_all:
        drift["fit_observations"] = "insufficient"
    else:
        drift["fit_observations"] = "none"
    t0 = drift["t0_epoch"]
    slope = drift["slope_mm_per_s"]

    def drift_corr(r: dict) -> float:
        return float(r["value_mm"]) - slope * (float(r["epoch"]) - t0)

    # 2) 常规测位漂移校正，按方向/区分组
    groups: list[dict] = [
        {"forward": [], "reverse": []} for _ in range(zone_count)
    ]
    interim = []  # (seq, zone, direction, drift_corrected, locked)
    for slot in plan:
        if slot["kind"] != "reading":
            continue
        r = current.get(slot["seq"])
        if r is None:
            continue
        v = drift_corr(r)
        groups[slot["zone_index"]][slot["direction"]].append(v)
        interim.append((slot["seq"], slot["zone_index"], slot["direction"], v,
                        bool(r.get("locked"))))

    # 3) 回程间隙：各区反向均值 − 正向均值，全局取中位数（对单区异常稳健）
    per_zone_backlash = []
    diffs = []
    for zi, g in enumerate(groups):
        fvals = g[FORWARD]
        rvals = g[REVERSE]
        if fvals and rvals:
            diff = float(np.mean(rvals) - np.mean(fvals))
            diffs.append(diff)
        else:
            diff = None
        per_zone_backlash.append(
            {
                "zone_index": zi,
                "forward_mean": float(np.mean(fvals)) if fvals else None,
                "reverse_mean": float(np.mean(rvals)) if rvals else None,
                "diff_mm": diff,
            }
        )
    backlash_mm = float(np.median(diffs)) if diffs else 0.0

    # 4) 反向读数扣除全局间隙，汇总每区结果
    corrected_rows = []
    zone_rows = []
    for zi, g in enumerate(groups):
        entries = {"forward": [], "reverse": []}
        for seq, z, direction, v_drift, locked in interim:
            if z != zi:
                continue
            v = v_drift - (backlash_mm if direction == REVERSE else 0.0)
            entries[direction].append((seq, v, locked))
            corrected_rows.append(
                {"seq": seq, "zone_index": zi, "direction": direction,
                 "corrected_mm": v, "locked": locked}
            )
        fwd = [e[1] for e in entries["forward"]]
        rev = [e[1] for e in entries["reverse"]]
        st = _stats(fwd + rev)
        row = {
            "zone_index": zi,
            "n": st["n"],
            "n_forward": len(fwd),
            "n_reverse": len(rev),
            "forward": [
                {"seq": s, "corrected_mm": v, "locked": lk}
                for s, v, lk in sorted(entries["forward"])
            ],
            "reverse": [
                {"seq": s, "corrected_mm": v, "locked": lk}
                for s, v, lk in sorted(entries["reverse"])
            ],
            "forward_mean": float(np.mean(fwd)) if fwd else None,
            "reverse_mean": float(np.mean(rev)) if rev else None,
            "direction_diff_mm": (
                float(np.mean(rev) - np.mean(fwd)) if fwd and rev else None
            ),
            "mean": st["mean"],
            "std": st["std"],
            "range_mm": st["range"],
            "min": st["min"],
            "max": st["max"],
        }
        zone_rows.append(row)
    corrected_rows.sort(key=lambda r: r["seq"])

    # 5) 完整性与阈值违例（锁定读数全部参与时该违例被抑制）
    missing_seqs = [
        s["seq"] for s in plan if current.get(s["seq"]) is None
    ]
    violations: list[dict] = []

    def _seqs_str(seqs: list[int]) -> str:
        return "、".join(f"#{n}" for n in seqs)

    if missing_seqs:
        seqs = ", ".join(f"#{n}" for n in missing_seqs)
        violations.append(
            {
                "type": "missing",
                "seqs": missing_seqs,
                "message": f"漏测：测次 {seqs} 尚无有效读数",
                "locked_suppressed": False,
            }
        )

    # 防御：起始方向上参考点不足 2 个则无法拟合线性漂移（正常计划每扫掠至少
    # 一次复测，加上锚点必然满足；此违例保护旧数据/手工构造的计划）
    if len(preferred) < 2:
        ref_seqs = sorted(o["seq"] for o in ref_all if o["direction"] == fit_direction)
        if ref_seqs:
            detail = f"（起始方向 {fit_direction} 仅有参考点 {_seqs_str(ref_seqs)}）"
        else:
            detail = f"（起始方向 {fit_direction} 无任何参考观测）"
        violations.append(
            {
                "type": "drift_underconstrained",
                "seqs": ref_seqs,
                "message": (
                    "穿插参考点不足，无法拟合零点线性漂移：至少需要起始方向上"
                    "2 个参考观测（锚点 + 一次复测）" + detail
                ),
                "locked_suppressed": False,
            }
        )

    disp_lim = thresholds.get("max_dispersion_mm")
    dir_lim = thresholds.get("max_direction_diff_mm")
    drift_lim = thresholds.get("max_drift_residual_mm")

    for row in zone_rows:
        zi = row["zone_index"]
        fseq = [e["seq"] for e in row["forward"]]
        rseq = [e["seq"] for e in row["reverse"]]
        seqs = fseq + rseq
        locks = {e["seq"]: e["locked"] for e in row["forward"] + row["reverse"]}
        all_locked = bool(seqs) and all(locks.values())
        if disp_lim is not None and row["n"] >= 2 and row["range_mm"] > disp_lim:
            violations.append(
                {
                    "type": "dispersion",
                    "zone_index": zi,
                    "seqs": seqs,
                    "value_mm": row["range_mm"],
                    "threshold_mm": disp_lim,
                    "message": (
                        f"分区 {zi} 测次 {_seqs_str(seqs)} 校正读数极差 "
                        f"{row['range_mm']:.6g} mm 超过离散度阈值 {disp_lim:g} mm"
                    ),
                    "locked_suppressed": all_locked,
                }
            )
        d = row["direction_diff_mm"]
        if dir_lim is not None and d is not None and abs(d) > dir_lim:
            violations.append(
                {
                    "type": "direction_diff",
                    "zone_index": zi,
                    "seqs": seqs,
                    "value_mm": d,
                    "threshold_mm": dir_lim,
                    "message": (
                        f"分区 {zi} 测次 {_seqs_str(seqs)} 正反向均值差 "
                        f"{d:.6g} mm 超过回程间隙阈值 {dir_lim:g} mm"
                    ),
                    "locked_suppressed": all_locked,
                }
            )

    locked_by_seq = {s["seq"]: bool(s.get("locked")) for s in ref_all}
    for item in drift["residuals"]:
        seq = item["seq"]
        resid = item["residual_mm"]
        if drift_lim is not None and abs(resid) > drift_lim:
            violations.append(
                {
                    "type": "drift_residual",
                    "seqs": [seq],
                    "value_mm": resid,
                    "threshold_mm": drift_lim,
                    "message": (
                        f"参考复测测次 #{seq} 漂移残差 {resid:.6g} mm "
                        f"超过阈值 {drift_lim:g} mm"
                    ),
                    "locked_suppressed": bool(locked_by_seq.get(seq)),
                }
            )

    blocking = [v for v in violations if not v["locked_suppressed"]]
    return {
        "n_slots": len(plan),
        "n_filled": len(plan) - len(missing_seqs),
        "missing_seqs": missing_seqs,
        "drift": drift,
        "backlash": {"global_mm": backlash_mm, "per_zone": per_zone_backlash},
        "zones": zone_rows,
        "corrected_readings": corrected_rows,
        "violations": violations,
        "blocking_violations": blocking,
        "can_finalize": not blocking,
    }
