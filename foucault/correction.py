"""分区修正量（磨除量）约束搜索。

给定某分析版本在各区等效半径处的残余面形误差 e_i (nm)，求每区磨除深度
c_i (nm，正值 = 去除玻璃)，使修正后的残余波前误差最小。约束：
- 0 <= c_i <= max_removal_nm（每区最大磨除量）；
- 边缘区（最外区）可单独限制 edge_zone_max_removal_nm，或 preserve_edge=True
  强制边缘区磨除量为 0（边缘保留，避免修出塌边）；
- 可选面积加权平均磨除深度上限 max_mean_removal_nm（罚函数法处理）。

目标函数：f(c) = 1/2 ||P (e - c)||^2 + λ/2 ||D c||^2
其中 P 为"拟合后取残差"的投影（默认扣除活塞与离焦 {1, u^2}），D 为二阶
差分算子（修正量的平滑性）。对一组平滑权重 λ 各解一个盒约束 QP
（投影梯度 + FISTA 加速），连同直接截断基线一起，按
(剩余波前 RMS, 修正平滑度, 材料去除量) 升序排序输出候选方案。
"""
from __future__ import annotations

import math

import numpy as np


class CorrectionError(Exception):
    """修正搜索前置条件不满足。"""


def _diff_matrix(n: int) -> np.ndarray:
    """二阶差分矩阵；分区太少时退化为一阶差分。"""
    if n < 2:
        return np.zeros((0, n))
    if n == 2:
        return np.array([[1.0, -1.0]])
    D = np.zeros((n - 2, n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


def _residual_projector(u: np.ndarray, refit_defocus: bool) -> np.ndarray:
    """投影矩阵 M：对 {1, u^2}（或仅 {1}）拟合后取残差。对称幂等。"""
    n = u.size
    if refit_defocus:
        B = np.column_stack([np.ones(n), u * u])
    else:
        B = np.ones((n, 1))
    return np.eye(n) - B @ np.linalg.pinv(B)


def _solve_qp(
    M: np.ndarray,
    e: np.ndarray,
    DtD: np.ndarray,
    lam: float,
    lb: np.ndarray,
    ub: np.ndarray,
    w: np.ndarray | None,
    b: float,
    pen: float,
    iterations: int = 4000,
    tol: float = 1e-10,
) -> np.ndarray:
    """投影梯度 + FISTA 解盒约束二次规划（带可选平均深度罚项）。"""
    n = e.size
    L = 1.0  # ||M||_2 = 1（投影）
    if lam > 0 and DtD.size:
        L += lam * float(np.linalg.eigvalsh(DtD).max())
    if pen > 0 and w is not None:
        L += pen * float(w @ w)
    c = np.clip(e.copy(), lb, ub)
    y = c.copy()
    t = 1.0
    for _ in range(iterations):
        g = M @ (y - e)
        if lam > 0 and DtD.size:
            g = g + lam * (DtD @ y)
        if pen > 0 and w is not None:
            excess = float(w @ y) - b
            if excess > 0:
                g = g + pen * excess * w
        c_new = np.clip(y - g / L, lb, ub)
        t_new = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = c_new + ((t - 1.0) / t_new) * (c_new - c)
        step = float(np.max(np.abs(c_new - c))) if n else 0.0
        c = c_new
        t = t_new
        if step < tol:
            break
    return c


def _evaluate(
    c: np.ndarray,
    e: np.ndarray,
    M: np.ndarray,
    D: np.ndarray,
    areas: np.ndarray,
    wavelength_nm: float,
) -> dict:
    """候选方案指标：剩余波前、修正平滑度、材料去除量。

    剩余波前 RMS 按分区面积加权（圆形口径面积元 ∝ 半径），与分析报告口径一致。
    """
    resid = M @ (e - c)
    w = areas / areas.sum()
    m = float(np.sum(w * resid))
    rms_nm = float(np.sqrt(np.sum(w * (resid - m) ** 2)))
    pv_nm = float(resid.max() - resid.min())
    sd = D @ c
    smooth = float(np.sqrt(np.mean(sd**2))) if sd.size else 0.0
    volume_mm3 = float(np.sum(c * 1e-6 * areas))  # nm→mm × mm²
    return {
        "residual_wavefront_rms_nm": 2.0 * rms_nm,
        "residual_wavefront_pv_nm": 2.0 * pv_nm,
        "residual_wavefront_rms_waves": 2.0 * rms_nm / wavelength_nm,
        "residual_wavefront_pv_waves": 2.0 * pv_nm / wavelength_nm,
        "smoothness_nm": smooth,
        "removal_volume_mm3": volume_mm3,
        "mean_removal_nm": float(np.sum(c * areas) / np.sum(areas)),
        "max_removal_nm": float(c.max()) if c.size else 0.0,
    }


def search_corrections(
    r_mean: list[float],
    inner: list[float],
    outer: list[float],
    residual_surface_nm: list[float],
    wavelength_nm: float,
    diameter: float,
    *,
    max_removal_nm: float,
    edge_zone_max_removal_nm: float | None = None,
    preserve_edge: bool = False,
    max_mean_removal_nm: float | None = None,
    smoothing_weights: list[float] | None = None,
    refit_defocus: bool = True,
) -> dict:
    """搜索分区修正量并返回排序后的候选方案。"""
    e = np.asarray(residual_surface_nm, dtype=float)
    n = e.size
    if n < 2:
        raise CorrectionError("至少需要两个分区才能搜索修正量")
    if max_removal_nm <= 0:
        raise CorrectionError("最大磨除量必须为正数")
    r_m = np.asarray(r_mean, dtype=float)
    rim = diameter / 2.0
    areas = math.pi * (np.asarray(outer) ** 2 - np.asarray(inner) ** 2)

    ub = np.full(n, float(max_removal_nm))
    if edge_zone_max_removal_nm is not None:
        if edge_zone_max_removal_nm < 0:
            raise CorrectionError("边缘区磨除量上限不能为负")
        ub[-1] = min(ub[-1], float(edge_zone_max_removal_nm))
    if preserve_edge:
        ub[-1] = 0.0
    lb = np.zeros(n)

    w_mean = None
    b_mean = 0.0
    if max_mean_removal_nm is not None:
        if max_mean_removal_nm <= 0:
            raise CorrectionError("平均磨除深度上限必须为正数")
        w_mean = areas / areas.sum()
        b_mean = float(max_mean_removal_nm)

    u = r_m / rim
    M = _residual_projector(u, refit_defocus)
    D = _diff_matrix(n)
    DtD = D.T @ D

    weights = (
        [float(x) for x in smoothing_weights]
        if smoothing_weights
        else [0.0, 0.01, 0.1, 1.0, 10.0]
    )

    candidates = []
    # 基线：直接截断（磨掉所有凸起，受约束截断）
    c0 = np.clip(e, lb, ub)
    candidates.append(("direct_clip", None, c0))
    for lam in weights:
        if lam < 0:
            raise CorrectionError("平滑权重不能为负")
        pen = 0.0
        c = np.full(n, 0.0)
        for _ in range(4):  # 平均深度约束的罚系数自适应
            c = _solve_qp(M, e, DtD, lam, lb, ub, w_mean, b_mean, pen)
            if w_mean is None or float(w_mean @ c) <= b_mean + 1e-9:
                break
            pen = 1e3 if pen == 0.0 else pen * 100.0
        candidates.append((f"smooth_{lam:g}", lam, c))

    # 评估 + 排序：剩余波前 RMS → 修正平滑度 → 材料去除量
    ranked = []
    for label, lam, c in candidates:
        metrics = _evaluate(c, e, M, D, areas, wavelength_nm)
        ranked.append(
            {
                "label": label,
                "smoothing_weight": lam,
                "corrections_nm": [float(v) for v in c],
                "metrics": metrics,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["metrics"]["residual_wavefront_rms_waves"],
            item["metrics"]["smoothness_nm"],
            item["metrics"]["removal_volume_mm3"],
        )
    )
    for rank, item in enumerate(ranked):
        item["rank"] = rank

    return {
        "constraints": {
            "max_removal_nm": float(max_removal_nm),
            "edge_zone_max_removal_nm": (
                None if edge_zone_max_removal_nm is None else float(edge_zone_max_removal_nm)
            ),
            "preserve_edge": bool(preserve_edge),
            "max_mean_removal_nm": (
                None if max_mean_removal_nm is None else float(max_mean_removal_nm)
            ),
            "refit_defocus": bool(refit_defocus),
        },
        "zone_upper_bounds_nm": [float(v) for v in ub],
        "candidates": ranked,
    }
