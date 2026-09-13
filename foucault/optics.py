"""Foucault 刀口仪数据还原的光学计算核心。

约定与公式（推导见 README）：
- 纵向像差 LA 以"固定光源等效"表示：固定光源时刀口位移即 LA；
  移动光源（光源随刀口一起移动）时刀口位移为真实 LA 的一半，读数差需乘以 2。
- 目标圆锥（圆锥常数 K）的理想纵向像差 LA_ideal(r) = -K * r^2 / R。
- 面形斜率误差 alpha(r) = -LA_err(r) * r / (2 R^2)，
  面形误差 h(r) = ∫0^r alpha dr（实际面相对目标圆锥，正值为"凸起"）。
- 波前误差 W = 2h（反射加倍）。
- 最佳拟合：对 h 拟合 {1, r^2, r^4} 基，r^4 系数换算圆锥常数修正
  ΔK = 8 R^3 a4，r^2 系数换算曲率半径修正 ΔR = -2 R^2 a2。
- 横向像差 TA(r) = LA_resid(r) * r / R（最佳焦点面处）。
- Strehl ≈ exp(-(2π σ_W / λ)^2)（Maréchal 近似）。

所有长度量在计算内部一律使用 mm（波前/面形结果另以 nm 与波长数给出）。
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np

# 光源模式 → 刀口读数差换算为纵向像差的倍率
SOURCE_FACTOR = {"fixed": 1.0, "moving": 2.0}
# 声明单位 → mm
UNIT_TO_MM = {"mm": 1.0, "in": 25.4}
NM_PER_MM = 1.0e6
ARCSEC_PER_RAD = 206264.806247


class AnalysisError(Exception):
    """分析前置条件不满足（如有效读数不足、分区非法）。"""


@dataclass
class ZoneInput:
    """单个 Couder 遮罩分区（mm）。"""

    inner: float
    outer: float
    readings: list[float]  # 有效读数（mm，已换算、已剔除异常值）

    @property
    def r_mean(self) -> float:
        """分区等效半径：面积均分半径 sqrt((r_in^2 + r_out^2)/2)。"""
        return math.sqrt((self.inner**2 + self.outer**2) / 2.0)


@dataclass
class Constants:
    diameter: float  # mm
    radius_of_curvature: float  # mm
    conic_constant: float
    wavelength_nm: float
    source_mode: str  # "fixed" | "moving"
    instrument_offset: float = 0.0  # mm，刀口测微器零位偏移


@dataclass
class Options:
    min_readings_per_zone: int = 2
    error_band_waves: float = 0.25
    grid_points: int = 400
    fit_conic: bool = True
    fit_defocus: bool = True
    reference: str = "innermost"  # 零点参考："innermost" | "mean"


def ideal_la(r, conic_constant: float, radius_of_curvature: float):
    """目标圆锥的理想纵向像差 LA_ideal(r) = -K · r² / R（固定光源等效，mm）。

    接受标量或 numpy 数组；遮罩设计与读数还原共用同一公式。
    """
    return -conic_constant * r**2 / radius_of_curvature


def compute_input_hash(payload: dict) -> str:
    """对规范化后的输入（常量 + 有效读数 + 算法选项）计算稳定哈希。"""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _interp_extrap(r: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """分段线性插值；两端超出采样点时用端侧两点的斜率线性外推。"""
    y = np.interp(r, xs, ys).copy()
    left = r < xs[0]
    if np.any(left):
        slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
        y[left] = ys[0] + slope * (r[left] - xs[0])
    right = r > xs[-1]
    if np.any(right):
        slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        y[right] = ys[-1] + slope * (r[right] - xs[-1])
    return y


def _cumulative_trapezoid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """累积梯形积分，返回与 x 等长的数组（首元素为 0）。"""
    dx = np.diff(x)
    avg = (y[:-1] + y[1:]) / 2.0
    return np.concatenate([[0.0], np.cumsum(dx * avg)])


def zone_statistics(readings: list[float]) -> dict:
    """单区读数离散度汇总。"""
    arr = np.asarray(readings, dtype=float)
    n = int(arr.size)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "range": float(arr.max() - arr.min()),
        "sem": std / math.sqrt(n) if n else 0.0,
    }


def reduce_test(constants: Constants, zones: list[ZoneInput], options: Options) -> dict:
    """执行完整还原：零点/偏移校正 → 离散度 → LA → 面形积分 → 拟合 → 指标。

    返回可 JSON 序列化的结果字典（所有数值为 Python float/int）。
    """
    if len(zones) < 2:
        raise AnalysisError("至少需要两个分区才能积分面形")
    R = constants.radius_of_curvature
    K = constants.conic_constant
    rim = constants.diameter / 2.0
    lam = constants.wavelength_nm
    if R <= 0 or rim <= 0 or lam <= 0:
        raise AnalysisError("口径、曲率半径、波长必须为正数")
    try:
        sf = SOURCE_FACTOR[constants.source_mode]
    except KeyError:
        raise AnalysisError(f"未知光源模式: {constants.source_mode!r}") from None

    # 1) 每区统计 + 有效读数检查
    stats = []
    for i, z in enumerate(zones):
        vals = [float(v) for v in z.readings]
        if len(vals) < options.min_readings_per_zone:
            raise AnalysisError(
                f"分区 {i} 有效读数不足：{len(vals)} < {options.min_readings_per_zone}"
            )
        st = zone_statistics(vals)
        st["index"] = i
        st["inner"] = z.inner
        st["outer"] = z.outer
        st["r_mean"] = z.r_mean
        stats.append(st)

    # 2) 仪器偏移校正（从每区均值扣除零位偏移）
    for st in stats:
        st["mean_corrected"] = st["mean"] - constants.instrument_offset

    # 3) 零点校正：LA 相对参考零点（最内区或全区均值）
    if options.reference == "innermost":
        ref = min(stats, key=lambda s: s["inner"])["mean_corrected"]
    else:
        ref = float(np.mean([s["mean_corrected"] for s in stats]))
    for st in stats:
        st["la"] = (st["mean_corrected"] - ref) * sf

    r_m = np.array([st["r_mean"] for st in stats])
    la = np.array([st["la"] for st in stats])

    # 4) 相对目标圆锥的纵向像差误差
    la_ideal = ideal_la(r_m, K, R)
    la_err = la - la_ideal

    # 5) 积分出面形轮廓 h(r)：alpha = -LA_err * r / (2 R^2)
    #    LA_err 曲线向中心/边缘按端侧两点斜率外推（零点差分后理想圆锥对应
    #    常数 LA_err，外推保持该常数，不产生虚假剖面）。
    g = np.linspace(0.0, rim, options.grid_points)
    la_g = _interp_extrap(g, r_m, la_err)
    alpha = -la_g * g / (2.0 * R * R)
    h_mm = _cumulative_trapezoid(alpha, g)
    h_nm = h_mm * NM_PER_MM

    # 6) 最佳拟合 {1, u^2, u^4}（u 为归一化半径，改善数值条件）
    u = g / rim
    cols = [np.ones_like(u)]
    names = ["piston"]
    if options.fit_defocus:
        cols.append(u * u)
        names.append("defocus")
    if options.fit_conic:
        cols.append(u**4)
        names.append("conic")
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, h_nm, rcond=None)
    fit_nm = A @ coef
    e_nm = h_nm - fit_nm  # 残余面形误差（nm）

    coef_map = dict(zip(names, (float(c) for c in coef)))
    # 系数换算：h[mm] = a2 * r^2 + a4 * r^4
    a2_mm = coef_map.get("defocus", 0.0) / NM_PER_MM / rim**2
    a4_mm = coef_map.get("conic", 0.0) / NM_PER_MM / rim**4
    delta_roc = -2.0 * R * R * a2_mm  # mm
    delta_k = 8.0 * R**3 * a4_mm  # 无量纲

    # 7) 波前误差与汇总指标
    #    RMS 按圆形口径面积加权：面元 dA = 2πr dr，均匀网格 dr 恒定 → 权重 ∝ r。
    #    均匀半径统计会低估边缘大误差，使 RMS 系统性偏低、Strehl 偏高。
    w_nm = 2.0 * e_nm
    w_area = g  # 权重正比于半径（dr 均匀，2π 为常数可约去）
    w_sum = float(w_area.sum())

    def _area_weighted_std(x: np.ndarray) -> float:
        m = float((w_area * x).sum() / w_sum)
        return float(math.sqrt(float((w_area * (x - m) ** 2).sum()) / w_sum))

    surf_pv = float(e_nm.max() - e_nm.min())
    surf_rms = _area_weighted_std(e_nm)
    wf_pv_nm = float(w_nm.max() - w_nm.min())
    wf_rms_nm = _area_weighted_std(w_nm)
    strehl = float(math.exp(-((2.0 * math.pi * wf_rms_nm / lam) ** 2)))

    # 8) 分区处指标：残余波前、横向像差、误差带标记
    e_zone = np.interp(r_m, g, e_nm)
    w_zone = 2.0 * e_zone
    la_fit = 2.0 * delta_roc - delta_k * r_m**2 / R
    la_resid = la_err - la_fit
    ta_mm = la_resid * r_m / R
    ta_arcsec = ta_mm / R * ARCSEC_PER_RAD
    band_nm = options.error_band_waves * lam
    out_flags = (np.abs(w_zone) > band_nm).tolist()

    zone_rows = []
    for i, st in enumerate(stats):
        zone_rows.append(
            {
                "index": i,
                "inner_radius": st["inner"],
                "outer_radius": st["outer"],
                "r_mean": st["r_mean"],
                "n_readings": st["n"],
                "readings_used": [float(v) for v in zones[i].readings],
                "mean": st["mean"],
                "mean_corrected": st["mean_corrected"],
                "std": st["std"],
                "min": st["min"],
                "max": st["max"],
                "range": st["range"],
                "sem": st["sem"],
                "la": float(la[i]),
                "la_ideal": float(la_ideal[i]),
                "la_error": float(la_err[i]),
                "la_residual": float(la_resid[i]),
                "transverse_aberration_mm": float(ta_mm[i]),
                "transverse_aberration_arcsec": float(ta_arcsec[i]),
                "wavefront_nm": float(w_zone[i]),
                "wavefront_waves": float(w_zone[i] / lam),
                "out_of_band": bool(out_flags[i]),
            }
        )

    out_of_band_zones = [i for i, f in enumerate(out_flags) if f]
    disp_std = [st["std"] for st in stats]
    disp_range = [st["range"] for st in stats]

    result = {
        "zones": zone_rows,
        "profile": {
            "r_mm": [float(v) for v in g],
            "surface_nm": [float(v) for v in h_nm],
            "residual_surface_nm": [float(v) for v in e_nm],
            "wavefront_nm": [float(v) for v in w_nm],
        },
        "fit": {
            "terms": names,
            "piston_nm": coef_map.get("piston", 0.0),
            "defocus_nm": coef_map.get("defocus", 0.0),
            "conic_nm": coef_map.get("conic", 0.0),
            "delta_k": float(delta_k),
            "best_fit_conic_constant": float(K + delta_k),
            "delta_radius_of_curvature_mm": float(delta_roc),
            "best_fit_radius_of_curvature_mm": float(R + delta_roc),
        },
        "summary": {
            "surface_pv_nm": surf_pv,
            "surface_rms_nm": surf_rms,
            "wavefront_pv_nm": wf_pv_nm,
            "wavefront_pv_waves": wf_pv_nm / lam,
            "wavefront_rms_nm": wf_rms_nm,
            "wavefront_rms_waves": wf_rms_nm / lam,
            "strehl": strehl,
            "n_zones": len(zones),
            "reference": options.reference,
            "source_factor": sf,
        },
        "dispersion": {
            "mean_std": float(np.mean(disp_std)),
            "max_std": float(np.max(disp_std)),
            "max_range": float(np.max(disp_range)),
        },
        "error_band": {
            "waves": options.error_band_waves,
            "nm": float(band_nm),
            "out_of_band_zones": out_of_band_zones,
        },
    }
    return result
