"""子午线复测研究的二次角向谐波拟合核心。

同一面镜在不同镜面旋转角 ψ、刀口扫描直径方位 α 下复测，每个半径分区的
纵向像差（LA）残差可分解为两类二次（2θ）像散：

- **镜面分量**（随镜面旋转）：在镜面坐标中方向固定，随 ψ 一起转动，
  是磨制/支撑留在镜坯上的面形象散；
- **架位分量**（固定在测试架方向）：在实验室坐标中方向固定，不随 ψ 转动，
  来自刀口仪/空气层/支架方向等系统误差。

对每个半径 r，用 N 份来源的 LA 残差 y_i 拟合

    y_i = c(r) + A_m·cos 2(ψ_i − φ_m) + A_l·cos 2(α_i − φ_l)
         = c + mc·cos2ψ + ms·sin2ψ + lc·cos2α + ls·sin2α

    A_m = hypot(mc, ms)，φ_m = atan2(ms, mc)/2（镜面坐标主轴）
    A_l = hypot(lc, ls)，φ_l = atan2(ls, lc)/2（架位坐标主轴）

角度约定（创建研究时冻结）：均从镜面光学面（刀口侧）观察，0° 为参考刻线
/ 参考水平方向，逆时针为正，单位为度（°）；直径方位与像散主轴以 180°
为周期，内部按 cos2/sin2 归算。

可识别性：
- 刀口方位 α 固定、只旋转镜面时，架位项在各来源间为常数（被截距吸收），
  只能识别镜面分量；反之镜面不旋转时只能识别架位分量。要分离两者，ψ 与 α
  都须有非共线的角度散布。矩阵欠秩时不强行给出伪结果，而是逐分量指出缺口
  与补测建议（MeridianError.gaps）。
- 截距 c（轴对称残余）在被谐波列张成（如 N=4 的一般角度）时自动剔除，
  此时四个谐波系数仍可识别。

LA → 波前：面形 h(r,θ)=H(r)cos2(θ−φ)，子午斜率关系
LA = −2R²/r·∂h/∂r ⇒ H′(r) = −r·A_LA(r)/(2R²)，从中心积分到口径边缘，
波前 W = 2h（反射加倍）。

主轴方向：逐区主轴直接由 LA 谐波系数 atan2/2 给出（镜面/架位各自坐标）。
LA→面形积分含一次负号，使整体波前主轴相对 LA 主轴存在 90° 象限二义性
（二次像散 cos2θ 的固有性质）；整体主轴取逐区 LA 主轴按环带面积的圆形
平均，与逐区报告保持同一约定，积分只决定幅值与 RMS。
"""
from __future__ import annotations

import math

import numpy as np

NM_PER_MM = 1.0e6

# 模型列：0 截距 / 1,2 镜面 cos2ψ,sin2ψ / 3,4 架位 cos2α,sin2α
_COL_INTERCEPT = 0
_COL_MIRROR = (1, 2)
_COL_LAB = (3, 4)
_ALL_COMPONENTS = (("mirror", _COL_MIRROR), ("lab_fixed", _COL_LAB))

ANGLE_CONVENTION = {
    "view": "从镜面光学面（刀口侧）观察",
    "zero": "0° = 镜面参考刻线 / 测试架参考水平方向",
    "positive": "逆时针为正",
    "unit": "degree",
    "period_deg": 180,
}


class MeridianError(Exception):
    """研究前置条件不满足（来源不一致、角度欠秩、有效来源不足等）。"""

    def __init__(self, message: str, gaps: list[str] | None = None):
        super().__init__(message)
        self.gaps = gaps or []


# ---------------- 统计辅助 ----------------


def _norm_rank(A: np.ndarray) -> int:
    """np.linalg.matrix_rank 的显式容差版本（与最小二乘判定一致）。"""
    s = np.linalg.svd(A, compute_uv=False)
    if s.size == 0 or s[0] <= 0:
        return 0
    tol = max(A.shape) * np.finfo(float).eps * s[0]
    return int(np.count_nonzero(s > tol))


def _t_quantile(p: float, dof: float) -> float:
    """学生 t 分位数（无 scipy）：Acklam 正态逆 + Cornish–Fisher 展开。"""
    # Peter Acklam 逆正态 CDF 近似系数
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    elif p <= phigh:
        q = p - 0.5
        r = q * q
        z = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    else:
        q = math.sqrt(-2 * math.log(1 - p))
        z = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if dof <= 0 or not math.isfinite(dof):
        return z
    nu = dof
    # Cornish–Fisher 项（到 ν^-3），少量自由度也足够稳健
    t = z + (z**3 + z) / (4 * nu) \
        + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * nu**2) \
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z) / (384 * nu**3)
    return t


def _axis_deg(x: float, y: float) -> float:
    """cos2/sin2 系数 → 主轴角度，归一化到 [0, 180)。"""
    deg = math.degrees(math.atan2(y, x) / 2.0)
    return deg % 180.0


def _axis_shift_deg(a: float, b: float) -> float:
    """两主轴（180° 周期、cos2 下正负 90° 等价）间的锐夹角，单位度。"""
    d = (a - b) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return d


def design_matrix(mirror_rot_deg, knife_az_deg) -> np.ndarray:
    """5 列设计矩阵 [1, cos2ψ, sin2ψ, cos2α, sin2α]。"""
    psi = np.radians(np.asarray(mirror_rot_deg, dtype=float))
    alpha = np.radians(np.asarray(knife_az_deg, dtype=float))
    return np.column_stack(
        [
            np.ones_like(psi),
            np.cos(2 * psi),
            np.sin(2 * psi),
            np.cos(2 * alpha),
            np.sin(2 * alpha),
        ]
    )


def _angle_gaps(psi: np.ndarray, alpha: np.ndarray, essential: list[bool]) -> list[str]:
    """把不可识别列翻译成面向操作者的角度缺口说明。"""
    gaps: list[str] = []

    def _noncollinear(angles: np.ndarray) -> bool:
        doubled = 2.0 * np.radians(angles)
        v = np.column_stack([np.cos(doubled), np.sin(doubled)])
        return _norm_rank(v) >= 2

    if not (essential[_COL_MIRROR[0]] and essential[_COL_MIRROR[1]]):
        tip = "补充镜面旋转角不同的来源"
        if not _noncollinear(psi):
            tip = ("镜面旋转角只在 0°/90° 等共线方位取值，需补充旋转角相差"
                   "非 0°/90° 的来源")
        gaps.append(
            "镜面随转分量无法识别：" + tip + "（至少两个二次谐波下非共线的旋转角）"
        )
    if not (essential[_COL_LAB[0]] and essential[_COL_LAB[1]]):
        tip = "补充刀口扫描直径方位不同的来源"
        if not _noncollinear(alpha):
            tip = ("刀口扫描直径方位只在 0°/90° 等共线方位取值，需补充方位相差"
                   "非 0°/90° 的来源")
        elif _noncollinear(psi):
            tip = ("刀口扫描直径方位在各来源间无独立变化（与镜面旋转角共线），"
                   "需在镜面旋转的同时改变刀口扫描直径方位")
        gaps.append("架位固定分量无法识别：" + tip)
    return gaps


# ---------------- 波前换算 ----------------


def _zone_radial_weights(inner, outer, rim: float, R: float):
    """各分区 LA 幅值对边缘面形 H(rim) 的贡献权重（mm→mm）。

    LA 像差幅值在分区物理环带 [inner_j, outer_j] 内按该区测量值（等效半径
    r_j 处）代表；未被环带覆盖的中心孔（0..inner_0）按最内区幅值延拓，
    外缘（outer_last..rim）按最外区幅值延拓（与 optics 模块向中心/边缘
    外推的约定一致），使幅值恒为 A 时 Σw_j = −rim²/(4R²)，与解析解一致。
    w_j = −∫ρ dρ/(2R²)，各分区覆盖区间：
      首区 [0, (outer_0+inner_1)/2]（梯形分担），末区延伸到 rim，
      中间区在相邻等效半径中点处相接。
    """
    radii_ref = np.sqrt((np.asarray(inner) ** 2 + np.asarray(outer) ** 2) / 2.0)
    order = np.argsort(radii_ref)
    r = radii_ref[order]
    m = r.size
    # 分区之间的边界取相邻等效半径中点；首边界 0，末边界 rim
    edges = np.zeros(m + 1)
    edges[0] = 0.0
    for j in range(m - 1):
        edges[j + 1] = 0.5 * (r[j] + r[j + 1])
    edges[m] = rim
    w_sorted = -(edges[1:] ** 2 - edges[:-1] ** 2) / (4.0 * R * R)
    w = np.zeros(m)
    w[order] = w_sorted
    return w


def _radial_surface_profile(prof_c, prof_s, radii, inner_radii, outer_radii,
                            rim: float, R: float, n_grid: int = 600):
    """由逐区 LA 谐波系数重建口径上的面形复振幅 h(r)=hc+i hs（mm）。

    LA 幅值在每个分区物理环带内按该区测量值（等效半径 r_j 处）代表，
    相邻分区在等效半径中点相接，中心 0..首中点幅值为零（光轴处轴对称残差
    为零），末段延伸到口径边缘 rim。inner/outer 仅用于接口对称，分段以
    等效半径中点为准。h(r) 用细密半径网格数值积分 h′=−r·A/(2R²)。
    返回 (r_grid, hc, hs)。
    """
    radii = np.asarray(radii, dtype=float)
    order = np.argsort(radii)
    r = radii[order]
    cc = np.asarray(prof_c, dtype=float)[order]
    ss = np.asarray(prof_s, dtype=float)[order]
    m = r.size
    # 幅值在相邻等效半径的中点之间按该分区常数分段；首段从 0（光轴幅值为零，
    # 线性到首个中点），末段延伸到口径边缘 rim。
    edges = [0.0]
    for j in range(m - 1):
        edges.append(0.5 * (r[j] + r[j + 1]))
    edges.append(rim)

    def amp_at(rho):
        if rho <= 0:
            return 0.0, 0.0
        for j in range(m):
            lo, hi = edges[j], edges[j + 1]
            if lo < rho <= hi or (j == m - 1 and rho >= hi):
                return cc[j], ss[j]
        return cc[-1], ss[-1]

    g = np.linspace(0.0, rim, n_grid)
    hc = np.zeros_like(g)
    hs = np.zeros_like(g)
    for k in range(1, g.size):
        ra, rb = g[k - 1], g[k]
        ca, sa = amp_at(ra)
        cb, sb = amp_at(rb)
        # h = −1/(2R²)∫ρ·A dρ；梯形 ∫_ra^rb ρ·A dρ ≈ Δr(ra·A_a+rb·A_b)/2
        hc[k] = hc[k - 1] - (rb - ra) * (ra * ca + rb * cb) / (4.0 * R * R)
        hs[k] = hs[k - 1] - (rb - ra) * (ra * sa + rb * sb) / (4.0 * R * R)
    return g, hc, hs


def _integrate_astigmatism(profile_c, profile_s, radii, inner_radii, outer_radii,
                           rim: float, R: float):
    """逐半径 LA 谐波系数 → 边缘面形复振幅 H(rim) 与各分区线性权重。

    返回 (H_rim(complex), weights(M))。边缘幅值与权重用于置信区间传播；
    径向 RMS 由 _radial_surface_profile 在细密网格上计算。
    """
    radii = np.asarray(radii, dtype=float)
    order = np.argsort(radii)
    w = _zone_radial_weights(inner_radii, outer_radii, rim, R)
    c = np.asarray(profile_c, dtype=float)
    s = np.asarray(profile_s, dtype=float)
    H_rim = complex(np.sum(w * c) + 1j * np.sum(w * s))
    return H_rim, w


# ---------------- 主拟合 ----------------


def fit_harmonics(
    mirror_rot_deg,
    knife_az_deg,
    signals,
    radii,
    inner_radii,
    outer_radii,
    R: float,
    rim: float,
    wavelength_nm: float,
    confidence_level: float = 0.95,
) -> dict:
    """对全部半径拟合同一设计矩阵的二次角向谐波。

    signals: N×M（来源 × 半径）的 LA 残差（mm）。
    返回可 JSON 序列化的结果字典；角度组合欠秩时抛 MeridianError。
    """
    psi = np.asarray(mirror_rot_deg, dtype=float)
    alpha = np.asarray(knife_az_deg, dtype=float)
    Y = np.asarray(signals, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n, m = Y.shape
    if psi.shape[0] != n or alpha.shape[0] != n:
        raise MeridianError("角度序列长度与来源数不一致")
    if not (radii.shape[0] == m and len(inner_radii) == m and len(outer_radii) == m):
        raise MeridianError("半径序列长度与信号列数不一致")
    if n < 4:
        raise MeridianError(
            f"有效来源仅 {n} 份，无法分离两类二次像散（至少需要 4 份不同角度来源）",
            ["补充更多不同镜面旋转角 / 刀口方位的复测来源（研究档案 4~16 份）"],
        )

    A_full = design_matrix(psi, alpha)
    rank_full = _norm_rank(A_full)
    essential = [
        rank_full > _norm_rank(np.delete(A_full, j, axis=1))
        for j in range(A_full.shape[1])
    ]
    gaps = _angle_gaps(psi, alpha, essential)
    if gaps:
        raise MeridianError(
            "角度组合使二次谐波拟合矩阵欠秩，镜面随转分量与架位固定分量无法分离",
            gaps,
        )

    # 截距被谐波列张成时不单独拟合（否则与谐波共线）；两对象散分量均已识别。
    use_cols = [_COL_INTERCEPT] if essential[_COL_INTERCEPT] else []
    use_cols += list(_COL_MIRROR) + list(_COL_LAB)
    use_cols.sort()
    A = A_full[:, use_cols]
    p = A.shape[1]
    rank = _norm_rank(A)
    if rank < p:  # 理论上不会发生（缺口已拦截），兜底
        raise MeridianError("拟合矩阵欠秩", ["补充角度散布更大的复测来源"])

    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)  # p×M
    resid = Y - A @ coef
    dof_total = m * (n - p)
    sigma2 = float((resid**2).sum() / dof_total) if dof_total > 0 else None
    # 所有半径共用设计矩阵：Cov(β_r) = σ² (AᵀA)⁻¹（块相同）
    AtA_inv = np.linalg.inv(A.T @ A)
    tcrit = (
        _t_quantile(0.5 + confidence_level / 2.0, dof_total)
        if sigma2 is not None
        else None
    )

    def col_index(global_col: int) -> int | None:
        return use_cols.index(global_col) if global_col in use_cols else None

    i_mc, i_ms = col_index(1), col_index(2)
    i_lc, i_ls = col_index(3), col_index(4)
    i_c = col_index(0)

    def component_ci(bc: float, bs: float, cov2: np.ndarray):
        """cos/sin 系数 → (幅值, 幅值CI半宽, 主轴°, 主轴CI半宽°)。"""
        amp = math.hypot(bc, bs)
        if sigma2 is None or amp <= 0:
            return amp, None, _axis_deg(bc, bs), None
        ga = np.array([bc / amp, bs / amp])
        se_amp = math.sqrt(max(0.0, float(ga @ cov2 @ ga)))
        # θ = atan2(bs,bc)/2 的梯度
        gx = np.array([-bs / (2.0 * amp**2), bc / (2.0 * amp**2)])
        se_axis = math.degrees(math.sqrt(max(0.0, float(gx @ cov2 @ gx))))
        return amp, tcrit * se_amp, _axis_deg(bc, bs), tcrit * se_axis

    # 逐半径行
    zone_rows = []
    prof_mc = np.zeros(m)
    prof_ms = np.zeros(m)
    prof_lc = np.zeros(m)
    prof_ls = np.zeros(m)
    for j in range(m):
        cov = (sigma2 or 0.0) * AtA_inv
        mc, ms = float(coef[i_mc, j]), float(coef[i_ms, j])
        lc, ls = float(coef[i_lc, j]), float(coef[i_ls, j])
        cv_m = cov[np.ix_([i_mc, i_ms], [i_mc, i_ms])]
        cv_l = cov[np.ix_([i_lc, i_ls], [i_lc, i_ls])]
        a_m, ci_m, ax_m, ciax_m = component_ci(mc, ms, cv_m)
        a_l, ci_l, ax_l, ciax_l = component_ci(lc, ls, cv_l)
        prof_mc[j], prof_ms[j] = mc, ms
        prof_lc[j], prof_ls[j] = lc, ls
        c0 = float(coef[i_c, j]) if i_c is not None else 0.0
        col_res = resid[:, j]
        stud = col_res / math.sqrt(sigma2) if sigma2 else np.zeros(n)
        per_source = [
            {
                "source_index": i,
                "mirror_rotation_deg": float(psi[i]),
                "knife_diameter_azimuth_deg": float(alpha[i]),
                "observed_mm": float(Y[i, j]),
                "predicted_mm": float(Y[i, j] - col_res[i]),
                "residual_mm": float(col_res[i]),
                "studentized_residual": float(stud[i]) if sigma2 else None,
            }
            for i in range(n)
        ]
        zone_rows.append(
            {
                "index": j,
                "inner_radius": float(inner_radii[j]),
                "outer_radius": float(outer_radii[j]),
                "r_mean": float(radii[j]),
                "mirror": {
                    "amplitude_mm": a_m,
                    "amplitude_ci_mm": [max(0.0, a_m - ci_m), a_m + ci_m]
                    if ci_m is not None
                    else None,
                    "axis_deg": ax_m,
                    "axis_ci_deg": [
                        (_axis_deg(mc, ms) - ciax_m) % 180.0,
                        (_axis_deg(mc, ms) + ciax_m) % 180.0,
                    ]
                    if ciax_m is not None
                    else None,
                },
                "lab_fixed": {
                    "amplitude_mm": a_l,
                    "amplitude_ci_mm": [max(0.0, a_l - ci_l), a_l + ci_l]
                    if ci_l is not None
                    else None,
                    "axis_deg": ax_l,
                    "axis_ci_deg": [
                        (_axis_deg(lc, ls) - ciax_l) % 180.0,
                        (_axis_deg(lc, ls) + ciax_l) % 180.0,
                    ]
                    if ciax_l is not None
                    else None,
                },
                "intercept_mm": c0,
                "rss_residual_mm": float(math.sqrt(float((col_res**2).sum()))),
                "max_abs_residual_mm": float(np.max(np.abs(col_res))),
                "mean_abs_studentized_residual": float(np.mean(np.abs(stud)))
                if sigma2
                else None,
                "max_abs_studentized_residual": float(np.max(np.abs(stud)))
                if sigma2
                else None,
                "per_source": per_source,
            }
        )

    # 整体像散波前：按分区物理环带把 LA 幅值积分到口径边缘（镜面 / 架位
    # 分别积分，保留各自相位）。
    #
    # 注意：LA 剖面经 H′ = −r·A_LA/(2R²) 积分会整体引入一次符号翻转，使由
    # H 算出的角度与直接从 LA 谐波得到的主轴相差 90°（二次像散主轴本有 90°
    # 象限二义性）。为与各半径逐区报告一致，整体主轴取逐区 LA 主轴按环带
    # 面积权重 |w_j| 的圆形平均；积分只决定幅值/RMS（不敏感于符号）。
    def wavefront_block(prof_c, prof_s, zone_blocks, i_c_coef, i_s_coef, axis_name: str):
        H_rim, w = _integrate_astigmatism(
            prof_c, prof_s, radii, inner_radii, outer_radii, rim, R
        )
        w_amp_mm = float(np.abs(H_rim))
        w_amp_nm = 2.0 * w_amp_mm * NM_PER_MM
        cov2 = (
            sigma2 * AtA_inv[np.ix_([i_c_coef, i_s_coef], [i_c_coef, i_s_coef])]
            if sigma2 is not None
            else None
        )
        se_amp_nm = None
        if cov2 is not None and w_amp_mm > 0:
            denom = np.hypot(prof_c, prof_s)
            gcos = w * prof_c / denom
            gsin = w * prof_s / denom
            var_H = (
                float((gcos**2).sum()) * cov2[0, 0]
                + float((gsin**2).sum()) * cov2[1, 1]
                + 2.0 * float((gcos * gsin).sum()) * cov2[0, 1]
            )
            se_amp_nm = 2.0 * NM_PER_MM * math.sqrt(max(0.0, var_H))
        # 主轴：逐区 LA 主轴按环带面积权重 |w| 做二次谐波圆形平均
        q = np.abs(w)
        qsum = float(q.sum())
        zone_axes = np.asarray([z["axis_deg"] for z in zone_blocks])
        sx = float((q * np.cos(np.radians(2.0 * zone_axes))).sum())
        sy = float((q * np.sin(np.radians(2.0 * zone_axes))).sum())
        axis = _axis_deg(sx, sy) if qsum > 0 else 0.0
        se_axis_deg = None
        if cov2 is not None and qsum > 0:
            Qc = q / qsum
            var_sum = 0.0
            for j in range(m):
                A_j = math.hypot(prof_c[j], prof_s[j])
                if A_j <= 0:
                    continue
                cv, sv = prof_c[j] / A_j, prof_s[j] / A_j
                g = (Qc[j] / A_j) * np.array(
                    [[sv * sv, -cv * sv], [-cv * sv, cv * cv]]
                )
                var_sum += float(np.trace(g @ cov2 @ g.T))
            se_axis_deg = math.degrees(0.5 * math.sqrt(max(0.0, var_sum)))
        # 面积加权波前 RMS：在细密半径网格上重建 h(r)（运行积分）。径向重建
        # 只关心幅值轮廓（相位随半径小幅变化、逐区局部），故用各分区谐波
        # 幅值 hypot(prof_c,prof_s) 作为剖面；|h(r)| 按面元 ∝ r 加权，
        # 角向 <cos²2θ>=1/2。
        prof_amp = np.hypot(prof_c, prof_s)
        g_r, hc_g, hs_g = _radial_surface_profile(
            prof_amp, np.zeros_like(prof_amp), radii, inner_radii,
            outer_radii, rim, R
        )
        h_abs = np.hypot(hc_g, hs_g)
        # 均匀 dr 网格：面积加权均值 = Σ r·x / Σ r（dr 与 2π 约去）；
        # 波前 = 2·面形（反射加倍）。
        peak_mm2 = float((g_r * h_abs**2).sum()) / float(g_r.sum())
        rms_peak_nm = 2.0 * math.sqrt(peak_mm2) * NM_PER_MM
        rms_wavefront_nm = rms_peak_nm / math.sqrt(2.0)  # 角向 <cos²2θ>=1/2
        return {
            f"principal_axis_{axis_name}_deg": axis,
            "rim_wavefront_amplitude_nm": w_amp_nm,
            "rim_wavefront_amplitude_waves": w_amp_nm / wavelength_nm,
            "wavefront_pv_nm": 2.0 * w_amp_nm,
            "wavefront_pv_waves": 2.0 * w_amp_nm / wavelength_nm,
            "wavefront_rms_nm": rms_wavefront_nm,
            "wavefront_rms_waves": rms_wavefront_nm / wavelength_nm,
            "peak_profile_rms_nm": rms_peak_nm,
            "peak_profile_rms_waves": rms_peak_nm / wavelength_nm,
            "angular_rms_nm": rms_wavefront_nm,
            "angular_rms_waves": rms_wavefront_nm / wavelength_nm,
            "rim_amplitude_ci_nm": [
                max(0.0, w_amp_nm - tcrit * se_amp_nm),
                w_amp_nm + tcrit * se_amp_nm,
            ]
            if se_amp_nm is not None
            else None,
            "axis_ci_deg": [
                (axis - tcrit * se_axis_deg) % 180.0,
                (axis + tcrit * se_axis_deg) % 180.0,
            ]
            if se_axis_deg is not None
            else None,
        }

    ast_m = wavefront_block(prof_mc, prof_ms, [z["mirror"] for z in zone_rows],
                            i_mc, i_ms, "mirror")
    ast_l = wavefront_block(prof_lc, prof_ls, [z["lab_fixed"] for z in zone_rows],
                            i_lc, i_ls, "lab")

    # 最可疑区段：扣除全局二次像散后，单来源标准化残差最突出的半径（局部
    # 缺陷/某次上机异常，区别于随角度平滑变化的像散）。用最大绝对值而非均值，
    # 对“仅在一份来源上异常”的区段更敏感。
    if sigma2:
        suspicion = [
            (z["max_abs_studentized_residual"], z["max_abs_residual_mm"], z["index"])
            for z in zone_rows
        ]
    else:
        suspicion = [
            (z["rss_residual_mm"], z["max_abs_residual_mm"], z["index"])
            for z in zone_rows
        ]
    worst = max(suspicion)
    wz = zone_rows[worst[2]]
    suspicious = {
        "index": worst[2],
        "inner_radius": wz["inner_radius"],
        "outer_radius": wz["outer_radius"],
        "r_mean": wz["r_mean"],
        "max_abs_residual_mm": wz["max_abs_residual_mm"],
        "mean_abs_studentized_residual": wz["mean_abs_studentized_residual"],
        "max_abs_studentized_residual": wz["max_abs_studentized_residual"],
        "mirror_amplitude_mm": wz["mirror"]["amplitude_mm"],
        "reason": "全局二次像散拟合后单来源标准化残差最大的半径区段"
        if sigma2
        else "残差最大的半径区段（有效自由度为 0，无法标准化）",
    }

    return {
        "n_sources": n,
        "n_zones": m,
        "rank": rank,
        "model_rank": rank_full,
        "model_columns": ["intercept", "mirror_cos2psi", "mirror_sin2psi",
                          "lab_cos2alpha", "lab_sin2alpha"],
        "fitted_columns": [
            ["intercept", "mirror_cos2psi", "mirror_sin2psi",
             "lab_cos2alpha", "lab_sin2alpha"][c]
            for c in use_cols
        ],
        "residual_dof": dof_total,
        "residual_sigma_mm": math.sqrt(sigma2) if sigma2 is not None else None,
        "convention": ANGLE_CONVENTION,
        "radii_mm": [float(r) for r in radii],
        "zones": zone_rows,
        "astigmatism": {
            "wavelength_nm": float(wavelength_nm),
            "mirror": ast_m,
            "lab_fixed": ast_l,
            "total_wavefront_rms_waves": math.sqrt(
                ast_m["wavefront_rms_waves"] ** 2 + ast_l["wavefront_rms_waves"] ** 2
            ),
        },
        "most_suspicious_zone": suspicious,
        "loo": None,
    }


def leave_one_out(
    mirror_rot_deg,
    knife_az_deg,
    signals,
    radii,
    inner_radii,
    outer_radii,
    R: float,
    rim: float,
    wavelength_nm: float,
    confidence_level: float = 0.95,
    axis_stability_deg: float = 15.0,
) -> dict:
    """留一法：逐份剔除来源重解，比较镜面像散主轴/幅值与可疑区段是否稳定。

    不修改研究数据（排除原因由调用方通过排除接口另行记录）。
    """
    psi = np.asarray(mirror_rot_deg, dtype=float)
    alpha = np.asarray(knife_az_deg, dtype=float)
    Y = np.asarray(signals, dtype=float)
    n = Y.shape[0]
    full = fit_harmonics(
        psi, alpha, Y, radii, inner_radii, outer_radii, R, rim,
        wavelength_nm, confidence_level,
    )
    full_ast = full["astigmatism"]["mirror"]
    full_axis = full_ast["principal_axis_mirror_deg"]
    full_amp = full_ast["rim_wavefront_amplitude_nm"]
    full_ci = full_ast["rim_amplitude_ci_nm"]
    full_susp = full["most_suspicious_zone"]["index"]

    rows = []
    stable = True
    for k in range(n):
        keep = [i for i in range(n) if i != k]
        row = {
            "excluded_source_index": k,
            "mirror_rotation_deg": float(psi[k]),
            "knife_diameter_azimuth_deg": float(alpha[k]),
        }
        try:
            loo = fit_harmonics(
                psi[keep], alpha[keep], Y[keep], radii, inner_radii, outer_radii,
                R, rim, wavelength_nm, confidence_level,
            )
        except MeridianError as exc:
            row.update({"rank_deficient": True, "gaps": exc.gaps})
            stable = False
            rows.append(row)
            continue
        ast = loo["astigmatism"]["mirror"]
        axis_shift = _axis_shift_deg(full_axis, ast["principal_axis_mirror_deg"])
        amp_shift = abs(ast["rim_wavefront_amplitude_nm"] - full_amp)
        amp_within_ci = full_ci is not None and amp_shift <= (full_ci[1] - full_ci[0]) / 2.0
        susp_same = loo["most_suspicious_zone"]["index"] == full_susp
        k_stable = (
            axis_shift <= axis_stability_deg and amp_within_ci and susp_same
        )
        stable = stable and k_stable
        row.update(
            {
                "rank_deficient": False,
                "mirror_principal_axis_deg": ast["principal_axis_mirror_deg"],
                "axis_shift_deg": axis_shift,
                "rim_wavefront_amplitude_nm": ast["rim_wavefront_amplitude_nm"],
                "amplitude_delta_nm": amp_shift,
                "amplitude_within_full_ci": amp_within_ci,
                "most_suspicious_zone": loo["most_suspicious_zone"]["index"],
                "suspicious_zone_unchanged": susp_same,
                "stable": k_stable,
            }
        )
        rows.append(row)
    return {
        "stable": stable,
        "axis_stability_threshold_deg": axis_stability_deg,
        "full_mirror_principal_axis_deg": full_axis,
        "full_rim_wavefront_amplitude_nm": full_amp,
        "full_most_suspicious_zone": full_susp,
        "leave_one_out": rows,
    }


def extract_zone_signals(version_result: dict, field: str = "la_residual") -> list[float]:
    """从既有分析版本结果按分区顺序取出纵向像差信号（mm）。

    默认 la_residual（最佳拟合圆锥/离焦之后的 LA 残差，即角度相关面形象散）；
    也可取 la_error（未扣除最佳拟合圆锥的 LA 误差）。
    """
    if field not in ("la_residual", "la_error"):
        raise MeridianError(f"未知拟合信号字段：{field!r}（支持 la_residual / la_error）")
    zones = sorted(version_result.get("zones", []), key=lambda z: z["index"])
    if not zones:
        raise MeridianError("分析版本结果中没有分区数据")
    vals = []
    for z in zones:
        if field not in z:
            raise MeridianError(
                f"分析版本结果分区缺少 {field} 字段（旧版本？请基于当前算法重新分析）"
            )
        v = z[field]
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise MeridianError(f"分区 {z['index']} 的 {field} 不是有限数值")
        vals.append(float(v))
    return vals
