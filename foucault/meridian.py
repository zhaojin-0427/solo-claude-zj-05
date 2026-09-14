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
- 刀口方位 α 固定、只旋转镜面时，架位列秩不足，只能识别镜面分量；反之镜面
  不旋转时只能识别架位分量；ψ 与 α 同步共线变化（α≡±ψ+常数）时两对象散
  混淆。欠秩时不强行给出伪结果，而是逐分量指出缺口与补测建议
  （MeridianError.gaps）。
可辨识性只看完整四列谐波矩阵（镜面 cos/sin、架位 cos/sin）是否满秩。
4 份来源且四列满秩（rank 4）时四个谐波系数唯一可解，直接拟合（不估轴对称
截距 c，c 并入残差），如 ψ=[0,30,60,90]、α=[0,90,30,135]；≥5 份且 c 相对
谐波独立可估（5 列满秩）时再拟合截距。

LA → 波前：各半径的复 LA 谐波 A_j = c_j + i·s_j（保留相位）按环带面积
积分 h = (1/2R²)∫₀^rim ρ·A(ρ) dρ，波前 W = 2h（反射加倍）。整体主轴由
复积分辐角 atan2(Im h, Re h)/2 给出——不先取实幅值、不做逐区轴圆形平均，
因而径向系数变号、α 独立变化时主轴与幅值都不失真。
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


def _harmonic_gaps(psi: np.ndarray, alpha: np.ndarray) -> list[str]:
    """按完整四列谐波矩阵是否满秩判定可辨识性，并翻译角度缺口。

    模型（不含轴对称截距）的四列为镜面 cos2ψ/sin2ψ 与架位 cos2α/sin2α：
    四列满秩（rank 4）时四个谐波系数唯一可解，N=4 即可建档求解。仅当真正
    欠秩（某角度无变化、只取 0/90 共线方位，或 ψ 与 α 同步使两类像散混淆）
    时才返回缺口说明。
    """
    M2 = np.column_stack([
        np.cos(2 * np.radians(psi)), np.sin(2 * np.radians(psi))
    ])
    L2 = np.column_stack([
        np.cos(2 * np.radians(alpha)), np.sin(2 * np.radians(alpha))
    ])
    mirror_diverse = _norm_rank(M2) >= 2
    lab_diverse = _norm_rank(L2) >= 2
    joint = np.column_stack([M2, L2])
    confounded = _norm_rank(joint) < 4  # 两对象散方向空间重叠（通常 α≡±ψ+const）

    gaps: list[str] = []
    if not mirror_diverse:
        gaps.append(
            "镜面随转分量无法识别：镜面旋转角只在 0°/90° 等共线方位取值或无"
            "变化，需补充旋转角相差非 0°/90° 的来源"
        )
    if not lab_diverse:
        gaps.append(
            "架位固定分量无法识别：刀口扫描直径方位只在 0°/90° 等共线方位取值"
            "或无变化，需补充方位相差非 0°/90° 的来源"
        )
    if mirror_diverse and lab_diverse and confounded:
        gaps.append(
            "镜面随转分量与架位固定分量互相混淆：镜面旋转角与刀口扫描直径"
            "方位同步共线变化（α≡±ψ+常数），需让 ψ 与 α 独立变化，或补充"
            "更多来源使四个谐波方向满秩"
        )
    return gaps


# ---------------- 波前换算 ----------------


def _zone_radial_weights(inner, outer, rim: float, R: float):
    """各分区 LA 谐波复系数对边缘波前复振幅的贡献权重（实数，mm→mm）。

    LA 像散幅值在相邻等效半径的中点之间按该分区测量值代表；首区从 0
    起、末区延伸到口径边缘 rim，使幅值恒为 A 时 Σw = rim²/(4R²)
    （波前 = 2·(1/2R²)·A·rim²/2 = A·rim²/(2R²)，取正号以保持
    LA 主轴与波前主轴一致，符号为像散主轴 90° 象限的约定）。
    返回按原始（未排序）分区顺序的权重。
    """
    radii_ref = np.sqrt((np.asarray(inner) ** 2 + np.asarray(outer) ** 2) / 2.0)
    order = np.argsort(radii_ref)
    r = radii_ref[order]
    m = r.size
    edges = np.zeros(m + 1)
    edges[0] = 0.0
    for j in range(m - 1):
        edges[j + 1] = 0.5 * (r[j] + r[j + 1])
    edges[m] = rim
    # 波前取 +(1/2R²)∫ρdρ（即 +∫ρdρ/(2R²)）；∫ρdρ=(e_{j+1}²−e_j²)/2
    w_sorted = (edges[1:] ** 2 - edges[:-1] ** 2) / (4.0 * R * R)
    w = np.zeros(m)
    w[order] = w_sorted
    return w


def _radial_complex_profile(profile_complex, radii, rim: float, R: float,
                            n_grid: int = 600):
    """由逐区 LA 谐波**复**系数重建口径上的波前复振幅包络 W(r)/2（mm）。

    profile_complex[j] = c_j + i·s_j（保留相位，不先取幅值）。复系数在相邻
    等效半径中点之间分段常数（首段从 0 起按首区值、末段延伸到 rim），逐段
    积分 h(r) = (1/2R²)∫₀ʳ ρ·A(ρ) dρ（取正号保持主轴方向）。
    返回 (r_grid, 复包络 h)。波前 = 2·h。
    """
    radii = np.asarray(radii, dtype=float)
    order = np.argsort(radii)
    r = radii[order]
    a = np.asarray(profile_complex, dtype=complex)[order]
    m = r.size
    edges = [0.0]
    for j in range(m - 1):
        edges.append(0.5 * (r[j] + r[j + 1]))
    edges.append(rim)

    def amp_at(rho):
        for j in range(m):
            lo, hi = edges[j], edges[j + 1]
            if j == 0:
                if rho <= hi:
                    return a[0]
            elif lo < rho <= hi:
                return a[j]
        return a[-1]

    g = np.linspace(0.0, rim, n_grid)
    h = np.zeros_like(g, dtype=complex)
    for k in range(1, g.size):
        ra, rb = g[k - 1], g[k]
        aa, ab = amp_at(ra), amp_at(rb)
        # h = (1/2R²)∫ρ·A dρ；梯形 ∫_ra^rb ρ·A dρ ≈ Δr(ra·A_a+rb·A_b)/2
        h[k] = h[k - 1] + (rb - ra) * (ra * aa + rb * ab) / (4.0 * R * R)
    return g, h


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

    # 可辨识性只看完整四列谐波矩阵是否满秩：rank 4 时四个谐波系数唯一可解
    # （N=4 即直接拟合谐波、不估轴对称截距）。只有真正欠秩的角度组合才报缺口。
    A_full = design_matrix(psi, alpha)
    rank_full = _norm_rank(A_full)
    A_harm = A_full[:, 1:5]
    if _norm_rank(A_harm) < 4:
        raise MeridianError(
            "角度组合使二次谐波拟合矩阵欠秩，镜面随转分量与架位固定分量无法分离",
            _harmonic_gaps(psi, alpha),
        )

    # N≥5 且截距相对四个谐波独立可估时再拟合截距（5 列满秩）；否则只拟合
    # 4 个谐波（N=4，或截距恰好落在谐波空间），轴对称残余并入残差。
    use_cols = list(_COL_MIRROR) + list(_COL_LAB)
    A4 = A_full[:, use_cols]
    if n >= 5 and _norm_rank(np.column_stack([np.ones(n), A4])) > 4:
        use_cols = [_COL_INTERCEPT] + use_cols
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

    # 整体像散波前：镜面 / 架位分别把**复** LA 谐波系数（c+i·s，保留相位）
    # 按环带面积积分到口径边缘。主轴直接由复数积分的辐角给出，不做逐区轴的
    # 圆形平均（后者会被径向系数变号污染），因此 α 独立变化、径向幅值起伏时
    # 主轴与幅值都不失真。
    def wavefront_block(prof_c, prof_s, i_c_coef, i_s_coef, axis_name: str):
        w = _zone_radial_weights(inner_radii, outer_radii, rim, R)
        prof = np.asarray(prof_c, dtype=float) + 1j * np.asarray(prof_s, dtype=float)
        # 边缘波前复振幅（面形，取正号保持与 LA 主轴同象限）：H = Σ w_j·A_j
        H = complex(np.sum(w * prof))
        h_amp_mm = float(np.abs(H))
        w_amp_nm = 2.0 * h_amp_mm * NM_PER_MM  # 波前 = 2·面形
        axis = _axis_deg(H.real, H.imag)
        cov2 = (
            sigma2 * AtA_inv[np.ix_([i_c_coef, i_s_coef], [i_c_coef, i_s_coef])]
            if sigma2 is not None
            else None
        )
        se_amp_nm = se_axis_deg = None
        if cov2 is not None and h_amp_mm > 0:
            # Hc=Σw·c, Hs=Σw·s；跨半径系数独立（同设计矩阵、σ² 块对角）
            var_c = float((w**2).sum()) * cov2[0, 0]
            var_s = float((w**2).sum()) * cov2[1, 1]
            cov_cs = float((w**2).sum()) * cov2[0, 1]
            Hc, Hs = H.real, H.imag
            var_amp = (
                Hc * Hc * var_c + Hs * Hs * var_s + 2 * Hc * Hs * cov_cs
            ) / (h_amp_mm**2)
            se_amp_nm = 2.0 * NM_PER_MM * math.sqrt(max(0.0, var_amp))
            # θ = atan2(Hs,Hc)/2
            var_axis_rad = (
                Hs * Hs * var_c + Hc * Hc * var_s - 2 * Hc * Hs * cov_cs
            ) / (4.0 * h_amp_mm**4)
            se_axis_deg = math.degrees(math.sqrt(max(0.0, var_axis_rad)))
        # 面积加权 RMS：在细密半径网格上用复系数重建波前包络（保留相位），
        # |h(r)| 按面元 ∝ r 加权；角向 <cos²2θ>=1/2。
        g_r, h_g = _radial_complex_profile(prof, radii, rim, R)
        h_abs = np.abs(h_g)
        # 均匀 dr：Σr/Σr 即面积加权均值；波前 = 2·面形
        peak_mm2 = float((g_r * h_abs**2).sum()) / float(g_r.sum())
        rms_peak_nm = 2.0 * math.sqrt(peak_mm2) * NM_PER_MM
        rms_wavefront_nm = rms_peak_nm / math.sqrt(2.0)
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

    ast_m = wavefront_block(prof_mc, prof_ms, i_mc, i_ms, "mirror")
    ast_l = wavefront_block(prof_lc, prof_ls, i_lc, i_ls, "lab")

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

    # 冻结本次求解所用来源的角度（顺序与信号矩阵行一致）；配合分区信号即可
    # 不依赖研究当前状态复算留一法/重解（历史版本严格使用该快照）。
    frozen_sources = [
        {
            "row_index": int(i),
            "mirror_rotation_deg": float(psi[i]),
            "knife_diameter_azimuth_deg": float(alpha[i]),
        }
        for i in range(n)
    ]

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
        "constants": {
            "radius_of_curvature_mm": float(R),
            "rim_radius_mm": float(rim),
            "wavelength_nm": float(wavelength_nm),
            "confidence_level": float(confidence_level),
        },
        "frozen_sources": frozen_sources,
        "radii_mm": [float(r) for r in radii],
        "inner_radii_mm": [float(x) for x in inner_radii],
        "outer_radii_mm": [float(x) for x in outer_radii],
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


def signals_from_fit_result(fit_result: dict) -> list[list[float]]:
    """从研究版本结果还原 N×M 信号矩阵（每份来源在每分区的观测 LA，mm）。

    per_source 按分区存储且行序与 frozen_sources 一致，重建后供留一法在不访问
    研究当前状态的情况下严格按该版本快照复算。
    """
    zones = sorted(fit_result["zones"], key=lambda z: z["index"])
    if not zones:
        raise MeridianError("版本结果中没有分区数据")
    n = fit_result["n_sources"]
    matrix = [[0.0] * len(zones) for _ in range(n)]
    for j, z in enumerate(zones):
        rows = sorted(z["per_source"], key=lambda p: p["source_index"])
        if len(rows) != n:
            raise MeridianError("版本 per_source 行数与 n_sources 不一致")
        for i, p in enumerate(rows):
            matrix[i][j] = float(p["observed_mm"])
    return matrix


def loo_from_fit_result(fit_result: dict, axis_stability_deg: float = 15.0) -> dict:
    """严格按某条研究版本冻结的来源/信号复算留一法（不混入研究当前状态）。"""
    c = fit_result["constants"]
    src = fit_result["frozen_sources"]
    psi = [s["mirror_rotation_deg"] for s in src]
    alpha = [s["knife_diameter_azimuth_deg"] for s in src]
    Y = signals_from_fit_result(fit_result)
    return leave_one_out(
        psi,
        alpha,
        Y,
        fit_result["radii_mm"],
        fit_result["inner_radii_mm"],
        fit_result["outer_radii_mm"],
        c["radius_of_curvature_mm"],
        c["rim_radius_mm"],
        c["wavelength_nm"],
        confidence_level=c["confidence_level"],
        axis_stability_deg=axis_stability_deg,
    )
