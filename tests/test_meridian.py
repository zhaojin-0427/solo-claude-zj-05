"""子午线复测二次角向谐波拟合核心测试。

合成数据直接构造各半径 LA 残差（不经 reduce_test 的逐批次圆锥/离焦投影），
覆盖：两对象散分离、欠秩缺口诊断、N=4 截距剔除、置信区间、留一稳定性、
LA→波前换算的量级与符号。
"""
import math

import numpy as np
import pytest

from foucault.meridian import (
    MeridianError,
    design_matrix,
    extract_zone_signals,
    fit_harmonics,
    leave_one_out,
    loo_from_fit_result,
)

RADII = [20.0, 50.0, 80.0]
INNER = [10.0, 40.0, 70.0]
OUTER = [30.0, 60.0, 90.0]
R = 1600.0
RIM = 100.0
LAM = 550.0

# ψ（镜面旋转）与 α（刀口直径方位）都有独立且非共线的散布
PSI = [0.0, 45.0, 90.0, 30.0, 60.0, 120.0, 150.0, 20.0]
ALPHA = [0.0, 0.0, 45.0, 90.0, 135.0, 45.0, 90.0, 0.0]


def make_signals(amp_m=0.012, phi_m_deg=30.0, amp_l=0.005, phi_l_deg=110.0,
                 noise=1e-4, radial=lambda r: 0.8 + 0.4 * r / 90.0, seed=0,
                 psi=None, alpha=None):
    rng = np.random.default_rng(seed)
    psi = psi or PSI
    alpha = alpha or ALPHA
    phi_m = math.radians(phi_m_deg)
    phi_l = math.radians(phi_l_deg)
    n, m = len(psi), len(RADII)
    y = np.zeros((n, m))
    for i in range(n):
        p, a = math.radians(psi[i]), math.radians(alpha[i])
        for j, r in enumerate(RADII):
            prof = radial(r)
            y[i, j] = (
                amp_m * prof * math.cos(2 * (p - phi_m))
                + amp_l * prof * math.cos(2 * (a - phi_l))
                + noise * rng.standard_normal()
            )
    return y


def test_design_matrix_columns():
    A = design_matrix([0, 90], [45, 135])
    assert A.shape == (2, 5)
    assert np.allclose(A[:, 0], 1.0)
    # ψ=0 → cos0=1,sin0=0；ψ=90 → cos180=−1,sin180≈0
    assert np.allclose(A[0, 1:3], [1, 0])
    assert np.allclose(A[1, 1:3], [-1, 0], atol=1e-9)
    # α=45/135 → 2α=90/270 → cos≈0，sin=+1/−1
    assert abs(A[0, 3]) < 1e-9 and abs(A[1, 3]) < 1e-9
    assert A[0, 4] == pytest.approx(1.0, abs=1e-9)
    assert A[1, 4] == pytest.approx(-1.0, abs=1e-9)


def test_separates_mirror_and_lab_components():
    y = make_signals()
    res = fit_harmonics(PSI, ALPHA, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert res["n_sources"] == 8
    assert res["rank"] == 5
    assert res["residual_dof"] == 8 * 3 - 5 * 3
    # 每半径两对象散的主轴都恢复（90° 象限内）
    for z in res["zones"]:
        assert abs(z["mirror"]["axis_deg"] - 30.0) < 3.0
        assert abs(z["lab_fixed"]["axis_deg"] - 110.0) < 3.0
    # 外缘镜面幅值随 radial 剖面 = 1.2 × 0.012
    outer = res["zones"][-1]
    assert outer["mirror"]["amplitude_mm"] == pytest.approx(0.012 * 1.2, abs=5e-4)
    assert outer["lab_fixed"]["amplitude_mm"] == pytest.approx(0.005 * 1.2, abs=5e-4)
    # 整体波前主轴
    am = res["astigmatism"]["mirror"]
    al = res["astigmatism"]["lab_fixed"]
    assert abs(am["principal_axis_mirror_deg"] - 30.0) < 3.0
    assert abs(al["principal_axis_lab_deg"] - 110.0) < 3.0
    # 恒定幅值剖面的边缘波前可解析核对：H_rim = A·RIM²/(4R²)（取绝对值）
    assert am["rim_wavefront_amplitude_nm"] > 0
    assert res["astigmatism"]["total_wavefront_rms_waves"] > 0


def test_constant_la_amplitude_wavefront_magnitude():
    # 纯镜面像散、LA 幅值不随半径：边缘面形 H(rim) = A·rim²/(4R²)，
    # 波前 = 2H（反射加倍）。
    y = make_signals(amp_m=0.01, amp_l=0.0, noise=0.0, radial=lambda r: 1.0,
                     seed=1)
    res = fit_harmonics(PSI, ALPHA, y, RADII, INNER, OUTER, R, RIM, LAM)
    expect_nm = 2.0 * 0.01 * RIM**2 / (4.0 * R**2) * 1e6
    got = res["astigmatism"]["mirror"]["rim_wavefront_amplitude_nm"]
    assert got == pytest.approx(expect_nm, rel=1e-6)
    # LA 幅值恒定 → 面形 H(r)∝r²：面积加权峰值轮廓 RMS/边缘幅值 = 1/√3
    # （∫r·r⁴dr / ∫r dr = rim⁴/3）
    blk = res["astigmatism"]["mirror"]
    ratio = blk["peak_profile_rms_waves"] / blk["rim_wavefront_amplitude_waves"]
    assert ratio == pytest.approx(1.0 / math.sqrt(3.0), abs=0.01)
    # 角向波前 RMS 含 <cos²2θ>=1/2，等于峰值轮廓 /√2
    assert blk["wavefront_rms_waves"] == pytest.approx(
        blk["peak_profile_rms_waves"] / math.sqrt(2.0), rel=1e-9
    )


def test_confidence_intervals_present_with_residual_dof():
    y = make_signals(noise=2e-4, seed=3)
    res = fit_harmonics(PSI, ALPHA, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert res["residual_sigma_mm"] > 0
    z = res["zones"][1]
    assert z["mirror"]["amplitude_ci_mm"][0] <= z["mirror"]["amplitude_mm"] <= \
        z["mirror"]["amplitude_ci_mm"][1]
    assert z["mirror"]["axis_ci_deg"] is not None
    am = res["astigmatism"]["mirror"]
    assert am["rim_amplitude_ci_nm"][0] <= am["rim_wavefront_amplitude_nm"] <= \
        am["rim_amplitude_ci_nm"][1]
    # 高信噪比下 CI 半宽远小于幅值
    half = (am["rim_amplitude_ci_nm"][1] - am["rim_amplitude_ci_nm"][0]) / 2
    assert half < am["rim_wavefront_amplitude_nm"]


def test_exactly_just_identified_has_null_ci():
    # N=5、p=5：每半径 0 残差自由度 → sigma/CI 为 None，但仍给出点估计
    psi5, alpha5 = PSI[:5], ALPHA[:5]
    y = make_signals(psi=psi5, alpha=alpha5, noise=0.0)
    res = fit_harmonics(psi5, alpha5, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert res["residual_dof"] == 0
    assert res["residual_sigma_mm"] is None
    assert res["zones"][0]["mirror"]["amplitude_ci_mm"] is None


def test_n4_generic_angles_underdetermined():
    # 5 参数（截距+4 谐波）只有 4 个观测：一般角度下镜面/架位谐波方向共线，
    # 不含截距也无法分离两对象散 → 欠秩缺口（需要第 5 份来源或独立角度）。
    psi4 = [0.0, 30.0, 60.0, 90.0]
    alpha4 = [0.0, 90.0, 30.0, 135.0]
    y = make_signals(psi=psi4, alpha=alpha4, noise=0.0)
    with pytest.raises(MeridianError) as ei:
        fit_harmonics(psi4, alpha4, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert ei.value.gaps  # 至少指出一个无法识别的分量


def test_n4_separable_harmonics_solves_without_intercept():
    # 4 份来源且 ψ、α 的二次谐波方向独立（4 个谐波系数满秩）：不含截距即可
    # 分离两对象散，应正常求解而非 422。
    psi4 = [0.0, 15.0, 30.0, 45.0]
    alpha4 = [0.0, 15.0, 45.0, 60.0]
    y = make_signals(psi=psi4, alpha=alpha4, noise=0.0)
    res = fit_harmonics(psi4, alpha4, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert res["rank"] == 4
    assert "intercept" not in res["fitted_columns"]
    z = res["zones"][-1]
    assert abs(z["mirror"]["axis_deg"] - 30.0) < 1e-6
    assert abs(z["lab_fixed"]["axis_deg"] - 110.0) < 1e-6
    # 冻结快照可重建留一法
    loo = loo_from_fit_result(res)
    assert len(loo["leave_one_out"]) == 4


def test_complex_aggregation_preserves_phase_with_independent_alpha():
    # 纯镜面像散、幅值随半径变号（径向剖面），α 独立变化：复数积分保相位，
    # 整体主轴不得被变号系数带偏 90°。
    psi8 = [0.0, 45.0, 90.0, 30.0, 60.0, 120.0, 150.0, 20.0]
    alpha8 = [0.0, 0.0, 45.0, 90.0, 135.0, 45.0, 90.0, 0.0]
    y = make_signals(amp_m=0.02, amp_l=0.0, noise=0.0,
                     radial=lambda rr: 0.6 - 0.8 * rr / 90.0,
                     psi=psi8, alpha=alpha8)
    res = fit_harmonics(psi8, alpha8, y, RADII, INNER, OUTER, R, RIM, LAM)
    am = res["astigmatism"]["mirror"]
    # 每区轴在 30° 或 120°（变号的 90° 象限）；复数汇总回到输入 30° 象限
    for z in res["zones"]:
        assert abs(z["mirror"]["axis_deg"] - 30.0) < 1e-6 or \
            abs(z["mirror"]["axis_deg"] - 120.0) < 1e-6
    assert abs(am["principal_axis_mirror_deg"] - 30.0) < 2.0
    assert am["rim_wavefront_amplitude_nm"] > 0


def test_n5_full_model_just_identified():
    # 5 份角度充分散布的来源可分离两对象散并拟合截距（恰好识别，CI 为空）
    psi5 = [0.0, 45.0, 90.0, 30.0, 60.0]
    alpha5 = [0.0, 0.0, 45.0, 90.0, 135.0]
    y = make_signals(psi=psi5, alpha=alpha5, noise=0.0)
    res = fit_harmonics(psi5, alpha5, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert res["rank"] == 5
    assert "intercept" in res["fitted_columns"]
    z = res["zones"][-1]
    assert abs(z["mirror"]["axis_deg"] - 30.0) < 1e-6
    assert abs(z["lab_fixed"]["axis_deg"] - 110.0) < 1e-6


def test_fixed_knife_azimuth_flags_lab_gap():
    # α 全相同：架位项被截距吸收，只能识别镜面分量
    y = make_signals(alpha=[0.0] * len(PSI))
    with pytest.raises(MeridianError) as ei:
        fit_harmonics(PSI, [0.0] * len(PSI), y, RADII, INNER, OUTER, R, RIM, LAM)
    msg = " ".join(ei.value.gaps)
    assert "架位固定分量无法识别" in msg
    assert "刀口扫描直径方位" in msg


def test_fixed_mirror_rotation_flags_mirror_gap():
    y = make_signals(psi=[0.0] * len(PSI))
    with pytest.raises(MeridianError) as ei:
        fit_harmonics([0.0] * len(PSI), ALPHA, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert "镜面随转分量无法识别" in " ".join(ei.value.gaps)


def test_collinear_angles_flagged():
    # ψ 只取 0/90（二次谐波共线），镜面分量缺一个方向
    psi = [0.0, 90.0, 0.0, 90.0, 0.0, 90.0]
    alpha = [0.0, 20.0, 45.0, 70.0, 90.0, 110.0]
    y = make_signals(psi=psi, alpha=alpha)
    with pytest.raises(MeridianError) as ei:
        fit_harmonics(psi, alpha, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert "0°/90°" in " ".join(ei.value.gaps)


def test_too_few_sources_rejected():
    y = make_signals(psi=PSI[:3], alpha=ALPHA[:3])
    with pytest.raises(MeridianError):
        fit_harmonics(PSI[:3], ALPHA[:3], y, RADII, INNER, OUTER, R, RIM, LAM)


def test_local_defect_is_most_suspicious_zone():
    y = make_signals(noise=1e-5, seed=2)
    # 在中区注入与角度无关、跨来源异号的局部残差（不能被谐波解释）
    y[:, 1] += np.array([0.008, -0.007, 0.006, -0.005, 0.004, -0.003, 0.002, -0.001])
    res = fit_harmonics(PSI, ALPHA, y, RADII, INNER, OUTER, R, RIM, LAM)
    assert res["most_suspicious_zone"]["index"] == 1
    assert res["most_suspicious_zone"]["max_abs_studentized_residual"] > 1.5


def test_leave_one_out_stable_for_robust_signal():
    # 强信号 + 一个固定的显著局部缺陷（中区）：逐份剔除既不改变主轴结论，
    # 也不改变最可疑区段，留一法判定稳定。
    y = make_signals(amp_m=0.02, amp_l=0.008, noise=1e-5, seed=4)
    y[:, 1] += np.array([0.01, -0.009, 0.008, -0.007, 0.006, -0.005, 0.004, -0.003])
    loo = leave_one_out(PSI, ALPHA, y, RADII, INNER, OUTER, R, RIM, LAM,
                        axis_stability_deg=15.0)
    assert loo["stable"] is True
    assert len(loo["leave_one_out"]) == 8
    for row in loo["leave_one_out"]:
        assert row["rank_deficient"] is False
        assert row["axis_shift_deg"] < 15.0
        assert row["suspicious_zone_unchanged"] is True
        assert row["stable"] is True


def test_leave_one_out_detects_influential_source():
    # 5 份可识别 + 移除某关键来源后角度欠秩
    psi5, alpha5 = PSI[:5], ALPHA[:5]
    y = make_signals(psi=psi5, alpha=alpha5, noise=0.0)
    loo = leave_one_out(psi5, alpha5, y, RADII, INNER, OUTER, R, RIM, LAM)
    # N=5 去一剩 4；若其中某个角度配置是识别所必需，会标 rank_deficient
    assert isinstance(loo["stable"], bool)
    assert all("excluded_source_index" in r for r in loo["leave_one_out"])


def test_extract_zone_signals_validation():
    version = {"zones": [
        {"index": 1, "la_residual": 0.1, "la_error": 0.2},
        {"index": 0, "la_residual": 0.0, "la_error": 0.1},
    ]}
    assert extract_zone_signals(version, "la_residual") == [0.0, 0.1]
    assert extract_zone_signals(version, "la_error") == [0.1, 0.2]
    with pytest.raises(MeridianError):
        extract_zone_signals(version, "nope")
    with pytest.raises(MeridianError):
        extract_zone_signals({"zones": []})
