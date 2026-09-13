"""光学还原精度测试：由已知面形误差合成刀口读数，验证还原结果。"""
import math

import numpy as np
import pytest

from foucault.optics import (
    AnalysisError,
    Constants,
    Options,
    ZoneInput,
    reduce_test,
)

D, R, K, LAM = 200.0, 1600.0, -1.0, 550.0
ZONES8 = [(0, 16), (16, 32), (32, 48), (48, 62), (62, 74), (74, 86), (86, 94), (94, 100)]


def r_mean(ri, ro):
    return math.sqrt((ri**2 + ro**2) / 2.0)


def synth_zones(e_true, source_mode="fixed", n=3, noise=0.0, seed=1, offset=0.5,
                zones=ZONES8, k_target=K):
    """由真实面形误差 e_true(r)[nm] 合成各区刀口读数（mm）。"""
    rng = np.random.default_rng(seed)
    sf = 1.0 if source_mode == "fixed" else 2.0
    out = []
    for ri, ro in zones:
        rm = r_mean(ri, ro)
        la_ideal = -k_target * rm**2 / R
        dr = 1e-3
        de_dr = (e_true(rm + dr) - e_true(rm - dr)) / (2 * dr)  # nm/mm
        la_err = -2.0 * R**2 / rm * de_dr * 1e-6  # mm
        base = offset + (la_ideal + la_err) / sf
        out.append(
            ZoneInput(ri, ro, [float(base + rng.normal(0, noise)) for _ in range(n)])
        )
    return out


def make_constants(source_mode="fixed"):
    return Constants(D, R, K, LAM, source_mode, instrument_offset=0.0)


def test_zero_error_parabola():
    """完美抛物面：残余误差为零，最佳拟合圆锥常数即目标值。"""
    zones = synth_zones(lambda r: 0.0)
    res = reduce_test(make_constants(), zones, Options())
    assert res["summary"]["wavefront_rms_nm"] < 1e-6
    assert res["summary"]["wavefront_pv_nm"] < 1e-5
    assert res["fit"]["best_fit_conic_constant"] == pytest.approx(-1.0, abs=1e-9)
    assert res["summary"]["strehl"] == pytest.approx(1.0, abs=1e-9)
    assert res["error_band"]["out_of_band_zones"] == []


def test_sphere_against_parabola_target():
    """球面按抛物面目标分析：各区读数相同，最佳拟合圆锥常数回到 0。"""
    zones = synth_zones(lambda r: 0.0, k_target=0.0)  # 球面：LA 处处为 0
    res = reduce_test(make_constants(), zones, Options())
    # 分段线性近似的固有偏差 ~0.004（对应面形误差 <0.02nm，可忽略）
    assert res["fit"]["best_fit_conic_constant"] == pytest.approx(0.0, abs=5e-3)
    assert res["summary"]["wavefront_rms_nm"] < 1.0  # 分段线性近似固有残余


def test_surface_profile_recovery():
    """含六阶项的面形误差：还原轮廓与理论投影残差一致。"""
    rim = D / 2.0
    e_true = lambda r: 30.0 * (r / rim) ** 6 - 12.0 * (r / rim) ** 4 + 2.0 * (r / rim) ** 2 + 5.0
    zones = synth_zones(e_true)
    res = reduce_test(make_constants(), zones, Options())
    g = np.array(res["profile"]["r_mm"])
    e_rec = np.array(res["profile"]["residual_surface_nm"])
    # 理论：e_true 投影掉 {1, u^2, u^4} 后的残差
    u = g / rim
    A = np.column_stack([np.ones_like(u), u**2, u**4])
    coef, *_ = np.linalg.lstsq(A, e_true(g), rcond=None)
    e_exp = e_true(g) - A @ coef
    assert np.max(np.abs(e_rec - e_exp)) < 1.5  # nm
    assert res["summary"]["surface_rms_nm"] == pytest.approx(e_exp.std(), rel=0.05)


def test_moving_source_factor():
    """移动光源读数差减半，还原结果与固定光源一致。"""
    e_true = lambda r: 25.0 * (r / 100.0) ** 6
    z_fixed = synth_zones(e_true, source_mode="fixed")
    z_moving = synth_zones(e_true, source_mode="moving")
    res_f = reduce_test(make_constants("fixed"), z_fixed, Options())
    res_m = reduce_test(make_constants("moving"), z_moving, Options())
    assert res_m["summary"]["wavefront_rms_nm"] == pytest.approx(
        res_f["summary"]["wavefront_rms_nm"], rel=1e-9
    )
    assert res_m["summary"]["source_factor"] == 2.0


def test_zero_point_and_offset_absorbed():
    """零点参考与仪器偏移只影响可被拟合吸收的常数项，残余不变。"""
    e_true = lambda r: 20.0 * (r / 100.0) ** 6
    z1 = synth_zones(e_true, offset=0.5)
    z2 = synth_zones(e_true, offset=3.7)  # 整体平移 = 不同零点
    res1 = reduce_test(make_constants(), z1, Options())
    res2 = reduce_test(make_constants(), z2, Options())
    assert res1["summary"]["wavefront_rms_nm"] == pytest.approx(
        res2["summary"]["wavefront_rms_nm"], rel=1e-9
    )
    # 仪器偏移同样被零点差分吸收
    c = make_constants()
    c.instrument_offset = 0.25
    res3 = reduce_test(c, z1, Options())
    assert res1["summary"]["wavefront_rms_nm"] == pytest.approx(
        res3["summary"]["wavefront_rms_nm"], rel=1e-9
    )
    # 校正后的均值确实扣除了偏移
    assert res3["zones"][0]["mean_corrected"] == pytest.approx(
        res1["zones"][0]["mean_corrected"] - 0.25
    )


def test_dispersion_and_error_band():
    """离散度汇总与误差带标记。"""
    # 边缘局部凸起（"塌边"的反向），投影残差大，边缘区必然超 λ/4
    e_true = lambda r: 300.0 * np.exp(-(((r / 100.0) - 0.9) / 0.08) ** 2)
    zones = synth_zones(e_true, noise=0.01, seed=3)
    res = reduce_test(make_constants(), zones, Options(error_band_waves=0.25))
    assert res["dispersion"]["mean_std"] > 0
    assert res["dispersion"]["max_range"] >= res["dispersion"]["max_std"] > 0
    assert res["error_band"]["out_of_band_zones"], "大误差应标出超差区段"
    for i in res["error_band"]["out_of_band_zones"]:
        assert abs(res["zones"][i]["wavefront_waves"]) > 0.25


def test_transverse_aberration_sign():
    """横向像差与残余纵向像差符号一致、量级合理。"""
    e_true = lambda r: 100.0 * (r / 100.0) ** 6
    zones = synth_zones(e_true)
    res = reduce_test(make_constants(), zones, Options())
    for z in res["zones"]:
        ta = z["transverse_aberration_mm"]
        la = z["la_residual"]
        assert ta == pytest.approx(la * z["r_mean"] / R, rel=1e-9)


def test_insufficient_readings_rejected():
    zones = synth_zones(lambda r: 0.0, n=1)
    with pytest.raises(AnalysisError):
        reduce_test(make_constants(), zones, Options(min_readings_per_zone=2))


def test_noise_robustness():
    """含噪声读数：多次平均后还原 RMS 与无噪声情况接近。"""
    e_true = lambda r: 400.0 * (r / 100.0) ** 6  # 波前残余 RMS ≈ 15nm
    z_clean = synth_zones(e_true, n=5)
    z_noisy = synth_zones(e_true, n=5, noise=0.001, seed=7)  # 1μm 重复性
    res_c = reduce_test(make_constants(), z_clean, Options())
    res_n = reduce_test(make_constants(), z_noisy, Options())
    assert res_n["summary"]["wavefront_rms_nm"] == pytest.approx(
        res_c["summary"]["wavefront_rms_nm"], rel=0.15
    )
