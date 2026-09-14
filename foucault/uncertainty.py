"""Foucault 分区分析的不确定度传播（Monte Carlo）。

请求可为口径、曲率半径、检测波长与仪器零位给出标准不确定度，并记录刀口尺
分辨率；分区重复性由重复读数估计或逐区指定。每轮抽样共同扰动常量与零位，
对每条读数叠加均匀量化误差（±分辨率/2 矩形分布）与重复性正态扰动，再复用
`optics.reduce_test` 完成光学还原。输出最佳拟合圆锥、RMS/PV 波前与 Strehl
的 P5/P50/P95、各区越过误差带的概率，以及逐项固定输入重算的方差贡献。

随机种子随分析版本冻结：请求未给种子时由输入哈希派生；所有随机数按固定
顺序一次性抽取（常量 → 量化 → 重复性），逐项固定输入的方差贡献重算使用
同一批随机数（公共随机数法），相同版本重复计算结果完全一致。
"""
from __future__ import annotations

import numpy as np

from .optics import (
    AnalysisError,
    Constants,
    Options,
    ZoneInput,
    reduce_test,
    zone_statistics,
)


class UncertaintyError(Exception):
    """不确定度传播前置条件不满足或抽样产生非法参数。"""


# 方差贡献逐项固定的输入分组（顺序固定，响应中按此顺序给出）
GROUPS = (
    "diameter",
    "radius_of_curvature",
    "wavelength",
    "instrument_offset",
    "quantization",
    "repeatability",
)


def derive_seed(input_hash: str) -> int:
    """由输入哈希派生 31 位随机种子（确定可复现，随版本冻结）。"""
    return int(input_hash[:16], 16) % (2**31)


def resolve_model(
    config: dict,
    unit_factor: float,
    zones: list[ZoneInput],
    mask_knife_resolution_mm: float | None = None,
) -> dict:
    """请求配置（声明单位）→ 冻结模型（mm）。

    seed 保持请求值（可为 None，由调用方在定哈希后派生）；knife_resolution
    缺省时回退到遮罩版本冻结的分辨率，无遮罩则为 0（不加入量化误差）；
    repeatability_mode=estimated 时逐区由有效重复读数估计标准差（n<2 为 0）。
    """
    f = float(unit_factor)
    kr = config.get("knife_resolution")
    if kr is None:
        kr_mm = float(mask_knife_resolution_mm) if mask_knife_resolution_mm else 0.0
    else:
        kr_mm = float(kr) * f
    if config["repeatability_mode"] == "specified":
        rep = [float(v) * f for v in config["zone_repeatability"]]
    else:
        rep = [
            zone_statistics(z.readings)["std"] if len(z.readings) >= 2 else 0.0
            for z in zones
        ]
    return {
        "n_samples": int(config["n_samples"]),
        "seed": config.get("seed"),
        "diameter_std_mm": float(config["diameter_std"]) * f,
        "radius_of_curvature_std_mm": float(config["radius_of_curvature_std"]) * f,
        "wavelength_std_nm": float(config["wavelength_std_nm"]),
        "instrument_offset_std_mm": float(config["instrument_offset_std"]) * f,
        "knife_resolution_mm": kr_mm,
        "repeatability_mode": config["repeatability_mode"],
        "zone_repeatability_mm": rep,
    }


def _group_active(model: dict, group: str) -> bool:
    """该输入分组是否带有非零不确定度（零不确定度分组固定后结果不变）。"""
    if group == "diameter":
        return model["diameter_std_mm"] > 0.0
    if group == "radius_of_curvature":
        return model["radius_of_curvature_std_mm"] > 0.0
    if group == "wavelength":
        return model["wavelength_std_nm"] > 0.0
    if group == "instrument_offset":
        return model["instrument_offset_std_mm"] > 0.0
    if group == "quantization":
        return model["knife_resolution_mm"] > 0.0
    return any(s > 0.0 for s in model["zone_repeatability_mm"])


def _mc_run(
    constants: Constants,
    zones: list[ZoneInput],
    options: Options,
    model: dict,
    fix: frozenset | set | tuple = (),
    want_zone_residuals: bool = False,
) -> dict:
    """执行 Monte Carlo 传播；fix 中的输入分组固定在标称值（仍消耗同序随机数）。

    所有分组均无有效不确定度时退化为单轮确定性计算（各轮必然相同）。
    """
    fix = frozenset(fix)
    active = {g: (g not in fix) and _group_active(model, g) for g in GROUPS}
    n_eff = int(model["n_samples"]) if any(active.values()) else 1
    rng = np.random.default_rng(model["seed"])
    # 固定抽取顺序：常量/零位 → 量化 → 重复性；fix 只置零扰动，不改变随机流
    z_const = rng.standard_normal((4, n_eff))
    counts = [len(z.readings) for z in zones]
    quant = rng.uniform(-0.5, 0.5, (n_eff, sum(counts)))
    repn = rng.standard_normal((n_eff, sum(counts)))

    sd_d = model["diameter_std_mm"]
    sd_r = model["radius_of_curvature_std_mm"]
    sd_l = model["wavelength_std_nm"]
    sd_o = model["instrument_offset_std_mm"]
    q = model["knife_resolution_mm"]
    rep_mm = model["zone_repeatability_mm"]

    n_zones = len(zones)
    conic = np.empty(n_eff)
    rms_nm = np.empty(n_eff)
    pv_nm = np.empty(n_eff)
    rms_w = np.empty(n_eff)
    pv_w = np.empty(n_eff)
    strehl = np.empty(n_eff)
    oob = np.zeros((n_eff, n_zones), dtype=bool)
    zone_resid = np.empty((n_eff, n_zones)) if want_zone_residuals else None

    for k in range(n_eff):
        diameter = constants.diameter
        roc = constants.radius_of_curvature
        lam = constants.wavelength_nm
        offset = constants.instrument_offset
        if active["diameter"]:
            diameter += z_const[0, k] * sd_d
        if active["radius_of_curvature"]:
            roc += z_const[1, k] * sd_r
        if active["wavelength"]:
            lam += z_const[2, k] * sd_l
        if active["instrument_offset"]:
            offset += z_const[3, k] * sd_o
        ck = Constants(
            diameter=diameter,
            radius_of_curvature=roc,
            conic_constant=constants.conic_constant,
            wavelength_nm=lam,
            source_mode=constants.source_mode,
            instrument_offset=offset,
        )
        perturbed = []
        idx = 0
        for i, z in enumerate(zones):
            vals = list(z.readings)
            m = len(vals)
            if active["quantization"] or active["repeatability"]:
                noise = np.zeros(m)
                if active["quantization"]:
                    noise += quant[k, idx : idx + m] * q
                if active["repeatability"]:
                    noise += repn[k, idx : idx + m] * rep_mm[i]
                vals = (np.asarray(vals) + noise).tolist()
            idx += m
            perturbed.append(ZoneInput(inner=z.inner, outer=z.outer, readings=vals))
        try:
            res = reduce_test(ck, perturbed, options)
        except AnalysisError as exc:
            raise UncertaintyError(
                f"第 {k + 1} 轮抽样产生非法参数，无法还原：{exc}"
                "（请检查标准不确定度是否相对标称值过大）"
            ) from None
        s = res["summary"]
        conic[k] = res["fit"]["best_fit_conic_constant"]
        rms_nm[k] = s["wavefront_rms_nm"]
        pv_nm[k] = s["wavefront_pv_nm"]
        rms_w[k] = s["wavefront_rms_waves"]
        pv_w[k] = s["wavefront_pv_waves"]
        strehl[k] = s["strehl"]
        oob[k] = [z["out_of_band"] for z in res["zones"]]
        if want_zone_residuals:
            zone_resid[k] = [z["wavefront_nm"] / 2.0 for z in res["zones"]]

    return {
        "n_eff": n_eff,
        "conic": conic,
        "rms_nm": rms_nm,
        "pv_nm": pv_nm,
        "rms_waves": rms_w,
        "pv_waves": pv_w,
        "strehl": strehl,
        "oob": oob,
        "zone_residuals_nm": zone_resid,
    }


def _percentile_block(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=float)
    p5, p50, p95 = (float(v) for v in np.percentile(arr, [5.0, 50.0, 95.0]))
    return {
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
    }


def run_uncertainty(
    constants: Constants,
    zones: list[ZoneInput],
    options: Options,
    model: dict,
) -> dict:
    """完整不确定度传播：主抽样 + 逐项固定输入重算的方差贡献。

    方差贡献：固定分组 g 重算方差 Var_{-g}，贡献 = (Var_full − Var_{-g})
    / Var_full（公共随机数法，负值截为 0）；零不确定度分组直接为 0，
    不重复抽样。对波前 RMS（nm）与 Strehl 两个输出分别给出——波长等
    分组只影响波数/Strehl 口径，单一输出会漏报其贡献。
    """
    full = _mc_run(constants, zones, options, model)
    n_eff = full["n_eff"]
    metrics = {
        "best_fit_conic_constant": _percentile_block(full["conic"]),
        "wavefront_rms_nm": _percentile_block(full["rms_nm"]),
        "wavefront_pv_nm": _percentile_block(full["pv_nm"]),
        "wavefront_rms_waves": _percentile_block(full["rms_waves"]),
        "wavefront_pv_waves": _percentile_block(full["pv_waves"]),
        "strehl": _percentile_block(full["strehl"]),
    }
    probabilities = [float(c) / n_eff for c in full["oob"].sum(axis=0)]

    def _var(arr: np.ndarray) -> float:
        return float(arr.var(ddof=1)) if arr.size > 1 else 0.0

    var_full = {
        "wavefront_rms_nm": _var(full["rms_nm"]),
        "strehl": _var(full["strehl"]),
    }
    # 每个非零分组只做一次重算，两个输出共用
    sub_vars: dict[str, dict[str, float] | None] = {}
    for g in GROUPS:
        if not _group_active(model, g):
            sub_vars[g] = None
            continue
        sub = _mc_run(constants, zones, options, model, fix={g})
        sub_vars[g] = {
            "wavefront_rms_nm": _var(sub["rms_nm"]),
            "strehl": _var(sub["strehl"]),
        }
    contributions = {}
    for output in ("wavefront_rms_nm", "strehl"):
        vf = var_full[output]
        groups = {}
        for g in GROUPS:
            sv = sub_vars[g]
            if vf <= 0.0 or sv is None:
                groups[g] = 0.0
            else:
                groups[g] = max(0.0, vf - sv[output]) / vf
        contributions[output] = {"total_variance": vf, "groups": groups}
    return {
        "metrics": metrics,
        "zone_out_of_band_probability": probabilities,
        "variance_contributions": contributions,
    }


def sample_zone_residuals(
    constants: Constants,
    zones: list[ZoneInput],
    options: Options,
    model: dict,
) -> list[list[float]]:
    """逐轮抽样的分区残余面形误差（nm，= 分区残余波前/2），供修正量搜索
    按置信分位评估候选；使用版本冻结的模型与种子，重复计算结果一致。"""
    out = _mc_run(constants, zones, options, model, want_zone_residuals=True)
    return [[float(v) for v in row] for row in out["zone_residuals_nm"]]
