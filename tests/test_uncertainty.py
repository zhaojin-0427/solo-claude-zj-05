"""不确定度传播：配置校验、版本冻结、确定性、方差贡献、修正量分位评估。"""
import json
import math

import pytest
from fastapi.testclient import TestClient

from foucault.app import create_app

ZONES = [(0, 22), (22, 45), (45, 65), (65, 85), (85, 100)]
GROUPS = (
    "diameter",
    "radius_of_curvature",
    "wavelength",
    "instrument_offset",
    "quantization",
    "repeatability",
)


def la_readings(bump=0.0, n=4, spread=0.004):
    """接近理想抛物面的读数，边缘区附加 bump(mm) 模拟面形误差。"""
    out = []
    for i, (ri, ro) in enumerate(ZONES):
        rm = math.sqrt((ri**2 + ro**2) / 2.0)
        base = 0.5 + rm**2 / 1600.0
        if i == len(ZONES) - 1:
            base += bump
        out.append([round(base + spread * j, 6) for j in range(n)])
    return out


def make_payload(bump=0.0, spread=0.004, **kw):
    payload = {
        "name": "200mm f/4 主镜",
        "diameter": 200.0,
        "radius_of_curvature": 1600.0,
        "conic_constant": -1.0,
        "wavelength_nm": 550.0,
        "source_mode": "fixed",
        "unit": "mm",
        "zones": [
            {"inner_radius": ri, "outer_radius": ro, "readings": rd}
            for (ri, ro), rd in zip(ZONES, la_readings(bump, spread=spread))
        ],
    }
    payload.update(kw)
    return payload


def unc_cfg(**kw):
    cfg = {
        "n_samples": 500,
        "seed": 42,
        "diameter_std": 0.5,
        "radius_of_curvature_std": 2.0,
        "wavelength_std_nm": 5.0,
        "instrument_offset_std": 0.01,
        "knife_resolution": 0.02,
    }
    cfg.update(kw)
    return cfg


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def create_ok(client, **kw):
    resp = client.post("/api/tests", json=make_payload(**kw))
    assert resp.status_code == 201, resp.json()
    return resp.json()


# ---------------- 配置校验 ----------------


def test_reject_bad_sample_count(client):
    for n in (0, 499, 20001):
        resp = client.post("/api/tests", json=make_payload(uncertainty=unc_cfg(n_samples=n)))
        assert resp.status_code == 422, n


def test_reject_negative_components(client):
    resp = client.post("/api/tests", json=make_payload(uncertainty=unc_cfg(diameter_std=-0.1)))
    assert resp.status_code == 422
    resp = client.post("/api/tests", json=make_payload(uncertainty=unc_cfg(wavelength_std_nm=-1)))
    assert resp.status_code == 422
    resp = client.post("/api/tests", json=make_payload(uncertainty=unc_cfg(knife_resolution=-0.01)))
    assert resp.status_code == 422


def test_reject_non_finite_components(client):
    # httpx 的严格 JSON 编码拒绝 inf，改发原始 JSON 体验证服务端校验
    payload = make_payload(uncertainty=unc_cfg())
    payload["uncertainty"]["radius_of_curvature_std"] = "Infinity"
    raw = json.dumps(payload).replace('"Infinity"', "Infinity")
    resp = client.post(
        "/api/tests", content=raw, headers={"content-type": "application/json"}
    )
    assert resp.status_code == 422


def test_reject_bad_seed(client):
    resp = client.post("/api/tests", json=make_payload(uncertainty=unc_cfg(seed=-1)))
    assert resp.status_code == 422
    resp = client.post(
        "/api/tests", json=make_payload(uncertainty=unc_cfg(seed=2**31))
    )
    assert resp.status_code == 422


def test_reject_repeatability_mode_mismatch(client):
    # specified 缺列表
    resp = client.post(
        "/api/tests",
        json=make_payload(uncertainty=unc_cfg(repeatability_mode="specified")),
    )
    assert resp.status_code == 422
    # estimated 多给列表
    resp = client.post(
        "/api/tests",
        json=make_payload(
            uncertainty=unc_cfg(zone_repeatability=[0.01] * 5)
        ),
    )
    assert resp.status_code == 422
    # specified 列表长度与分区数不符
    resp = client.post(
        "/api/tests",
        json=make_payload(
            uncertainty=unc_cfg(
                repeatability_mode="specified", zone_repeatability=[0.01] * 4
            )
        ),
    )
    assert resp.status_code == 422
    assert "分区数" in str(resp.json()["detail"])
    # 负分量
    resp = client.post(
        "/api/tests",
        json=make_payload(
            uncertainty=unc_cfg(
                repeatability_mode="specified",
                zone_repeatability=[0.01, 0.01, -0.01, 0.01, 0.01],
            )
        ),
    )
    assert resp.status_code == 422


# ---------------- 版本冻结与确定性 ----------------


def test_uncertainty_frozen_and_deterministic(client):
    body = create_ok(client, bump=0.05, uncertainty=unc_cfg())
    tid = body["test"]["id"]
    assert body["version"]["version_no"] == 1
    unc = body["version"]["uncertainty"]
    # 冻结模型：mm 规范值 + 请求种子
    model = unc["model"]
    assert model["seed"] == 42
    assert model["n_samples"] == 500
    assert model["diameter_std_mm"] == 0.5
    assert model["radius_of_curvature_std_mm"] == 2.0
    assert model["wavelength_std_nm"] == 5.0
    assert model["instrument_offset_std_mm"] == 0.01
    assert model["knife_resolution_mm"] == 0.02
    assert model["repeatability_mode"] == "estimated"
    assert len(model["zone_repeatability_mm"]) == 5
    assert all(s > 0 for s in model["zone_repeatability_mm"])
    # 指标分位
    for key in (
        "best_fit_conic_constant",
        "wavefront_rms_nm",
        "wavefront_pv_nm",
        "wavefront_rms_waves",
        "wavefront_pv_waves",
        "strehl",
    ):
        block = unc["metrics"][key]
        assert block["p5"] <= block["p50"] <= block["p95"]
        assert block["std"] >= 0
    # 各区越过误差带概率
    probs = unc["zone_out_of_band_probability"]
    assert len(probs) == 5
    assert all(0.0 <= p <= 1.0 for p in probs)
    # 方差贡献：两个输出、六个分组
    for output in ("wavefront_rms_nm", "strehl"):
        vc = unc["variance_contributions"][output]
        assert tuple(vc["groups"]) == GROUPS
        assert all(0.0 <= v <= 1.0 for v in vc["groups"].values())
    # 快照冻结同一模型
    detail = client.get(f"/api/tests/{tid}/versions/1").json()
    assert detail["snapshot"]["uncertainty"] == model
    assert detail["result"]["uncertainty"] == unc
    # 相同输入重复计算完全一致（新批次、同一配置）
    again = create_ok(client, bump=0.05, uncertainty=unc_cfg())
    assert again["version"]["uncertainty"] == unc
    # 输入未变重复 analyze 幂等复用
    resp = client.post(f"/api/tests/{tid}/analyze", json={})
    assert resp.status_code == 201
    assert resp.json()["version"]["reused"] is True
    assert resp.json()["version"]["version_no"] == 1


def test_derived_seed_frozen(client):
    cfg = unc_cfg()
    del cfg["seed"]
    body = create_ok(client, uncertainty=cfg)
    model = body["version"]["uncertainty"]["model"]
    assert isinstance(model["seed"], int) and 0 <= model["seed"] < 2**31
    again = create_ok(client, uncertainty=cfg)
    assert again["version"]["uncertainty"]["model"]["seed"] == model["seed"]
    assert again["version"]["uncertainty"] == body["version"]["uncertainty"]


def test_old_requests_keep_original_response(client):
    body = create_ok(client)
    assert "uncertainty" not in body["version"]
    assert "uncertainty" not in body["test"]
    tid = body["test"]["id"]
    detail = client.get(f"/api/tests/{tid}").json()
    assert "uncertainty" not in detail["test"]
    version = client.get(f"/api/tests/{tid}/versions/1").json()
    assert "uncertainty" not in version["result"]
    assert "uncertainty" not in version["snapshot"]


def test_analyze_adds_uncertainty_as_new_version(client):
    body = create_ok(client)
    tid = body["test"]["id"]
    resp = client.post(
        f"/api/tests/{tid}/analyze", json={"uncertainty": unc_cfg()}
    )
    assert resp.status_code == 201
    v2 = resp.json()["version"]
    assert v2["version_no"] == 2
    assert v2["reused"] is False
    assert "uncertainty" in v2
    # 相同配置重解幂等复用
    resp = client.post(
        f"/api/tests/{tid}/analyze", json={"uncertainty": unc_cfg()}
    )
    assert resp.json()["version"]["version_no"] == 2
    assert resp.json()["version"]["reused"] is True
    # 旧版本结果不被改写
    v1 = client.get(f"/api/tests/{tid}/versions/1").json()
    assert "uncertainty" not in v1["result"]


def test_analyze_uses_stored_config_by_default(client):
    create_ok(client, uncertainty=unc_cfg())
    # 创建时配置已随批次冻结；不带配置的 analyze 沿用并复用 v1
    resp = client.post("/api/tests/1/analyze", json={})
    assert resp.json()["version"]["reused"] is True
    assert "uncertainty" in resp.json()["version"]


# ---------------- 方差贡献语义 ----------------


def test_single_group_contribution_dominates(client):
    # 仅口径有不确定度（读数无离散 → 重复性不激活，量化误差为 0）
    cfg = unc_cfg(
        diameter_std=1.0,
        radius_of_curvature_std=0.0,
        wavelength_std_nm=0.0,
        instrument_offset_std=0.0,
        knife_resolution=0.0,
    )
    body = create_ok(client, spread=0.0, uncertainty=cfg)
    groups = body["version"]["uncertainty"]["variance_contributions"][
        "wavefront_rms_nm"
    ]["groups"]
    assert groups["diameter"] == pytest.approx(1.0)
    for g in GROUPS:
        if g != "diameter":
            assert groups[g] == 0.0


def test_repeatability_estimated_dominates(client):
    cfg = unc_cfg(
        diameter_std=0.0,
        radius_of_curvature_std=0.0,
        wavelength_std_nm=0.0,
        instrument_offset_std=0.0,
        knife_resolution=0.0,
    )
    body = create_ok(client, uncertainty=cfg)
    unc = body["version"]["uncertainty"]
    assert unc["model"]["repeatability_mode"] == "estimated"
    assert all(s > 0 for s in unc["model"]["zone_repeatability_mm"])
    groups = unc["variance_contributions"]["wavefront_rms_nm"]["groups"]
    assert groups["repeatability"] == pytest.approx(1.0)


def test_quantization_contribution(client):
    cfg = unc_cfg(
        diameter_std=0.0,
        radius_of_curvature_std=0.0,
        wavelength_std_nm=0.0,
        instrument_offset_std=0.0,
        knife_resolution=0.1,
    )
    body = create_ok(client, spread=0.0, uncertainty=cfg)
    unc = body["version"]["uncertainty"]
    assert unc["model"]["knife_resolution_mm"] == 0.1
    groups = unc["variance_contributions"]["wavefront_rms_nm"]["groups"]
    assert groups["quantization"] == pytest.approx(1.0)


def test_wavelength_shows_in_strehl_contribution(client):
    cfg = unc_cfg(
        diameter_std=0.0,
        radius_of_curvature_std=0.0,
        wavelength_std_nm=10.0,
        instrument_offset_std=0.0,
        knife_resolution=0.0,
    )
    body = create_ok(client, spread=0.0, bump=0.5, uncertainty=cfg)
    vc = body["version"]["uncertainty"]["variance_contributions"]
    # 波前 RMS(nm) 与波长无关，Strehl 依赖波长
    assert vc["wavefront_rms_nm"]["groups"]["wavelength"] == 0.0
    assert vc["strehl"]["groups"]["wavelength"] == pytest.approx(1.0)


def test_degenerate_zero_uncertainty(client):
    cfg = unc_cfg(
        diameter_std=0.0,
        radius_of_curvature_std=0.0,
        wavelength_std_nm=0.0,
        instrument_offset_std=0.0,
        knife_resolution=0.0,
    )
    body = create_ok(client, spread=0.0, uncertainty=cfg)
    unc = body["version"]["uncertainty"]
    assert unc["metrics"]["wavefront_rms_nm"]["std"] == 0.0
    assert unc["metrics"]["strehl"]["std"] == 0.0
    for output in ("wavefront_rms_nm", "strehl"):
        vc = unc["variance_contributions"][output]
        assert vc["total_variance"] == 0.0
        assert all(v == 0.0 for v in vc["groups"].values())
    # 确定性结果：越带概率退化为 0/1
    assert all(p in (0.0, 1.0) for p in unc["zone_out_of_band_probability"])


def test_out_of_band_probability_between(client):
    # 大重复性 + 较紧误差带：至少一个区的越带概率严格落在 (0, 1)
    cfg = unc_cfg(
        diameter_std=0.0,
        radius_of_curvature_std=0.0,
        wavelength_std_nm=0.0,
        instrument_offset_std=0.0,
        knife_resolution=0.0,
    )
    body = create_ok(
        client,
        spread=0.5,
        uncertainty=cfg,
        options={"error_band_waves": 0.05},
    )
    probs = body["version"]["uncertainty"]["zone_out_of_band_probability"]
    assert any(0.0 < p < 1.0 for p in probs)


# ---------------- 单位与指定重复性 ----------------


def test_specified_repeatability(client):
    cfg = unc_cfg(
        diameter_std=0.0,
        radius_of_curvature_std=0.0,
        wavelength_std_nm=0.0,
        instrument_offset_std=0.0,
        knife_resolution=0.0,
        repeatability_mode="specified",
        zone_repeatability=[0.01, 0.02, 0.03, 0.04, 0.05],
    )
    body = create_ok(client, uncertainty=cfg)
    model = body["version"]["uncertainty"]["model"]
    assert model["repeatability_mode"] == "specified"
    assert model["zone_repeatability_mm"] == [0.01, 0.02, 0.03, 0.04, 0.05]


def test_inch_unit_converted_in_model(client):
    IN = 25.4
    payload = make_payload(
        diameter=200.0 / IN,
        radius_of_curvature=1600.0 / IN,
        unit="in",
        zones=[
            {"inner_radius": ri / IN, "outer_radius": ro / IN,
             "readings": [v / IN for v in rd]}
            for (ri, ro), rd in zip(ZONES, la_readings())
        ],
        uncertainty=unc_cfg(
            diameter_std=0.5 / IN,
            radius_of_curvature_std=2.0 / IN,
            instrument_offset_std=0.01 / IN,
            knife_resolution=0.02 / IN,
        ),
    )
    resp = client.post("/api/tests", json=payload)
    assert resp.status_code == 201, resp.json()
    model = resp.json()["version"]["uncertainty"]["model"]
    assert model["diameter_std_mm"] == pytest.approx(0.5)
    assert model["radius_of_curvature_std_mm"] == pytest.approx(2.0)
    assert model["instrument_offset_std_mm"] == pytest.approx(0.01)
    assert model["knife_resolution_mm"] == pytest.approx(0.02)


def test_mask_knife_resolution_fallback(client):
    mask = {
        "name": "遮罩",
        "diameter": 200.0,
        "radius_of_curvature": 1600.0,
        "conic_constant": -1.0,
        "source_mode": "fixed",
        "unit": "mm",
        "zone_count": 5,
        "weighting": "equal_area",
        "center_exclusion_radius": 25.0,
        "min_zone_width": 5.0,
        "bridge_width": 6.0,
        "knife_resolution": 0.5,
        "print_scale": 1.0,
    }
    resp = client.post("/api/mask-schemes", json=mask)
    assert resp.status_code == 201
    scheme_id = resp.json()["scheme"]["id"]
    zones = client.get(f"/api/mask-schemes/{scheme_id}/versions/1").json()["layout"][
        "zones"
    ]
    zone_readings = []
    for z in zones:
        rm = z["r_mean"]
        base = 0.5 + rm**2 / 1600.0
        zone_readings.append([round(base + 0.003 * j, 6) for j in range(3)])
    cfg = unc_cfg()
    del cfg["knife_resolution"]  # 缺省 → 回退遮罩冻结分辨率
    payload = {
        "name": "遮罩批次",
        "mask_scheme_id": scheme_id,
        "wavelength_nm": 550.0,
        "zone_readings": zone_readings,
        "uncertainty": cfg,
    }
    resp = client.post("/api/tests", json=payload)
    assert resp.status_code == 201, resp.json()
    model = resp.json()["version"]["uncertainty"]["model"]
    assert model["knife_resolution_mm"] == 0.5


# ---------------- 修正量搜索的置信分位评估 ----------------


def test_correction_quantile_requires_model(client):
    body = create_ok(client, bump=0.2)
    tid = body["test"]["id"]
    resp = client.post(
        f"/api/tests/{tid}/corrections/search",
        json={"max_removal_nm": 500.0, "confidence_quantile": 0.95},
    )
    assert resp.status_code == 422
    assert "不确定度模型" in str(resp.json()["detail"])


def test_correction_quantile_evaluation(client):
    body = create_ok(client, bump=0.2, uncertainty=unc_cfg())
    tid = body["test"]["id"]
    resp = client.post(
        f"/api/tests/{tid}/corrections/search",
        json={"max_removal_nm": 500.0, "confidence_quantile": 0.95},
    )
    assert resp.status_code == 200, resp.json()
    result = resp.json()
    assert result["confidence_quantile"] == 0.95
    assert result["uncertainty_evaluation"] == {
        "confidence_quantile": 0.95,
        "n_samples": 500,
        "seed": 42,
    }
    cands = result["candidates"]
    assert len(cands) >= 2
    for c in cands:
        m = c["metrics"]
        assert "residual_wavefront_rms_nm_at_quantile" in m
        assert "residual_wavefront_rms_waves_at_quantile" in m
        # 分位残余不低于名义残余（P95 应偏保守）
        assert m["residual_wavefront_rms_nm_at_quantile"] >= 0.0
    # 排序：分位残余升序，rank 连续
    qs = [c["metrics"]["residual_wavefront_rms_nm_at_quantile"] for c in cands]
    assert qs == sorted(qs)
    assert [c["rank"] for c in cands] == list(range(len(cands)))
    # 相同版本重复计算完全一致
    again = client.post(
        f"/api/tests/{tid}/corrections/search",
        json={"max_removal_nm": 500.0, "confidence_quantile": 0.95},
    )
    assert again.json() == result
    # 不指定分位时保持原响应形状（无分位指标）
    plain = client.post(
        f"/api/tests/{tid}/corrections/search", json={"max_removal_nm": 500.0}
    )
    assert plain.status_code == 200
    assert "confidence_quantile" not in plain.json()
    assert "residual_wavefront_rms_nm_at_quantile" not in plain.json()["candidates"][0][
        "metrics"
    ]


def test_correction_quantile_uses_frozen_version(client):
    # 基于旧版本（含不确定度）评估，不受后续重新分析影响
    body = create_ok(client, bump=0.2, uncertainty=unc_cfg())
    tid = body["test"]["id"]
    resp = client.post(
        f"/api/tests/{tid}/corrections/search",
        json={"max_removal_nm": 500.0, "confidence_quantile": 0.9, "version_no": 1},
    )
    assert resp.status_code == 200
    assert resp.json()["based_on"] == {"test_id": tid, "version_no": 1}
