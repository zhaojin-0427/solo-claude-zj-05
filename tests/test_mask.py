"""Couder 遮罩：布局几何、版本化、刀口位移预测、边界搜索、SVG、批次冻结。"""
import math

import pytest
from fastapi.testclient import TestClient

from foucault.app import create_app

MASK = {
    "name": "200mm f/4 遮罩",
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


@pytest.fixture()
def client(tmp_path):
    app = create_app(str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def create_scheme(client, **kw):
    payload = {**MASK, **kw}
    resp = client.post("/api/mask-schemes", json=payload)
    assert resp.status_code == 201, resp.json()
    return resp.json()


def version_detail(client, scheme_id, version_no=1):
    resp = client.get(f"/api/mask-schemes/{scheme_id}/versions/{version_no}")
    assert resp.status_code == 200, resp.json()
    return resp.json()


# ---------------- 布局几何 ----------------

def test_equal_area_layout(client):
    sid = create_scheme(client)["scheme"]["id"]
    lay = version_detail(client, sid)["layout"]
    zones = lay["zones"]
    assert len(zones) == 5
    # 等面积：各环带面积相等，边界从禁测半径到镜面边缘连续覆盖
    areas = [z["area"] for z in zones]
    assert max(areas) == pytest.approx(min(areas), rel=1e-9)
    assert zones[0]["inner_radius"] == pytest.approx(25.0)
    assert zones[-1]["outer_radius"] == pytest.approx(100.0)
    for a, b in zip(zones, zones[1:]):
        assert a["outer_radius"] == pytest.approx(b["inner_radius"])
    # 等效半径 = 面积均分半径
    for z in zones:
        assert z["r_mean"] == pytest.approx(
            math.sqrt((z["inner_radius"] ** 2 + z["outer_radius"] ** 2) / 2)
        )
    # 开窗在环带内、以等效半径为中心，左右开窗关于中心线对称
    for z in zones:
        assert z["inner_radius"] - 1e-9 <= z["y_bottom"]
        assert z["y_top"] <= z["outer_radius"] + 1e-9
        assert (z["y_bottom"] + z["y_top"]) / 2 == pytest.approx(z["r_mean"])
        assert z["window_left"]["x_from"] == pytest.approx(-z["window_right"]["x_to"])
        assert z["window_left"]["x_to"] == pytest.approx(-z["window_right"]["x_from"])
        assert z["window_right"]["x_to"] <= z["outer_radius"]
    # 相邻开窗间距（桥）满足桥宽
    for br in lay["bridges"]:
        assert br["gap"] >= MASK["bridge_width"] - 1e-6


def test_custom_weights_layout(client):
    body = create_scheme(
        client, weighting="custom", weights=[3.0, 2.0, 2.0, 1.5, 1.5], bridge_width=7.0
    )
    lay = version_detail(client, body["scheme"]["id"])["layout"]
    areas = [z["area"] for z in lay["zones"]]
    total_w = 10.0
    for a, w in zip(areas, [3.0, 2.0, 2.0, 1.5, 1.5]):
        assert a == pytest.approx(w / total_w * sum(areas), rel=1e-9)


def test_inch_unit_normalized_to_mm(client):
    IN = 25.4
    body = create_scheme(
        client,
        unit="in",
        diameter=200.0 / IN,
        radius_of_curvature=1600.0 / IN,
        center_exclusion_radius=25.0 / IN,
        min_zone_width=5.0 / IN,
        bridge_width=6.0 / IN,
        knife_resolution=0.5 / IN,
    )
    params = version_detail(client, body["scheme"]["id"])["params"]
    assert params["diameter"] == pytest.approx(200.0)
    assert params["bridge_width"] == pytest.approx(6.0)


# ---------------- 拒绝生成（指明环带） ----------------

def reject_errors(client, **kw):
    resp = client.post("/api/mask-schemes", json={**MASK, **kw})
    assert resp.status_code == 422, resp.json()
    return resp.json()["detail"]["errors"]


def test_reject_out_of_bounds(client):
    errors = reject_errors(client, center_exclusion_radius=100.0)
    assert any("分区越界" in e for e in errors)


def test_reject_min_zone_width(client):
    # 等环宽 5mm < 最小环宽 10mm，逐环带指明
    errors = reject_errors(
        client,
        weighting="equal_width",
        center_exclusion_radius=90.0,
        zone_count=2,
        bridge_width=2.0,
        min_zone_width=10.0,
    )
    assert any("环宽不足" in e and "环带 0" in e for e in errors)
    assert any("环宽不足" in e and "环带 1" in e for e in errors)


def test_reject_no_bridge(client):
    # 环宽 5mm ≤ 桥宽 6mm
    errors = reject_errors(
        client,
        weighting="equal_width",
        center_exclusion_radius=90.0,
        zone_count=2,
        bridge_width=6.0,
        min_zone_width=2.0,
    )
    assert any("结构无法留桥" in e and "环带 0" in e for e in errors)


def test_reject_window_overlap(client):
    # 内环带过宽而桥宽太小：开窗顶边越出环带外边界
    errors = reject_errors(
        client,
        zone_count=3,
        center_exclusion_radius=20.0,
        bridge_width=1.0,
        min_zone_width=2.0,
    )
    assert any("开窗相交" in e and "环带 0" in e for e in errors)


def test_reject_bad_weights(client):
    errors = reject_errors(client, weighting="custom", weights=[1.0, 2.0])
    assert any("权重" in e for e in errors)
    errors = reject_errors(
        client, weighting="custom", weights=[1.0, 2.0, 0.0, 1.0, 1.0]
    )
    assert any("权重" in e for e in errors)


def test_reject_bad_constants(client):
    resp = client.post("/api/mask-schemes", json={**MASK, "diameter": -200.0})
    assert resp.status_code == 422
    resp = client.post("/api/mask-schemes", json={**MASK, "source_mode": "laser"})
    assert resp.status_code == 422


# ---------------- 刀口位移预测 ----------------

def test_knife_prediction_fixed_source(client):
    sid = create_scheme(client)["scheme"]["id"]
    pred = version_detail(client, sid)["layout"]["prediction"]
    assert pred["reference"] == "innermost"
    zones = pred["zones"]
    # 固定光源：刀口位移 = LA_ideal = -K·r²/R；零点在最内环带
    for row in zones:
        rm = row["r_mean"]
        assert row["la_ideal_mm"] == pytest.approx(rm**2 / 1600.0)
        assert row["knife_reading_mm"] == pytest.approx(row["la_ideal_mm"])
    assert zones[0]["knife_relative_mm"] == pytest.approx(0.0)
    # 相对值单调递增（抛物面 LA 随 r² 增长）
    rel = [row["knife_relative_mm"] for row in zones]
    assert rel == sorted(rel)
    # 相邻区对读数差与分辨率比较
    for a in pred["adjacent"]:
        i, j = a["pair"]
        assert a["delta_mm"] == pytest.approx(abs(rel[j] - rel[i]))
        assert a["resolvable"] == (a["delta_mm"] >= 0.5 - 1e-9)
    assert pred["unresolvable_pairs"] == []  # 5 区固定光源，差值均 > 1mm


def test_knife_prediction_moving_source_flags_pairs(client):
    # 移动光源读数差减半 + 10 区：相邻差低于 0.5mm 分辨率的区对被标出
    body = create_scheme(
        client,
        source_mode="moving",
        zone_count=10,
        bridge_width=2.0,
        min_zone_width=2.0,
    )
    pred = version_detail(client, body["scheme"]["id"])["layout"]["prediction"]
    for row in pred["zones"]:
        assert row["knife_reading_mm"] == pytest.approx(row["la_ideal_mm"] / 2.0)
    assert pred["unresolvable_pairs"], "应标出低于分辨率的相邻区对"
    for pair in pred["unresolvable_pairs"]:
        i, j = pair
        delta = abs(
            pred["zones"][j]["knife_relative_mm"]
            - pred["zones"][i]["knife_relative_mm"]
        )
        assert delta < 0.5


# ---------------- 版本化 ----------------

def test_mask_versioning_and_idempotency(client):
    body = create_scheme(client)
    sid = body["scheme"]["id"]
    assert body["version"]["version_no"] == 1
    assert body["version"]["reused"] is False
    # 相同参数 → 复用最新版本
    resp = client.post(
        f"/api/mask-schemes/{sid}/versions",
        json={k: v for k, v in MASK.items() if k != "name"},
    )
    assert resp.json()["version"]["reused"] is True
    assert resp.json()["version"]["version_no"] == 1
    # 修改制作限制 → 新版本，冻结各自参数
    resp = client.post(
        f"/api/mask-schemes/{sid}/versions",
        json={**{k: v for k, v in MASK.items() if k != "name"}, "bridge_width": 5.0},
    )
    assert resp.json()["version"]["version_no"] == 2
    v1 = version_detail(client, sid, 1)
    v2 = version_detail(client, sid, 2)
    assert v1["params"]["bridge_width"] == pytest.approx(6.0)
    assert v2["params"]["bridge_width"] == pytest.approx(5.0)
    assert v1["params_hash"] != v2["params_hash"]
    # 方案列表与详情
    schemes = client.get("/api/mask-schemes").json()["schemes"]
    assert schemes[0]["n_versions"] == 2
    detail = client.get(f"/api/mask-schemes/{sid}").json()["scheme"]
    assert [v["version_no"] for v in detail["versions"]] == [1, 2]
    # 版本不可变：重复读取一致
    assert version_detail(client, sid, 1) == v1


def test_mask_not_found(client):
    assert client.get("/api/mask-schemes/999").status_code == 404
    assert client.get("/api/mask-schemes/999/versions/1").status_code == 404
    body = create_scheme(client)
    sid = body["scheme"]["id"]
    assert client.get(f"/api/mask-schemes/{sid}/versions/9").status_code == 404


# ---------------- SVG ----------------

def test_svg_one_to_one_with_dims_and_ruler(client):
    body = create_scheme(client, print_scale=1.02)
    sid = body["scheme"]["id"]
    resp = client.get(f"/api/mask-schemes/{sid}/versions/1/svg")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    svg = resp.text
    assert "校准尺" in svg  # 校准尺
    assert "⌀" in svg  # 直径尺寸线
    assert 'marker-start="url(#arr)"' in svg  # 尺寸线箭头
    # 1:1：width 以 mm 声明且与 viewBox 一致；打印缩放 1.02 已计入几何
    import xml.etree.ElementTree as ET

    root = ET.fromstring(svg)
    w = float(root.attrib["width"].removesuffix("mm"))
    vb_w = float(root.attrib["viewBox"].split()[2])
    assert w == pytest.approx(vb_w)
    ns = {"s": "http://www.w3.org/2000/svg"}
    radii = [float(el.attrib["r"]) for el in root.findall("s:circle", ns)]
    assert max(radii) == pytest.approx(100.0 * 1.02)  # 镜面半径 × 打印缩放


# ---------------- 边界搜索 ----------------

SEARCH = {
    "diameter": 200.0,
    "radius_of_curvature": 1600.0,
    "conic_constant": -1.0,
    "source_mode": "fixed",
    "unit": "mm",
    "center_exclusion_radius": 25.0,
    "min_zone_width": 4.0,
    "knife_resolution": 0.5,
    "zone_count_min": 4,
    "zone_count_max": 8,
    "bridge_width_min": 3.0,
    "bridge_width_max": 7.0,
    "bridge_width_step": 2.0,
    "top": 5,
}


def test_search_ranked_and_feasible(client):
    resp = client.post("/api/mask-schemes/search", json=SEARCH)
    assert resp.status_code == 200, resp.json()
    data = resp.json()
    assert data["evaluated"] == 5 * 3 * 2  # 分区数 × 桥宽 × 权重模式
    assert data["feasible"] > 0
    cands = data["candidates"]
    assert len(cands) <= 5
    # 排序：最小可分辨位移降序（同值再比面积均衡度、制作余量）
    keys = [
        (
            -c["metrics"]["min_resolvable_delta_mm"],
            -c["metrics"]["area_balance"],
            -c["metrics"]["fabrication_margin_mm"],
        )
        for c in cands
    ]
    assert keys == sorted(keys)
    assert [c["rank"] for c in cands] == list(range(len(cands)))
    # 候选均可行：环宽、桥宽、开窗约束满足
    for c in cands:
        m = c["metrics"]
        assert m["min_zone_width_mm"] >= 4.0 - 1e-9
        assert m["min_window_height_mm"] > 0
        assert m["min_bridge_gap_mm"] >= c["bridge_width"] - 1e-6
    # 候选参数可直接建版本
    p = dict(cands[0]["params"])
    resp = client.post("/api/mask-schemes", json=p)
    assert resp.status_code == 201, resp.json()


def test_search_rejects_bad_range(client):
    resp = client.post(
        "/api/mask-schemes/search", json={**SEARCH, "zone_count_max": 2, "zone_count_min": 5}
    )
    assert resp.status_code == 422
    resp = client.post(
        "/api/mask-schemes/search", json={**SEARCH, "center_exclusion_radius": 100.0}
    )
    assert resp.status_code == 422


# ---------------- 遮罩版本 → 测试批次 ----------------

def la_readings_for(zones, n=3):
    out = []
    for z in zones:
        rm = z["r_mean"]
        base = 0.5 + rm**2 / 1600.0
        out.append([round(base + 0.003 * j, 6) for j in range(n)])
    return out


def test_create_test_from_mask_version(client):
    sid = create_scheme(client)["scheme"]["id"]
    lay = version_detail(client, sid)["layout"]
    resp = client.post(
        "/api/tests",
        json={
            "name": "遮罩批次",
            "mask_scheme_id": sid,
            "mask_version_no": 1,
            "wavelength_nm": 550.0,
            "zone_readings": la_readings_for(lay["zones"]),
        },
    )
    assert resp.status_code == 201, resp.json()
    body = resp.json()
    # 批次冻结遮罩版本 ID 与环带边界
    test = body["test"]
    assert test["mask_scheme_id"] == sid
    assert test["mask_version_no"] == 1
    assert len(test["zones"]) == len(lay["zones"])
    for tz, mz in zip(test["zones"], lay["zones"]):
        assert tz["inner_radius"] == pytest.approx(mz["inner_radius"])
        assert tz["outer_radius"] == pytest.approx(mz["outer_radius"])
    # 常量取自遮罩冻结值，分析自动生成
    assert test["diameter"] == pytest.approx(200.0)
    assert test["source_mode"] == "fixed"
    assert body["version"]["summary"]["wavefront_rms_waves"] < 0.01
    # 遮罩另建版本不改写既有分析
    tid = test["id"]
    v1_before = client.get(f"/api/tests/{tid}/versions/1").json()
    client.post(
        f"/api/mask-schemes/{sid}/versions",
        json={**{k: v for k, v in MASK.items() if k != "name"}, "bridge_width": 5.0},
    )
    v1_after = client.get(f"/api/tests/{tid}/versions/1").json()
    assert v1_before == v1_after


def test_create_test_from_mask_latest_version_default(client):
    sid = create_scheme(client)["scheme"]["id"]
    client.post(
        f"/api/mask-schemes/{sid}/versions",
        json={**{k: v for k, v in MASK.items() if k != "name"}, "bridge_width": 5.0},
    )
    lay = version_detail(client, sid, 2)["layout"]
    resp = client.post(
        "/api/tests",
        json={
            "mask_scheme_id": sid,  # 缺省取最新版本 v2
            "wavelength_nm": 550.0,
            "zone_readings": la_readings_for(lay["zones"]),
        },
    )
    assert resp.status_code == 201, resp.json()
    assert resp.json()["test"]["mask_version_no"] == 2


def test_create_test_from_mask_validation(client):
    sid = create_scheme(client)["scheme"]["id"]
    lay = version_detail(client, sid)["layout"]
    readings = la_readings_for(lay["zones"])
    # 常量与遮罩冻结值不一致
    resp = client.post(
        "/api/tests",
        json={
            "mask_scheme_id": sid,
            "wavelength_nm": 550.0,
            "diameter": 300.0,
            "zone_readings": readings,
        },
    )
    assert resp.status_code == 422
    assert "冻结" in str(resp.json()["detail"])
    # zone_readings 数量不符
    resp = client.post(
        "/api/tests",
        json={
            "mask_scheme_id": sid,
            "wavelength_nm": 550.0,
            "zone_readings": readings[:-1],
        },
    )
    assert resp.status_code == 422
    # 缺 zone_readings
    resp = client.post(
        "/api/tests", json={"mask_scheme_id": sid, "wavelength_nm": 550.0}
    )
    assert resp.status_code == 422
    # 读数不足（每区至少 2 条）
    resp = client.post(
        "/api/tests",
        json={
            "mask_scheme_id": sid,
            "wavelength_nm": 550.0,
            "zone_readings": [[r[0]] for r in readings],
        },
    )
    assert resp.status_code == 422
    assert "有效读数不足" in str(resp.json()["detail"])
    # 遮罩版本不存在
    resp = client.post(
        "/api/tests",
        json={
            "mask_scheme_id": 999,
            "wavelength_nm": 550.0,
            "zone_readings": readings,
        },
    )
    assert resp.status_code == 404


def test_create_test_missing_fields_without_mask(client):
    resp = client.post("/api/tests", json={"wavelength_nm": 550.0})
    assert resp.status_code == 422
    assert "缺少必填字段" in str(resp.json()["detail"])
