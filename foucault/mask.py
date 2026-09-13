"""Couder 遮罩（分区光阑）设计：环带划分、开窗几何、刀口位移预测与边界搜索。

几何约定（与 README 光学模型一致，模块内部一律 mm）：
- 环带为 [r0, rim] 上的连续同心环，r0 = 中心禁测半径，rim = 口径 / 2。
- 分区权重：equal_area（等面积）/ equal_width（等环宽）/ custom（自定义权重，
  各环带面积正比于权重）。
- 环带等效半径 r_m = sqrt((r_in² + r_out²) / 2)（面积均分半径，与分析模型一致）。
- 每环带开左右两个矩形窗：窗竖直方向以 r_m 为中心，窗高在满足
  "相邻开窗间距 ≥ 桥宽" 且 "窗高 ≤ 环宽 - 桥宽" 的约束下取最大
  （投影法求解）；窗水平方向由环带边界的弦长决定
  （x_inner = sqrt(max(0, r_in² - y1²))，x_outer = sqrt(max(0, r_out² - y2²))，
  y1/y2 为窗底/顶边高度）。内环带的窗底边高于 r_in 时 x_inner = 0，
  左右开窗在中心线处相连（梯式遮罩），属正常几何。

拒绝生成（MaskError，错误逐条指明相关环带）：
- 分区越界：中心禁测半径不小于镜面半径、权重非法等；
- 环宽不足：环宽 < 最小环宽；
- 结构无法留桥：环宽 ≤ 桥宽、相邻环带等效半径间距 < 桥宽，
  或留桥约束下开窗高度被压为零；
- 开窗相交：开窗顶/底边越出本环带边界（侵入相邻环带），或开窗水平宽度为零。

刀口位移预测：LA_ideal(r_m) = -K · r_m² / R（固定光源等效），移动光源的
刀口位移减半（复用 optics.SOURCE_FACTOR / optics.ideal_la）；以最内环带为零点，
相邻环带读数差低于刀口尺分辨率的区对会被标出。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from xml.sax.saxutils import escape

from .optics import SOURCE_FACTOR, ideal_la

TOL = 1e-9
WEIGHTING_MODES = ("equal_area", "equal_width", "custom")


class MaskError(Exception):
    """遮罩设计校验失败；errors 逐条指明相关环带。"""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = list(errors)


@dataclass
class MaskParams:
    """冻结进遮罩版本的全部参数（mm）。"""

    diameter: float
    radius_of_curvature: float
    conic_constant: float
    source_mode: str  # "fixed" | "moving"
    zone_count: int
    weighting: str  # "equal_area" | "equal_width" | "custom"
    weights: list[float] | None  # custom 时必填，长度 = zone_count
    center_exclusion_radius: float
    min_zone_width: float
    bridge_width: float
    knife_resolution: float
    print_scale: float = 1.0

    def normalized(self) -> dict:
        """规范化参数字典（用于冻结与哈希）。"""
        return {
            "diameter": self.diameter,
            "radius_of_curvature": self.radius_of_curvature,
            "conic_constant": self.conic_constant,
            "source_mode": self.source_mode,
            "zone_count": self.zone_count,
            "weighting": self.weighting,
            "weights": (
                [float(w) for w in self.weights] if self.weights is not None else None
            ),
            "center_exclusion_radius": self.center_exclusion_radius,
            "min_zone_width": self.min_zone_width,
            "bridge_width": self.bridge_width,
            "knife_resolution": self.knife_resolution,
            "print_scale": self.print_scale,
        }


# ---------------- 环带边界 ----------------


def _zone_boundaries(p: MaskParams) -> list[float]:
    """按分区权重计算 n+1 条环带边界半径（含 r0 与 rim）。"""
    n = p.zone_count
    r0 = p.center_exclusion_radius
    rim = p.diameter / 2.0
    if p.weighting == "equal_width":
        return [r0 + (rim - r0) * i / n for i in range(n + 1)]
    if p.weighting == "equal_area":
        frac = [i / n for i in range(n + 1)]
    else:  # custom：环带面积 ∝ 权重
        w = p.weights or []
        if len(w) != n:
            raise MaskError(
                [f"自定义权重数量 {len(w)} 与计划分区数 {n} 不一致"]
            )
        bad = [i for i, x in enumerate(w) if not (math.isfinite(x) and x > 0)]
        if bad:
            raise MaskError(
                [f"自定义权重必须为正有限数（非法权重环带：{bad}）"]
            )
        total = sum(w)
        acc = 0.0
        frac = [0.0]
        for x in w:
            acc += x
            frac.append(acc / total)
    return [math.sqrt(r0 * r0 + (rim * rim - r0 * r0) * f) for f in frac]


# ---------------- 布局计算与校验 ----------------


def validate_constants(p: MaskParams) -> list[str]:
    errs: list[str] = []
    for name, val in (
        ("diameter", p.diameter),
        ("radius_of_curvature", p.radius_of_curvature),
        ("conic_constant", p.conic_constant),
        ("center_exclusion_radius", p.center_exclusion_radius),
        ("min_zone_width", p.min_zone_width),
        ("bridge_width", p.bridge_width),
        ("knife_resolution", p.knife_resolution),
        ("print_scale", p.print_scale),
    ):
        if not math.isfinite(val):
            errs.append(f"{name} 必须是有限数值")
    if errs:
        return errs
    if p.diameter <= 0:
        errs.append("镜面口径必须为正数")
    if p.radius_of_curvature <= 0:
        errs.append("曲率半径必须为正数")
    if abs(p.conic_constant) > 10:
        errs.append("目标圆锥常数超出合理范围 |K| <= 10")
    if p.source_mode not in SOURCE_FACTOR:
        errs.append(f"未知光源模式: {p.source_mode!r}")
    if p.weighting not in WEIGHTING_MODES:
        errs.append(f"未知分区权重模式: {p.weighting!r}")
    if p.zone_count < 2:
        errs.append("计划分区数至少为 2")
    if p.min_zone_width <= 0:
        errs.append("最小环宽必须为正数")
    if p.bridge_width <= 0:
        errs.append("桥宽必须为正数")
    if p.knife_resolution <= 0:
        errs.append("刀口尺分辨率必须为正数")
    if p.print_scale <= 0:
        errs.append("打印缩放校准必须为正数")
    if errs:
        return errs
    rim = p.diameter / 2.0
    r0 = p.center_exclusion_radius
    if r0 < 0:
        errs.append("中心禁测半径不能为负")
    if r0 >= rim - TOL:
        errs.append(
            f"分区越界：中心禁测半径 {r0:g} 不小于镜面半径 {rim:g}，无环带可布"
        )
    return errs


def _window_heights(
    widths: list[float], r_means: list[float], bridge_width: float
) -> list[float]:
    """求各环带开窗高度：以 r_m 为中心，在留桥约束下尽量开大。

    约束：h_i <= 环宽_i - 桥宽（环带内留料），
    h_i + h_{i+1} <= 2·(Δr_m,i - 桥宽)（相邻开窗间距 ≥ 桥宽）。
    从上限出发对相邻约束循环投影（每步将超限的一对各削减超出量的一半），
    收敛到可行解；调用方已保证 Δr_m,i >= 桥宽（零高度可行），
    收敛后由调用方检查 h_i > 0。
    """
    n = len(widths)
    h = [w - bridge_width for w in widths]
    limits = [
        2.0 * (r_means[i + 1] - r_means[i] - bridge_width) for i in range(n - 1)
    ]
    for _ in range(500):
        violation = 0.0
        for i, lim in enumerate(limits):
            excess = h[i] + h[i + 1] - lim
            if excess > 0.0:
                cut = excess / 2.0
                h[i] = max(0.0, h[i] - cut)
                h[i + 1] = max(0.0, h[i + 1] - cut)
                violation = max(violation, excess)
        if violation < 1e-10:
            break
    return h


def compute_layout(p: MaskParams) -> dict:
    """计算环带边界、等效半径、左右开窗与刀口位移预测。

    校验失败抛 MaskError（errors 逐条指明相关环带），成功返回可 JSON 序列化
    的布局字典（zones / bridges / prediction / metrics）。
    """
    errs = validate_constants(p)
    if errs:
        raise MaskError(errs)
    bounds = _zone_boundaries(p)  # custom 权重非法时抛 MaskError

    n = p.zone_count
    b = p.bridge_width
    zones = []
    for i in range(n):
        inner, outer = bounds[i], bounds[i + 1]
        width = outer - inner
        r_mean = math.sqrt((inner**2 + outer**2) / 2.0)
        zones.append(
            {
                "index": i,
                "inner_radius": inner,
                "outer_radius": outer,
                "width": width,
                "r_mean": r_mean,
                "area": math.pi * (outer**2 - inner**2),
            }
        )

    # 第一阶段：环宽与留桥的基本可行性（失败则不再计算开窗，避免误导性错误）
    errs = []
    for z in zones:
        label = f"环带 {z['index']}（{z['inner_radius']:.3f}~{z['outer_radius']:.3f} mm）"
        if z["width"] < p.min_zone_width - TOL:
            errs.append(
                f"环宽不足：{label} 环宽 {z['width']:.3f} < 最小环宽 {p.min_zone_width:g}"
            )
        if z["width"] <= b + TOL:
            errs.append(
                f"结构无法留桥：{label} 环宽 {z['width']:.3f} <= 桥宽 {b:g}"
            )
    if errs:
        raise MaskError(errs)

    # 相邻环带等效半径间距必须容纳桥宽（开窗高度再小也无法绕过该约束）
    errs = []
    for i in range(n - 1):
        d_rm = zones[i + 1]["r_mean"] - zones[i]["r_mean"]
        if d_rm < b - TOL:
            errs.append(
                f"结构无法留桥：环带 {i} 与环带 {i + 1} 等效半径间距 "
                f"{d_rm:.3f} < 桥宽 {b:g}"
            )
    if errs:
        raise MaskError(errs)

    # 开窗竖直范围：以 r_m 为中心，窗高在留桥约束下取最大（投影法）
    heights = _window_heights(
        [z["width"] for z in zones], [z["r_mean"] for z in zones], b
    )
    errs = []
    for z, h in zip(zones, heights):
        if h <= TOL:
            errs.append(
                f"结构无法留桥：环带 {z['index']} 在留桥约束下开窗高度为零"
            )
        z["window_height"] = h
        z["y_bottom"] = z["r_mean"] - h / 2.0
        z["y_top"] = z["r_mean"] + h / 2.0
    if errs:
        raise MaskError(errs)

    # 第二阶段：开窗不得越出本环带（侵入相邻环带即与相邻开窗/桥相交）
    errs = []
    for z in zones:
        label = f"环带 {z['index']}（{z['inner_radius']:.3f}~{z['outer_radius']:.3f} mm）"
        if z["y_top"] > z["outer_radius"] + TOL:
            errs.append(
                f"开窗相交：{label} 开窗顶边越出环带外边界"
                f"（{z['y_top']:.3f} > {z['outer_radius']:.3f}），侵入相邻环带"
            )
        if z["y_bottom"] < z["inner_radius"] - TOL:
            errs.append(
                f"开窗相交：{label} 开窗底边越出环带内边界"
                f"（{z['y_bottom']:.3f} < {z['inner_radius']:.3f}），侵入相邻环带"
            )
    if errs:
        raise MaskError(errs)

    # 开窗水平范围（弦长）；x_inner = 0 表示左右开窗在中心线相连
    errs = []
    for z in zones:
        x_in = math.sqrt(max(0.0, z["inner_radius"] ** 2 - z["y_bottom"] ** 2))
        x_out = math.sqrt(max(0.0, z["outer_radius"] ** 2 - z["y_top"] ** 2))
        if x_out - x_in <= TOL:
            errs.append(
                f"开窗相交：环带 {z['index']} 开窗水平宽度为零，无法开窗"
            )
        z["x_inner"] = x_in
        z["x_outer"] = x_out
        z["window_width"] = x_out - x_in
        z["window_left"] = {"x_from": -x_out, "x_to": -x_in}
        z["window_right"] = {"x_from": x_in, "x_to": x_out}
    if errs:
        raise MaskError(errs)

    bridges = [
        {
            "between": [i, i + 1],
            "gap": zones[i + 1]["y_bottom"] - zones[i]["y_top"],
        }
        for i in range(n - 1)
    ]
    prediction = predict_knife(
        [z["r_mean"] for z in zones],
        p.conic_constant,
        p.radius_of_curvature,
        p.source_mode,
        p.knife_resolution,
    )
    metrics = _layout_metrics(zones, bridges, b, prediction)
    return {
        "zones": zones,
        "bridges": bridges,
        "prediction": prediction,
        "metrics": metrics,
    }


def _layout_metrics(
    zones: list[dict], bridges: list[dict], bridge_width: float, prediction: dict
) -> dict:
    widths = [z["width"] for z in zones]
    areas = [z["area"] for z in zones]
    heights = [z["window_height"] for z in zones]
    gaps = [br["gap"] for br in bridges]
    # 制作余量 = 最薄开窗高度：开窗在留桥约束下已取最大，桥宽富余恒为 0，
    # 真正区分制作难度的是最薄开窗（越薄越难裁切与对准）。
    return {
        "min_zone_width_mm": min(widths),
        "max_zone_width_mm": max(widths),
        "min_window_height_mm": min(heights),
        "min_bridge_gap_mm": min(gaps) if gaps else None,
        "area_balance": min(areas) / max(areas),
        "fabrication_margin_mm": min(heights),
        "min_resolvable_delta_mm": prediction["min_adjacent_delta_mm"],
        "unresolvable_pairs": prediction["unresolvable_pairs"],
    }


# ---------------- 刀口位移预测 ----------------


def predict_knife(
    r_means: list[float],
    conic_constant: float,
    radius_of_curvature: float,
    source_mode: str,
    knife_resolution: float,
) -> dict:
    """目标圆锥下各区理想刀口位移与相邻区对可分辨性。

    刀口位移换算复用光学模型的光源模式因子：固定光源位移 = LA，
    移动光源位移 = LA / 2。零点取最内环带（与分析默认参考一致）。
    """
    sf = SOURCE_FACTOR[source_mode]
    rows = []
    for i, rm in enumerate(r_means):
        la = float(ideal_la(rm, conic_constant, radius_of_curvature))
        rows.append(
            {
                "index": i,
                "r_mean": rm,
                "la_ideal_mm": la,
                "knife_reading_mm": la / sf,
            }
        )
    ref = rows[0]["knife_reading_mm"] if rows else 0.0
    for row in rows:
        row["knife_relative_mm"] = row["knife_reading_mm"] - ref
    adjacent = []
    for i in range(len(rows) - 1):
        delta = abs(
            rows[i + 1]["knife_relative_mm"] - rows[i]["knife_relative_mm"]
        )
        adjacent.append(
            {
                "pair": [i, i + 1],
                "delta_mm": delta,
                "resolvable": bool(delta >= knife_resolution - TOL),
            }
        )
    unresolvable = [a["pair"] for a in adjacent if not a["resolvable"]]
    deltas = [a["delta_mm"] for a in adjacent]
    return {
        "reference": "innermost",
        "source_mode": source_mode,
        "knife_resolution_mm": knife_resolution,
        "zones": rows,
        "adjacent": adjacent,
        "min_adjacent_delta_mm": min(deltas) if deltas else None,
        "unresolvable_pairs": unresolvable,
    }


# ---------------- 边界搜索 ----------------


def search_layouts(
    base: MaskParams,
    zone_counts: list[int],
    bridge_widths: list[float],
    layouts: list[str],
    top: int = 10,
) -> dict:
    """在分区数 × 桥宽 × 权重模式范围内搜索可行布局并排序。

    排序准则（全部降序）：最小可分辨位移（相邻区对刀口位移差的最小值）、
    面积均衡度（最小/最大环带面积）、制作余量（最薄开窗高度）。
    仅返回通过全部结构校验的候选；被否决的组合按类别计数说明。
    """
    candidates = []
    evaluated = 0
    rejected: dict[str, int] = {}
    for layout_mode in layouts:
        for n in zone_counts:
            for bw in bridge_widths:
                p = replace(
                    base,
                    zone_count=int(n),
                    bridge_width=float(bw),
                    weighting=layout_mode,
                    weights=None,
                )
                evaluated += 1
                try:
                    lay = compute_layout(p)
                except MaskError as exc:
                    for e in exc.errors:
                        category = e.split("：", 1)[0]
                        rejected[category] = rejected.get(category, 0) + 1
                    continue
                m = lay["metrics"]
                candidates.append(
                    {
                        "weighting": layout_mode,
                        "zone_count": int(n),
                        "bridge_width": float(bw),
                        "metrics": m,
                        "zones": [
                            {
                                "index": z["index"],
                                "inner_radius": z["inner_radius"],
                                "outer_radius": z["outer_radius"],
                                "r_mean": z["r_mean"],
                                "width": z["width"],
                            }
                            for z in lay["zones"]
                        ],
                        "params": {
                            **p.normalized(),
                            "unit": "mm",
                        },
                    }
                )
    candidates.sort(
        key=lambda c: (
            -(c["metrics"]["min_resolvable_delta_mm"] or 0.0),
            -c["metrics"]["area_balance"],
            -c["metrics"]["fabrication_margin_mm"],
        )
    )
    for rank, c in enumerate(candidates):
        c["rank"] = rank
    return {
        "evaluated": evaluated,
        "feasible": len(candidates),
        "rejected_by_category": rejected,
        "ranking": [
            "min_resolvable_delta_mm",
            "area_balance",
            "fabrication_margin_mm",
        ],
        "candidates": candidates[: max(0, top)],
    }


# ---------------- 1:1 SVG（带尺寸线与校准尺） ----------------


def _fmt(v: float, nd: int = 3) -> str:
    s = f"{v:.{nd}f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def render_mask_svg(
    params: dict,
    layout: dict,
    *,
    title: str,
    subtitle: str,
) -> str:
    """生成 1:1 遮罩 SVG：几何按 print_scale 缩放，单位为 mm。

    包含：镜面外圆、禁测区、各环带边界、左右开窗、中心线、直径尺寸线、
    各环带边界的半径尺寸线、开窗高度尺寸线、校准尺与参数说明。
    width/height 以 mm 声明，100% 打印即为实际尺寸（打印缩放校准已计入）。
    """
    s = float(params["print_scale"])
    rim = params["diameter"] / 2.0
    r0 = params["center_exclusion_radius"]
    zones = layout["zones"]
    n = len(zones)

    # 画布布局（mm，未缩放坐标系，y 向上为正，镜面中心为原点）
    side = 16.0
    dia_level = rim + 8.0  # 直径尺寸线（镜面下方）
    radial_start = rim + 14.0
    radial_step = 4.5
    radial_bottom = radial_start + (n + 1) * radial_step
    ruler_y = radial_bottom + 10.0
    text_top = ruler_y + 10.0
    cx = rim + side
    cy = rim + side
    width_mm = 2.0 * (rim + side)
    height_mm = cy + text_top + 4.0 * 4.4 + 4.0

    def X(x: float) -> float:
        return (cx + x) * s

    def Y(y: float) -> float:
        return (cy - y) * s

    def L(v: float) -> float:
        return v * s

    el: list[str] = []
    el.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_fmt(width_mm * s, 2)}mm" height="{_fmt(height_mm * s, 2)}mm" '
        f'viewBox="0 0 {_fmt(width_mm * s, 3)} {_fmt(height_mm * s, 3)}">'
    )
    el.append(
        '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="5" markerHeight="5" markerUnits="userSpaceOnUse" '
        'orient="auto-start-reverse">'
        '<path d="M0,0 L10,5 L0,10 z" fill="#000"/></marker></defs>'
    )
    el.append(f'<rect x="0" y="0" width="{_fmt(width_mm * s, 3)}" '
              f'height="{_fmt(height_mm * s, 3)}" fill="#fff"/>')

    # 中心线（点划线）
    for x1, y1, x2, y2 in ((-rim - 8, 0, rim + 8, 0), (0, -rim - 8, 0, rim + 8)):
        el.append(
            f'<line x1="{_fmt(X(x1))}" y1="{_fmt(Y(y1))}" x2="{_fmt(X(x2))}" '
            f'y2="{_fmt(Y(y2))}" stroke="#888" stroke-width="{_fmt(L(0.2))}" '
            f'stroke-dasharray="{_fmt(L(4))},{_fmt(L(1))},{_fmt(L(1))},{_fmt(L(1))}"/>'
        )
    # 镜面外圆与禁测区
    el.append(
        f'<circle cx="{_fmt(X(0))}" cy="{_fmt(Y(0))}" r="{_fmt(L(rim))}" '
        f'fill="none" stroke="#000" stroke-width="{_fmt(L(0.4))}"/>'
    )
    if r0 > TOL:
        el.append(
            f'<circle cx="{_fmt(X(0))}" cy="{_fmt(Y(0))}" r="{_fmt(L(r0))}" '
            f'fill="none" stroke="#c00" stroke-width="{_fmt(L(0.3))}" '
            f'stroke-dasharray="{_fmt(L(2))},{_fmt(L(1.5))}"/>'
        )
        el.append(
            f'<text x="{_fmt(X(r0 * 0.7071) + L(1))}" y="{_fmt(Y(r0 * 0.7071))}" '
            f'font-size="{_fmt(L(2.8))}" fill="#c00" font-family="sans-serif">'
            f'禁测区 R{_fmt(r0)}</text>'
        )
    # 环带边界（细灰圆）
    for z in zones[:-1]:
        el.append(
            f'<circle cx="{_fmt(X(0))}" cy="{_fmt(Y(0))}" '
            f'r="{_fmt(L(z["outer_radius"]))}" fill="none" stroke="#bbb" '
            f'stroke-width="{_fmt(L(0.15))}"/>'
        )
    # 左右开窗
    for z in zones:
        for key in ("window_left", "window_right"):
            w = z[key]
            el.append(
                f'<rect x="{_fmt(X(w["x_from"]))}" y="{_fmt(Y(z["y_top"]))}" '
                f'width="{_fmt(L(w["x_to"] - w["x_from"]))}" '
                f'height="{_fmt(L(z["y_top"] - z["y_bottom"]))}" '
                f'fill="rgba(0,80,200,0.08)" stroke="#0050c8" '
                f'stroke-width="{_fmt(L(0.35))}"/>'
            )
        # 环带标注：序号 + 等效半径（右侧开窗旁，交错避免重叠）
        dy = 0.0 if z["index"] % 2 == 0 else 2.6
        el.append(
            f'<text x="{_fmt(X(z["x_outer"]) + L(1.2))}" '
            f'y="{_fmt(Y(z["r_mean"]) + L(0.9) + L(dy))}" '
            f'font-size="{_fmt(L(2.4))}" fill="#0050c8" font-family="sans-serif">'
            f'{z["index"] + 1}: {_fmt(z["r_mean"], 2)}</text>'
        )
        # 开窗高度尺寸线（左侧，交错外移）
        dim_x = -(z["x_outer"] + 2.5 + (z["index"] % 2) * 3.5)
        el.append(
            f'<line x1="{_fmt(X(dim_x))}" y1="{_fmt(Y(z["y_bottom"]))}" '
            f'x2="{_fmt(X(dim_x))}" y2="{_fmt(Y(z["y_top"]))}" stroke="#000" '
            f'stroke-width="{_fmt(L(0.18))}" marker-start="url(#arr)" '
            f'marker-end="url(#arr)"/>'
        )
        el.append(
            f'<text x="{_fmt(X(dim_x) - L(1))}" y="{_fmt(Y(z["r_mean"]))}" '
            f'font-size="{_fmt(L(2.2))}" fill="#000" font-family="sans-serif" '
            f'text-anchor="middle" transform="rotate(-90 {_fmt(X(dim_x) - L(1))} '
            f'{_fmt(Y(z["r_mean"]))})">{_fmt(z["window_height"], 2)}</text>'
        )

    # 直径尺寸线
    yd = -dia_level
    for xx in (-rim, rim):
        el.append(
            f'<line x1="{_fmt(X(xx))}" y1="{_fmt(Y(0))}" x2="{_fmt(X(xx))}" '
            f'y2="{_fmt(Y(yd - 1))}" stroke="#999" stroke-width="{_fmt(L(0.15))}"/>'
        )
    el.append(
        f'<line x1="{_fmt(X(-rim))}" y1="{_fmt(Y(yd))}" x2="{_fmt(X(rim))}" '
        f'y2="{_fmt(Y(yd))}" stroke="#000" stroke-width="{_fmt(L(0.25))}" '
        f'marker-start="url(#arr)" marker-end="url(#arr)"/>'
    )
    el.append(
        f'<text x="{_fmt(X(0))}" y="{_fmt(Y(yd) + L(3.4))}" '
        f'font-size="{_fmt(L(3.2))}" text-anchor="middle" font-family="sans-serif">'
        f'⌀ {_fmt(params["diameter"])}</text>'
    )
    # 半径尺寸线（自中心向右的基线尺寸，逐条下移）
    bounds = [z["inner_radius"] for z in zones] + [zones[-1]["outer_radius"]]
    el.append(
        f'<line x1="{_fmt(X(0))}" y1="{_fmt(Y(-rim - 2))}" x2="{_fmt(X(0))}" '
        f'y2="{_fmt(Y(-(radial_start + n * radial_step) - 1))}" stroke="#999" '
        f'stroke-width="{_fmt(L(0.15))}"/>'
    )
    for j, rb in enumerate(bounds):
        ly = -(radial_start + j * radial_step)
        el.append(
            f'<line x1="{_fmt(X(rb))}" y1="{_fmt(Y(0))}" x2="{_fmt(X(rb))}" '
            f'y2="{_fmt(Y(ly - 1))}" stroke="#999" stroke-width="{_fmt(L(0.15))}"/>'
        )
        el.append(
            f'<line x1="{_fmt(X(0))}" y1="{_fmt(Y(ly))}" x2="{_fmt(X(rb))}" '
            f'y2="{_fmt(Y(ly))}" stroke="#000" stroke-width="{_fmt(L(0.18))}" '
            f'marker-start="url(#arr)" marker-end="url(#arr)"/>'
        )
        el.append(
            f'<text x="{_fmt(X(rb / 2))}" y="{_fmt(Y(ly) - L(0.8))}" '
            f'font-size="{_fmt(L(2.4))}" text-anchor="middle" '
            f'font-family="sans-serif">R{_fmt(rb)}</text>'
        )

    # 校准尺（随打印缩放一同缩放：打印后实测核对 1:1）
    ruler_len = 100.0 if 2 * rim >= 120 else (50.0 if 2 * rim >= 60 else 20.0)
    rx0, ry = -rim, -ruler_y
    el.append(
        f'<line x1="{_fmt(X(rx0))}" y1="{_fmt(Y(ry))}" x2="{_fmt(X(rx0 + ruler_len))}" '
        f'y2="{_fmt(Y(ry))}" stroke="#000" stroke-width="{_fmt(L(0.3))}"/>'
    )
    step = ruler_len / 10.0
    for k in range(11):
        tx = rx0 + k * step
        tick = 2.2 if k % 5 == 0 else 1.4
        el.append(
            f'<line x1="{_fmt(X(tx))}" y1="{_fmt(Y(ry))}" x2="{_fmt(X(tx))}" '
            f'y2="{_fmt(Y(ry - tick))}" stroke="#000" stroke-width="{_fmt(L(0.25))}"/>'
        )
        if k % 5 == 0:
            el.append(
                f'<text x="{_fmt(X(tx))}" y="{_fmt(Y(ry - tick - 1))}" '
                f'font-size="{_fmt(L(2.4))}" text-anchor="middle" '
                f'font-family="sans-serif">{_fmt(k * step, 0)}</text>'
            )
    el.append(
        f'<text x="{_fmt(X(rx0))}" y="{_fmt(Y(ry - 6))}" '
        f'font-size="{_fmt(L(2.8))}" font-family="sans-serif">'
        f'校准尺 {_fmt(ruler_len, 0)} mm（1:1 打印后实测核对，误差应 ≤ 0.2 mm）</text>'
    )

    # 参数说明
    lines = [escape(title), escape(subtitle)]
    lines.append(
        escape(
            f"D={_fmt(params['diameter'])} mm  R={_fmt(params['radius_of_curvature'])} mm"
            f"  K={_fmt(params['conic_constant'])}  光源={params['source_mode']}"
        )
    )
    lines.append(
        escape(
            f"分区 {n}（{params['weighting']}）  禁测半径 "
            f"{_fmt(params['center_exclusion_radius'])}  桥宽 {_fmt(params['bridge_width'])}"
            f"  最小环宽 {_fmt(params['min_zone_width'])}"
        )
    )
    lines.append(
        escape(
            f"刀口尺分辨率 {_fmt(params['knife_resolution'])} mm  "
            f"打印缩放 {_fmt(params['print_scale'], 4)}"
        )
    )
    for i, line in enumerate(lines):
        el.append(
            f'<text x="{_fmt(X(-rim))}" y="{_fmt(Y(-(text_top + i * 4.4)))}" '
            f'font-size="{_fmt(L(3.0))}" font-family="sans-serif">{line}</text>'
        )
    el.append("</svg>")
    return "".join(el)
