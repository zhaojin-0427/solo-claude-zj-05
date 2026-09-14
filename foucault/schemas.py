"""API 请求/响应的 Pydantic 模型。

单位约定：diameter、radius_of_curvature、分区半径、读数、instrument_offset
一律使用 `unit` 声明的单位（"mm" 或 "in"）；波长固定为 nm。
未声明或无法识别的单位会被拒绝（422）。
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Unit = Literal["mm", "in"]
SourceMode = Literal["fixed", "moving"]
WeightingMode = Literal["equal_area", "equal_width", "custom"]
SearchLayout = Literal["equal_area", "equal_width"]


class ZoneIn(BaseModel):
    inner_radius: float
    outer_radius: float
    readings: list[float] = Field(min_length=1)


class AnalysisOptionsIn(BaseModel):
    min_readings_per_zone: int = Field(default=2, ge=1, le=100)
    error_band_waves: float = Field(default=0.25, gt=0, le=10)
    grid_points: int = Field(default=400, ge=50, le=5000)
    fit_conic: bool = True
    fit_defocus: bool = True
    reference: Literal["innermost", "mean"] = "innermost"


class TestCreateIn(BaseModel):
    """创建测试批次：直接给出分区，或引用遮罩版本（分区边界取自遮罩）。

    引用遮罩时：常量（口径/曲率半径/圆锥常数/光源模式）以遮罩版本冻结值
    为准，payload 若重复提供且不一致则 422；读数由 zone_readings 按遮罩
    环带顺序给出；波长、仪器偏移与算法选项仍由 payload 提供。
    """

    name: str | None = None
    notes: str | None = None
    diameter: float | None = None
    radius_of_curvature: float | None = None
    conic_constant: float | None = None
    wavelength_nm: float
    source_mode: SourceMode | None = None
    unit: Unit | None = None
    instrument_offset: float = 0.0
    zones: list[ZoneIn] | None = Field(default=None, min_length=2)
    mask_scheme_id: int | None = None
    mask_version_no: int | None = None  # 缺省取该方案最新版本
    zone_readings: list[list[float]] | None = None  # 引用遮罩时必填
    options: AnalysisOptionsIn = Field(default_factory=AnalysisOptionsIn)


class ExcludeItem(BaseModel):
    reading_id: int
    reason: str = Field(min_length=1, max_length=500)


class ExcludeIn(BaseModel):
    items: list[ExcludeItem] = Field(min_length=1)
    analyze: bool = True  # 剔除后自动重分析生成新版本


class RestoreIn(BaseModel):
    reading_ids: list[int] = Field(min_length=1)
    analyze: bool = True


class FreezeIn(BaseModel):
    reading_ids: list[int] | None = None  # None + all_valid=True → 冻结全部有效读数
    all_valid: bool = False


class AnalyzeIn(BaseModel):
    options: AnalysisOptionsIn | None = None  # 缺省沿用创建时的算法选项


class CompareEntry(BaseModel):
    test_id: int
    version_no: int | None = None  # 缺省取最新版本


class CompareIn(BaseModel):
    entries: list[CompareEntry] = Field(min_length=2, max_length=10)


class CorrectionSearchIn(BaseModel):
    version_no: int | None = None  # 缺省基于最新版本
    max_removal_nm: float = Field(gt=0)
    edge_zone_max_removal_nm: float | None = Field(default=None, ge=0)
    preserve_edge: bool = False
    max_mean_removal_nm: float | None = Field(default=None, gt=0)
    smoothing_weights: list[float] | None = None
    refit_defocus: bool = True


class MaskParamsIn(BaseModel):
    """遮罩版本冻结的全部参数（常量与制作限制）。

    长度量按 unit 声明的单位提交，内部一律换算为 mm 冻结。
    """

    diameter: float
    radius_of_curvature: float
    conic_constant: float
    source_mode: SourceMode
    unit: Unit
    zone_count: int = Field(ge=2, le=200)
    weighting: WeightingMode = "equal_area"
    weights: list[float] | None = None  # weighting=custom 时必填，长度 = zone_count
    center_exclusion_radius: float = Field(ge=0)
    min_zone_width: float = Field(gt=0)
    bridge_width: float = Field(gt=0)
    knife_resolution: float = Field(gt=0)
    print_scale: float = Field(default=1.0, gt=0)


class MaskSchemeCreateIn(MaskParamsIn):
    name: str | None = None
    notes: str | None = None


class MaskSearchIn(BaseModel):
    """遮罩边界搜索：在分区数 × 环宽 × 桥宽 × 权重模式范围内枚举并排序。

    环宽（最小环宽约束）与桥宽均为 [min, max] 范围：给 step 按步长取遍
    （含端点，不截断）；缺省 max 时取单值 min；缺省 step 且 max > min 时
    取 5 个等距点。组合总数超限时明确拒绝（422），不静默漏算。
    """

    diameter: float
    radius_of_curvature: float
    conic_constant: float
    source_mode: SourceMode
    unit: Unit
    center_exclusion_radius: float = Field(ge=0)
    knife_resolution: float = Field(gt=0)
    zone_count_min: int = Field(ge=2, le=200)
    zone_count_max: int = Field(ge=2, le=200)
    zone_width_min: float = Field(gt=0)
    zone_width_max: float | None = Field(default=None, gt=0)
    zone_width_step: float | None = Field(default=None, gt=0)
    bridge_width_min: float = Field(gt=0)
    bridge_width_max: float | None = Field(default=None, gt=0)
    bridge_width_step: float | None = Field(default=None, gt=0)
    layouts: list[SearchLayout] = Field(
        default_factory=lambda: ["equal_area", "equal_width"]
    )
    top: int = Field(default=10, ge=1, le=100)


# ---------------- 往返测量会话 ----------------

Direction = Literal["forward", "reverse"]


class SessionThresholdsIn(BaseModel):
    """定稿质量阈值（长度量按会话 unit 提交，内部换算为 mm）；
    缺省 None 时由遮罩版本冻结的刀口尺分辨率补齐。"""

    max_dispersion: float | None = Field(default=None, gt=0)
    max_direction_diff: float | None = Field(default=None, gt=0)
    max_drift_residual: float | None = Field(default=None, gt=0)


class SessionCreateIn(BaseModel):
    """创建往返测量会话：引用不可变遮罩版本并填写采集计划参数。"""

    name: str | None = None
    notes: str | None = None
    mask_scheme_id: int
    mask_version_no: int | None = None  # 缺省取该方案最新版本
    unit: Unit = "mm"
    repeats_per_zone: int = Field(ge=1, le=50)  # 每方向重复次数（每区合计 2n）
    start_direction: Direction = "forward"
    reference_zone_index: int = Field(default=0, ge=0)
    reference_refresh: int = Field(default=2, ge=1)  # 每 N 个常规测位穿插一次参考复测
    collect_deadline: datetime  # 采集时限，必填；naive 按 UTC
    wavelength_nm: float = Field(gt=0)
    instrument_offset: float = 0.0
    thresholds: SessionThresholdsIn = Field(default_factory=SessionThresholdsIn)
    options: AnalysisOptionsIn | None = None  # 缺省 min_readings = 2·repeats


class ReadingSubmitIn(BaseModel):
    """逐笔提交刀口读数：必须严格落在计划中的下一个待采测次上。"""

    seq: int = Field(ge=1)
    knife_position: float  # 会话 unit 下的刀口测微器读数
    zone_index: int = Field(ge=0)
    direction: Direction
    collected_at: datetime


class RetestIn(BaseModel):
    """对指定测次补测：旧 attempt 保留，新增一次 attempt。"""

    seq: int = Field(ge=1)
    knife_position: float
    zone_index: int = Field(ge=0)
    direction: Direction
    collected_at: datetime


class SessionExcludeItem(BaseModel):
    reading_id: int
    reason: str = Field(min_length=1, max_length=500)


class SessionExcludeIn(BaseModel):
    items: list[SessionExcludeItem] = Field(min_length=1)


class SessionLockIn(BaseModel):
    seqs: list[int] | None = Field(default=None, min_length=1)
    all_collected: bool = False  # True → 锁定全部已采集测位
