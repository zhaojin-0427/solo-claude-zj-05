"""API 请求/响应的 Pydantic 模型。

单位约定：diameter、radius_of_curvature、分区半径、读数、instrument_offset
一律使用 `unit` 声明的单位（"mm" 或 "in"）；波长固定为 nm。
未声明或无法识别的单位会被拒绝（422）。
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Unit = Literal["mm", "in"]
SourceMode = Literal["fixed", "moving"]


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
    name: str | None = None
    notes: str | None = None
    diameter: float
    radius_of_curvature: float
    conic_constant: float
    wavelength_nm: float
    source_mode: SourceMode
    unit: Unit
    instrument_offset: float = 0.0
    zones: list[ZoneIn] = Field(min_length=2)
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
