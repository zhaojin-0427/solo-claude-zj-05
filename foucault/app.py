"""FastAPI 路由：测试批次管理、读数冻结/剔除、分析版本、批次对比、修正搜索、
Couder 遮罩方案（版本化布局、边界搜索、1:1 SVG）、往返测量会话
（逐笔采集、漂移/回程校正、定稿生成测试批次），以及子午线复测研究
（4~16 份冻结分析版本的二次角向谐波拟合、镜面/架位像散分离、留一稳定性、
定稿冻结与复制追加）。"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

from . import __version__
from .correction import CorrectionError, search_corrections
from .mask import (
    MaskError,
    MaskParams,
    compute_layout,
    render_mask_svg,
    search_layouts,
    validate_constants as validate_mask_constants,
)
from .meridian import (
    ANGLE_CONVENTION,
    MeridianError,
    extract_zone_signals,
    fit_harmonics,
    leave_one_out,
    loo_from_fit_result,
)
from .optics import (
    UNIT_TO_MM,
    AnalysisError,
    Constants,
    Options,
    ZoneInput,
    compute_input_hash,
    reduce_test,
)
from .schemas import (
    AnalyzeIn,
    AnalysisOptionsIn,
    CompareIn,
    CorrectionSearchIn,
    ExcludeIn,
    FreezeIn,
    MaskParamsIn,
    MaskSchemeCreateIn,
    MaskSearchIn,
    ReadingSubmitIn,
    RestoreIn,
    RetestIn,
    SessionCreateIn,
    SessionExcludeIn,
    SessionLockIn,
    StudyAppendIn,
    StudyCopyIn,
    StudyCreateIn,
    StudyExcludeIn,
    StudyFinalizeIn,
    StudySolveIn,
    TestCreateIn,
)
from .session import (
    SessionError,
    build_plan,
    correct_session,
    iso as iso_dt,
)
from .storage import ConflictError, Database, NotFoundError

DEFAULT_DB = os.environ.get("FOUCAULT_DB", "foucault.db")

# 遮罩边界搜索的组合数上限（分区数 × 环宽 × 桥宽 × 权重模式），超限明确拒绝
SEARCH_COMBO_LIMIT = 20000


def _frange(lo: float, hi: float, step: float | None) -> list[float]:
    """[lo, hi] 范围内按步长取值的完整序列（含端点，不截断）。

    step 为 None：lo == hi 时取单值，否则取 5 个等距点。
    """
    if step is None:
        if abs(hi - lo) < 1e-12:
            return [lo]
        return [lo + (hi - lo) * i / 4.0 for i in range(5)]
    out = []
    k = 0
    while True:
        v = lo + k * step  # 每次由起点计算，避免累加误差
        if v > hi + 1e-9:
            break
        out.append(round(min(v, hi), 9))
        k += 1
    return out


def _err(status: int, message: str, details: list | None = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"message": message, "errors": details or []},
    )


def _validate_constants(
    diameter: float,
    radius_of_curvature: float,
    conic_constant: float,
    wavelength_nm: float,
    instrument_offset: float,
) -> list[str]:
    errs: list[str] = []
    for name, val in (
        ("diameter", diameter),
        ("radius_of_curvature", radius_of_curvature),
        ("conic_constant", conic_constant),
        ("wavelength_nm", wavelength_nm),
        ("instrument_offset", instrument_offset),
    ):
        if not math.isfinite(val):
            errs.append(f"{name} 必须是有限数值")
    if diameter <= 0:
        errs.append("镜面口径必须为正数")
    if radius_of_curvature <= 0:
        errs.append("曲率半径必须为正数")
    if wavelength_nm <= 0:
        errs.append("检测波长必须为正数")
    if abs(conic_constant) > 10:
        errs.append("目标圆锥常数超出合理范围 |K| <= 10")
    return errs


def _validate_zones(diameter: float, zones: list, min_n: int) -> list[str]:
    """分区业务校验；zones 与 diameter 同单位（读数只查有限性与数量）。"""
    errs: list[str] = []
    rim = diameter / 2.0
    zones = sorted(zones, key=lambda z: (z.inner_radius, z.outer_radius))
    for i, z in enumerate(zones):
        label = f"分区 {i}（{z.inner_radius}~{z.outer_radius}）"
        if not (math.isfinite(z.inner_radius) and math.isfinite(z.outer_radius)):
            errs.append(f"{label}：半径必须是有限数值")
            continue
        if z.inner_radius < 0:
            errs.append(f"{label}：内半径不能为负")
        if z.outer_radius <= z.inner_radius:
            errs.append(f"{label}：外半径必须大于内半径")
        if z.outer_radius > rim + 1e-9:
            errs.append(f"{label}：分区越出镜面（半径上限 {rim}）")
        if i > 0 and z.inner_radius < zones[i - 1].outer_radius - 1e-12:
            errs.append(f"{label}：与上一分区重叠")
        bad = [r for r in z.readings if not math.isfinite(r)]
        if bad:
            errs.append(f"{label}：读数包含非数值")
        elif len(z.readings) < min_n:
            errs.append(
                f"{label}：有效读数不足（{len(z.readings)} < {min_n}）"
            )
    return errs


def _validate_payload(p: TestCreateIn) -> list[str]:
    """业务校验：返回错误列表（空 = 通过）。类型/字面量错误由 Pydantic 拦截。"""
    errs = _validate_constants(
        p.diameter,
        p.radius_of_curvature,
        p.conic_constant,
        p.wavelength_nm,
        p.instrument_offset,
    )
    if errs:
        return errs
    return _validate_zones(p.diameter, p.zones, p.options.min_readings_per_zone)


def _constants_of(test: dict) -> Constants:
    return Constants(
        diameter=test["diameter"],
        radius_of_curvature=test["radius_of_curvature"],
        conic_constant=test["conic_constant"],
        wavelength_nm=test["wavelength_nm"],
        source_mode=test["source_mode"],
        instrument_offset=test["instrument_offset"],
    )


def _options_of(options_dict: dict) -> Options:
    return Options(
        min_readings_per_zone=options_dict["min_readings_per_zone"],
        error_band_waves=options_dict["error_band_waves"],
        grid_points=options_dict["grid_points"],
        fit_conic=options_dict["fit_conic"],
        fit_defocus=options_dict["fit_defocus"],
        reference=options_dict["reference"],
    )


def _collect_valid_zones(test: dict) -> list[ZoneInput]:
    """按分区顺序收集有效（未剔除）读数。"""
    zones = []
    for z in sorted(test["zones"], key=lambda x: x["zone_index"]):
        vals = [r["value"] for r in z["readings"] if not r["excluded"]]
        zones.append(ZoneInput(inner=z["inner_radius"], outer=z["outer_radius"], readings=vals))
    return zones


def _build_snapshot(test: dict, options_dict: dict) -> dict:
    """版本快照：常量 + 算法选项 + 全部原始读数（含剔除状态与原因）。

    快照必须能完整还原创建该版本时的输入面貌：每条读数的原始提交值、
    规范值（mm）、是否被剔除及剔除原因都原样冻结。
    """
    zones = []
    for z in sorted(test["zones"], key=lambda x: x["zone_index"]):
        readings = [
            {
                "id": r["id"],
                "value": r["value"],
                "value_original": r["value_original"],
                "excluded": bool(r["excluded"]),
                "exclude_reason": r["exclude_reason"],
                "frozen": bool(r["frozen"]),
            }
            for r in z["readings"]
        ]
        zones.append(
            {
                "index": z["zone_index"],
                "inner_radius": z["inner_radius"],
                "outer_radius": z["outer_radius"],
                "readings": readings,
            }
        )
    return {
        "constants": {
            "diameter": test["diameter"],
            "radius_of_curvature": test["radius_of_curvature"],
            "conic_constant": test["conic_constant"],
            "wavelength_nm": test["wavelength_nm"],
            "source_mode": test["source_mode"],
            "unit": test["unit"],
            "instrument_offset": test["instrument_offset"],
        },
        "options": options_dict,
        "zones": zones,
    }


def _hash_payload(snapshot: dict) -> dict:
    """输入哈希只覆盖真正参与分析的数据：常量、选项、有效读数值。

    剔除原因、冻结标记等元数据不改变分析输入，不进哈希；
    剔除/恢复改变有效读数集合，会改变哈希从而产生新版本。
    """
    return {
        "constants": snapshot["constants"],
        "options": snapshot["options"],
        "zones": [
            {
                "inner_radius": z["inner_radius"],
                "outer_radius": z["outer_radius"],
                "readings": [
                    r["value"] for r in z["readings"] if not r["excluded"]
                ],
            }
            for z in snapshot["zones"]
        ],
    }


def _run_analysis(db: Database, test_id: int, options_dict: dict | None = None) -> dict:
    """执行分析并写入新版本；输入与最新版本一致时复用该版本（幂等）。"""
    test = db.get_test(test_id)
    opts = options_dict if options_dict is not None else test["options"]
    zones = _collect_valid_zones(test)
    snapshot = _build_snapshot(test, opts)
    input_hash = compute_input_hash(_hash_payload(snapshot))
    try:
        latest = db.get_latest_version(test_id)
    except NotFoundError:
        latest = None
    if latest is not None and latest["input_hash"] == input_hash:
        latest["reused"] = True
        return latest
    result = reduce_test(_constants_of(test), zones, _options_of(opts))
    version = db.create_version(test_id, input_hash, snapshot, result)
    version["reused"] = False
    return version


def _version_brief(v: dict) -> dict:
    return {
        "test_id": v["test_id"],
        "version_no": v["version_no"],
        "created_at": v["created_at"],
        "input_hash": v["input_hash"],
        "summary": v["result"].get("summary", {}),
        "reused": v.get("reused", False),
    }


def create_app(db_path: str | None = None) -> FastAPI:
    db = Database(db_path or DEFAULT_DB)
    app = FastAPI(
        title="Foucault 刀口仪镜面分析 API",
        version=__version__,
        description="将 Couder 遮罩分区读数还原为镜面/波前误差，支持版本化分析、"
        "读数冻结与剔除、批次对比、分区修正量搜索、Couder 遮罩方案设计"
        "（版本化布局、边界搜索、1:1 SVG）、上机往返测量会话"
        "（逐笔校验、零点漂移/回程间隙校正、定稿冻结并生成测试批次），"
        "以及子午线复测研究（4~16 份冻结版本的二次角向谐波拟合、"
        "镜面/架位像散分离、留一稳定性、定稿冻结与复制追加）。",
    )

    @app.exception_handler(NotFoundError)
    async def _not_found(_, exc: NotFoundError):
        return JSONResponse(status_code=404, content={"detail": {"message": str(exc)}})

    @app.exception_handler(ConflictError)
    async def _conflict(_, exc: ConflictError):
        return JSONResponse(status_code=409, content={"detail": {"message": str(exc)}})

    @app.exception_handler(AnalysisError)
    async def _analysis_error(_, exc: AnalysisError):
        return JSONResponse(status_code=422, content={"detail": {"message": str(exc)}})

    @app.exception_handler(CorrectionError)
    async def _correction_error(_, exc: CorrectionError):
        return JSONResponse(status_code=422, content={"detail": {"message": str(exc)}})

    @app.exception_handler(MaskError)
    async def _mask_error(_, exc: MaskError):
        return JSONResponse(
            status_code=422,
            content={"detail": {"message": "遮罩设计校验失败", "errors": exc.errors}},
        )

    @app.exception_handler(MeridianError)
    async def _meridian_error(_, exc: MeridianError):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "message": str(exc),
                    "errors": exc.gaps,
                }
            },
        )

    # ---------------- 测试批次 ----------------

    def _create_test_from_mask(payload: TestCreateIn) -> dict:
        """引用遮罩版本创建批次：常量与环带边界取自遮罩版本冻结值。"""
        if payload.mask_version_no is not None:
            mv = db.get_mask_version(payload.mask_scheme_id, payload.mask_version_no)
        else:
            mv = db.get_latest_mask_version(payload.mask_scheme_id)
        mp = mv["params"]  # 冻结常量（mm）
        unit = payload.unit or "mm"
        factor = UNIT_TO_MM[unit]
        mismatches = []
        for name, provided, frozen_mm in (
            ("diameter", payload.diameter, mp["diameter"]),
            (
                "radius_of_curvature",
                payload.radius_of_curvature,
                mp["radius_of_curvature"],
            ),
        ):
            if provided is not None and abs(provided * factor - frozen_mm) > 1e-6:
                mismatches.append(
                    f"{name}={provided} 与遮罩版本冻结值 {frozen_mm} mm 不一致"
                )
        if (
            payload.conic_constant is not None
            and abs(payload.conic_constant - mp["conic_constant"]) > 1e-9
        ):
            mismatches.append(
                f"conic_constant={payload.conic_constant} 与遮罩版本冻结值 "
                f"{mp['conic_constant']} 不一致"
            )
        if payload.source_mode is not None and payload.source_mode != mp["source_mode"]:
            mismatches.append(
                f"source_mode={payload.source_mode} 与遮罩版本冻结值 "
                f"{mp['source_mode']} 不一致"
            )
        if mismatches:
            raise _err(422, "与遮罩版本冻结常量不一致", mismatches)
        if payload.zone_readings is None:
            raise _err(422, "引用遮罩版本时需提供 zone_readings（按遮罩环带顺序）")
        mzones = mv["layout"]["zones"]
        if len(payload.zone_readings) != len(mzones):
            raise _err(
                422,
                f"zone_readings 数量 {len(payload.zone_readings)} 与遮罩环带数 "
                f"{len(mzones)} 不一致",
            )
        errs = _validate_constants(
            mp["diameter"],
            mp["radius_of_curvature"],
            mp["conic_constant"],
            payload.wavelength_nm,
            payload.instrument_offset,
        )
        zone_objs = [
            SimpleNamespace(
                inner_radius=z["inner_radius"],
                outer_radius=z["outer_radius"],
                readings=rd,
            )
            for z, rd in zip(mzones, payload.zone_readings)
        ]
        errs += _validate_zones(
            mp["diameter"], zone_objs, payload.options.min_readings_per_zone
        )
        if errs:
            raise _err(422, "测试数据校验失败", errs)
        zones = [
            {
                "inner_radius": z["inner_radius"],
                "outer_radius": z["outer_radius"],
                "readings": [(r * factor, r) for r in rd],
            }
            for z, rd in zip(mzones, payload.zone_readings)
        ]
        record = {
            "name": payload.name,
            "notes": payload.notes,
            "unit": unit,
            "source_mode": mp["source_mode"],
            "diameter": mp["diameter"],
            "radius_of_curvature": mp["radius_of_curvature"],
            "conic_constant": mp["conic_constant"],
            "wavelength_nm": payload.wavelength_nm,
            "instrument_offset": payload.instrument_offset * factor,
            "options": payload.options.model_dump(),
            "mask_scheme_id": mv["scheme_id"],
            "mask_version_no": mv["version_no"],
        }
        test_id = db.create_test(record, zones)
        version = _run_analysis(db, test_id)
        return {
            "test": db.get_test(test_id),
            "version": _version_brief(version),
            "mask": {"scheme_id": mv["scheme_id"], "version_no": mv["version_no"]},
        }

    @app.post("/api/tests", status_code=201)
    def create_test(payload: TestCreateIn):
        if payload.mask_scheme_id is not None:
            return _create_test_from_mask(payload)
        missing = [
            name
            for name, val in (
                ("diameter", payload.diameter),
                ("radius_of_curvature", payload.radius_of_curvature),
                ("conic_constant", payload.conic_constant),
                ("source_mode", payload.source_mode),
                ("unit", payload.unit),
            )
            if val is None
        ]
        if payload.zones is None:
            missing.append("zones")
        if missing:
            raise _err(
                422,
                f"缺少必填字段：{', '.join(missing)}"
                "（或提供 mask_scheme_id 引用遮罩版本）",
            )
        errs = _validate_payload(payload)
        if errs:
            raise _err(422, "测试数据校验失败", errs)
        factor = UNIT_TO_MM[payload.unit]
        zones_sorted = sorted(payload.zones, key=lambda z: (z.inner_radius, z.outer_radius))
        zones = [
            {
                "inner_radius": z.inner_radius * factor,
                "outer_radius": z.outer_radius * factor,
                "readings": [(r * factor, r) for r in z.readings],
            }
            for z in zones_sorted
        ]
        record = {
            "name": payload.name,
            "notes": payload.notes,
            "unit": payload.unit,
            "source_mode": payload.source_mode,
            "diameter": payload.diameter * factor,
            "radius_of_curvature": payload.radius_of_curvature * factor,
            "conic_constant": payload.conic_constant,
            "wavelength_nm": payload.wavelength_nm,
            "instrument_offset": payload.instrument_offset * factor,
            "options": payload.options.model_dump(),
        }
        test_id = db.create_test(record, zones)
        version = _run_analysis(db, test_id)
        return {
            "test": db.get_test(test_id),
            "version": _version_brief(version),
        }

    @app.get("/api/tests")
    def list_tests():
        return {"tests": db.list_tests()}

    @app.get("/api/tests/{test_id}")
    def get_test(test_id: int):
        test = db.get_test(test_id)
        versions = db.list_versions(test_id)
        return {"test": test, "versions": versions}

    # ---------------- 读数冻结 / 剔除 / 恢复 ----------------

    @app.post("/api/tests/{test_id}/readings/freeze")
    def freeze_readings(test_id: int, payload: FreezeIn):
        db.get_test(test_id)
        if payload.all_valid:
            count = db.set_frozen(test_id, None)
        elif payload.reading_ids:
            for rid in payload.reading_ids:
                r = db.get_reading(rid)
                if r["test_id"] != test_id:
                    raise _err(409, f"读数 {rid} 不属于测试 {test_id}")
            count = db.set_frozen(test_id, payload.reading_ids)
        else:
            raise _err(422, "请提供 reading_ids 或设置 all_valid=true")
        return {"frozen": count}

    @app.post("/api/tests/{test_id}/readings/exclude")
    def exclude_readings(test_id: int, payload: ExcludeIn):
        test = db.get_test(test_id)
        min_n = test["options"]["min_readings_per_zone"]
        # 先整体校验，再执行，避免半提交状态
        targets = []
        for item in payload.items:
            r = db.get_reading(item.reading_id)
            if r["test_id"] != test_id:
                raise _err(409, f"读数 {item.reading_id} 不属于测试 {test_id}")
            if r["frozen"]:
                raise _err(409, f"读数 {item.reading_id} 已冻结，不能剔除")
            targets.append((item, r))
        # 模拟剔除后每区剩余有效读数
        remaining: dict[int, int] = {}
        for z in test["zones"]:
            remaining[z["id"]] = sum(1 for r in z["readings"] if not r["excluded"])
        for item, r in targets:
            if not r["excluded"]:
                remaining[r["zone_id"]] -= 1
        for z in test["zones"]:
            if remaining[z["id"]] < min_n:
                raise _err(
                    409,
                    f"剔除后分区 {z['zone_index']} 有效读数将不足 "
                    f"（{remaining[z['id']]} < {min_n}），请先恢复其他读数或调整选项",
                )
        results = []
        for item, r in targets:
            db.set_excluded(item.reading_id, True, item.reason)
            results.append(
                {
                    "reading_id": item.reading_id,
                    "status": "updated" if r["excluded"] else "excluded",
                    "reason": item.reason,
                }
            )
        out = {"results": results}
        if payload.analyze:
            out["version"] = _version_brief(_run_analysis(db, test_id))
        return out

    @app.post("/api/tests/{test_id}/readings/restore")
    def restore_readings(test_id: int, payload: RestoreIn):
        db.get_test(test_id)
        for rid in payload.reading_ids:
            r = db.get_reading(rid)
            if r["test_id"] != test_id:
                raise _err(409, f"读数 {rid} 不属于测试 {test_id}")
        for rid in payload.reading_ids:
            db.set_excluded(rid, False, None)
        out = {"restored": len(payload.reading_ids)}
        if payload.analyze:
            out["version"] = _version_brief(_run_analysis(db, test_id))
        return out

    # ---------------- 分析版本 ----------------

    @app.post("/api/tests/{test_id}/analyze", status_code=201)
    def analyze(test_id: int, payload: AnalyzeIn):
        db.get_test(test_id)
        options_dict = payload.options.model_dump() if payload.options else None
        version = _run_analysis(db, test_id, options_dict)
        return {"version": _version_brief(version)}

    @app.get("/api/tests/{test_id}/versions")
    def list_versions(test_id: int):
        db.get_test(test_id)
        return {"versions": db.list_versions(test_id)}

    @app.get("/api/tests/{test_id}/versions/{version_no}")
    def get_version(test_id: int, version_no: int):
        v = db.get_version(test_id, version_no)
        return {
            "test_id": v["test_id"],
            "version_no": v["version_no"],
            "created_at": v["created_at"],
            "input_hash": v["input_hash"],
            "snapshot": v["snapshot"],
            "result": v["result"],
        }

    # ---------------- 批次对比 ----------------

    @app.post("/api/compare")
    def compare(payload: CompareIn):
        entries = []
        for e in payload.entries:
            db.get_test(e.test_id)
            v = (
                db.get_version(e.test_id, e.version_no)
                if e.version_no is not None
                else db.get_latest_version(e.test_id)
            )
            entries.append(v)
        # 遮罩一致性：各区等效半径序列完全对齐时给出逐区差值
        r_seqs = [
            [z["r_mean"] for z in v["result"]["zones"]] for v in entries
        ]
        aligned = all(
            len(rs) == len(r_seqs[0])
            and all(abs(a - b) < 1e-9 for a, b in zip(rs, r_seqs[0]))
            for rs in r_seqs
        )
        out_entries = []
        for v in entries:
            out_entries.append(
                {
                    "test_id": v["test_id"],
                    "version_no": v["version_no"],
                    "input_hash": v["input_hash"],
                    "summary": v["result"]["summary"],
                    "fit": v["result"]["fit"],
                    "zones": [
                        {
                            "index": z["index"],
                            "r_mean": z["r_mean"],
                            "wavefront_nm": z["wavefront_nm"],
                            "wavefront_waves": z["wavefront_waves"],
                        }
                        for z in v["result"]["zones"]
                    ],
                }
            )
        resp = {"entries": out_entries, "mask_aligned": aligned}
        if aligned:
            base = entries[0]["result"]["zones"]
            zonal = []
            for i, bz in enumerate(base):
                row = {
                    "index": i,
                    "r_mean": bz["r_mean"],
                    "wavefront_nm": [
                        v["result"]["zones"][i]["wavefront_nm"] for v in entries
                    ],
                    "delta_vs_first_nm": [
                        v["result"]["zones"][i]["wavefront_nm"] - bz["wavefront_nm"]
                        for v in entries
                    ],
                }
                zonal.append(row)
            resp["zonal_comparison"] = zonal
        return resp

    # ---------------- 修正量搜索 ----------------

    @app.post("/api/tests/{test_id}/corrections/search")
    def corrections_search(test_id: int, payload: CorrectionSearchIn):
        test = db.get_test(test_id)
        v = (
            db.get_version(test_id, payload.version_no)
            if payload.version_no is not None
            else db.get_latest_version(test_id)
        )
        zones = v["result"]["zones"]
        try:
            result = search_corrections(
                r_mean=[z["r_mean"] for z in zones],
                inner=[z["inner_radius"] for z in zones],
                outer=[z["outer_radius"] for z in zones],
                residual_surface_nm=[z["wavefront_nm"] / 2.0 for z in zones],
                wavelength_nm=test["wavelength_nm"],
                diameter=test["diameter"],
                max_removal_nm=payload.max_removal_nm,
                edge_zone_max_removal_nm=payload.edge_zone_max_removal_nm,
                preserve_edge=payload.preserve_edge,
                max_mean_removal_nm=payload.max_mean_removal_nm,
                smoothing_weights=payload.smoothing_weights,
                refit_defocus=payload.refit_defocus,
            )
        except CorrectionError as exc:
            raise _err(422, str(exc)) from exc
        result["based_on"] = {"test_id": test_id, "version_no": v["version_no"]}
        return result

    # ---------------- Couder 遮罩方案 ----------------

    def _mask_params_of(p: MaskParamsIn) -> MaskParams:
        f = UNIT_TO_MM[p.unit]
        return MaskParams(
            diameter=p.diameter * f,
            radius_of_curvature=p.radius_of_curvature * f,
            conic_constant=p.conic_constant,
            source_mode=p.source_mode,
            zone_count=p.zone_count,
            weighting=p.weighting,
            weights=p.weights,
            center_exclusion_radius=p.center_exclusion_radius * f,
            min_zone_width=p.min_zone_width * f,
            bridge_width=p.bridge_width * f,
            knife_resolution=p.knife_resolution * f,
            print_scale=p.print_scale,
        )

    def _create_mask_version_idem(
        scheme_id: int, params: MaskParams, layout: dict | None = None
    ) -> dict:
        """计算布局并写入遮罩版本；参数与最新版本一致时复用（幂等）。"""
        if layout is None:
            layout = compute_layout(params)  # MaskError → 422
        norm = params.normalized()
        params_hash = compute_input_hash(norm)
        try:
            latest = db.get_latest_mask_version(scheme_id)
        except NotFoundError:
            latest = None
        if latest is not None and latest["params_hash"] == params_hash:
            latest["reused"] = True
            return latest
        version = db.create_mask_version(scheme_id, norm, layout, params_hash)
        version["reused"] = False
        return version

    def _mask_version_brief(v: dict) -> dict:
        return {
            "scheme_id": v["scheme_id"],
            "version_no": v["version_no"],
            "created_at": v["created_at"],
            "params_hash": v["params_hash"],
            "metrics": v["layout"].get("metrics", {}),
            "reused": v.get("reused", False),
        }

    @app.post("/api/mask-schemes", status_code=201)
    def create_mask_scheme(payload: MaskSchemeCreateIn):
        params = _mask_params_of(payload)
        # 先完成布局校验再落库：校验失败时不留下任何方案数据
        layout = compute_layout(params)  # MaskError → 422
        scheme_id = db.create_mask_scheme(payload.name, payload.notes)
        version = _create_mask_version_idem(scheme_id, params, layout)
        return {
            "scheme": db.get_mask_scheme(scheme_id),
            "version": _mask_version_brief(version),
        }

    @app.get("/api/mask-schemes")
    def list_mask_schemes():
        return {"schemes": db.list_mask_schemes()}

    @app.post("/api/mask-schemes/search")
    def search_mask(payload: MaskSearchIn):
        if payload.zone_count_max < payload.zone_count_min:
            raise _err(422, "zone_count_max 不能小于 zone_count_min")
        bw_min = payload.bridge_width_min
        bw_max = (
            payload.bridge_width_max
            if payload.bridge_width_max is not None
            else bw_min
        )
        zw_min = payload.zone_width_min
        zw_max = (
            payload.zone_width_max if payload.zone_width_max is not None else zw_min
        )
        if bw_max < bw_min:
            raise _err(422, "bridge_width_max 不能小于 bridge_width_min")
        if zw_max < zw_min:
            raise _err(422, "zone_width_max 不能小于 zone_width_min")
        counts = list(range(payload.zone_count_min, payload.zone_count_max + 1))
        bridges = _frange(bw_min, bw_max, payload.bridge_width_step)
        widths = _frange(zw_min, zw_max, payload.zone_width_step)
        combos = (
            len(counts) * len(widths) * len(bridges) * len(payload.layouts)
        )
        if combos > SEARCH_COMBO_LIMIT:
            raise _err(
                422,
                f"搜索组合 {combos} 个超出上限 {SEARCH_COMBO_LIMIT}"
                "（分区数 × 环宽 × 桥宽 × 权重模式），请缩小范围或增大步长",
            )
        f = UNIT_TO_MM[payload.unit]
        base = MaskParams(
            diameter=payload.diameter * f,
            radius_of_curvature=payload.radius_of_curvature * f,
            conic_constant=payload.conic_constant,
            source_mode=payload.source_mode,
            zone_count=payload.zone_count_min,
            weighting="equal_area",
            weights=None,
            center_exclusion_radius=payload.center_exclusion_radius * f,
            min_zone_width=zw_min * f,
            bridge_width=bw_min * f,
            knife_resolution=payload.knife_resolution * f,
            print_scale=1.0,
        )
        const_errs = validate_mask_constants(base)
        if const_errs:
            raise _err(422, "遮罩常量校验失败", const_errs)
        result = search_layouts(
            base,
            counts,
            [w * f for w in widths],
            [b * f for b in bridges],
            list(payload.layouts),
            top=payload.top,
        )
        return result

    @app.get("/api/mask-schemes/{scheme_id}")
    def get_mask_scheme(scheme_id: int):
        return {"scheme": db.get_mask_scheme(scheme_id)}

    @app.post("/api/mask-schemes/{scheme_id}/versions", status_code=201)
    def create_mask_version(scheme_id: int, payload: MaskParamsIn):
        db.get_mask_scheme(scheme_id)
        version = _create_mask_version_idem(scheme_id, _mask_params_of(payload))
        return {"version": _mask_version_brief(version)}

    @app.get("/api/mask-schemes/{scheme_id}/versions/{version_no}")
    def get_mask_version(scheme_id: int, version_no: int):
        v = db.get_mask_version(scheme_id, version_no)
        return {
            "scheme_id": v["scheme_id"],
            "version_no": v["version_no"],
            "created_at": v["created_at"],
            "params_hash": v["params_hash"],
            "params": v["params"],
            "layout": v["layout"],
        }

    @app.get("/api/mask-schemes/{scheme_id}/versions/{version_no}/svg")
    def mask_svg(scheme_id: int, version_no: int):
        scheme = db.get_mask_scheme(scheme_id)
        v = db.get_mask_version(scheme_id, version_no)
        name = scheme.get("name") or "遮罩方案"
        svg = render_mask_svg(
            v["params"],
            v["layout"],
            title=f"Couder 遮罩 {name} #{scheme_id} v{version_no}",
            subtitle=f"生成 {v['created_at']}",
        )
        return Response(content=svg, media_type="image/svg+xml")

    # ---------------- 往返测量会话 ----------------

    def _get_mask_version_for_session(scheme_id: int, version_no: int | None) -> dict:
        if version_no is not None:
            return db.get_mask_version(scheme_id, version_no)
        return db.get_latest_mask_version(scheme_id)

    def _current_readings_map(session: dict) -> dict:
        """{seq: 当前有效读数行}；含漏测的 seq 映射为 None。"""
        current: dict[int, dict | None] = {s["seq"]: None for s in session["plan"]}
        # 取每个 seq 最大 attempt 的未剔除读数
        best: dict[int, dict] = {}
        for r in session["readings"]:
            if r["excluded"]:
                continue
            old = best.get(r["seq"])
            if old is None or r["attempt"] > old["attempt"]:
                best[r["seq"]] = r
        current.update(best)
        return current

    def _session_correction(session: dict) -> dict:
        current = _current_readings_map(session)
        inputs: dict[int, dict | None] = {}
        for seq, r in current.items():
            if r is None:
                inputs[seq] = None
            else:
                slot = next(s for s in session["plan"] if s["seq"] == seq)
                inputs[seq] = {
                    "value_mm": r["knife_position"],
                    "epoch": datetime.fromisoformat(r["collected_at"]).timestamp(),
                    "direction": slot["direction"],
                    "locked": bool(r["locked"]),
                }
        return correct_session(
            session["plan"],
            inputs,
            session["thresholds"],
            fit_direction=session["start_direction"],
        )

    def _slot_of(session: dict, seq: int) -> dict:
        for s in session["plan"]:
            if s["seq"] == seq:
                return s
        raise _err(422, f"测次 #{seq} 不在本会话计划内（计划共 "
                        f"{len(session['plan'])} 个测次）")

    def _require_writable(session: dict, seq: int | None = None) -> None:
        if session["status"] == "finalized":
            target = f"测次 #{seq} " if seq is not None else ""
            raise _err(
                409,
                f"会话 {session['id']} 已定稿（批次 {session['test_id']}），"
                f"{target}不能在定稿后追加或修改，原始记录已冻结",
            )

    def _validate_reading_payload(session: dict, slot: dict, payload, *, retest: bool) -> None:
        """逐笔读数的业务校验（类型/字面量由 Pydantic 拦截）。"""
        errs: list[str] = []
        if payload.zone_index != slot["zone_index"]:
            errs.append(
                f"测次 #{slot['seq']} 计划分区为 {slot['zone_index']}，"
                f"提交为 {payload.zone_index}（漏测或跳步请先补该测次）"
            )
        if payload.direction != slot["direction"]:
            errs.append(
                f"测次 #{slot['seq']} 计划移动方向为 {slot['direction']}，"
                f"提交为 {payload.direction}"
            )
        collected = payload.collected_at
        if collected.tzinfo is None:
            collected = collected.replace(tzinfo=timezone.utc)
        created_at = datetime.fromisoformat(session["created_at"])
        if collected < created_at - timedelta(seconds=1):
            errs.append(
                f"测次 #{slot['seq']} 采集时间 {iso_dt(collected)} 早于建会话时间 "
                f"{session['created_at']}"
            )
        if session.get("deadline"):
            deadline = datetime.fromisoformat(session["deadline"])
            if collected > deadline + timedelta(seconds=1):
                errs.append(
                    f"测次 #{slot['seq']} 采集时间 {iso_dt(collected)} 超出采集时限 "
                    f"{session['deadline']}"
                )
        if not math.isfinite(payload.knife_position):
            errs.append(f"测次 #{slot['seq']} 刀口位置必须是有限数值")
        if errs:
            raise _err(422, "读数校验失败", errs)
        # 重复提交：非补测时同一计划位已有有效读数
        if not retest:
            existing = [
                r for r in session["readings"]
                if r["seq"] == slot["seq"] and not r["excluded"]
            ]
            if existing:
                raise _err(
                    409,
                    f"测次 #{slot['seq']} 已有有效读数（attempt "
                    f"{max(r['attempt'] for r in existing)}）：同一计划位重复提交被拒绝，"
                    "如需覆盖请调用补测接口",
                )

    def _next_expected_seq(session: dict) -> int | None:
        current = _current_readings_map(session)
        for s in session["plan"]:
            if current[s["seq"]] is None:
                return s["seq"]
        return None

    def _session_brief(session: dict) -> dict:
        return {
            "id": session["id"],
            "name": session["name"],
            "status": session["status"],
            "created_at": session["created_at"],
            "deadline": session.get("deadline"),
            "unit": session["unit"],
            "repeats_per_zone": session["repeats_per_zone"],
            "start_direction": session["start_direction"],
            "reference_zone_index": session["reference_zone_index"],
            "reference_refresh": session["reference_refresh"],
            "mask_scheme_id": session["mask_scheme_id"],
            "mask_version_no": session["mask_version_no"],
            "test_id": session.get("test_id"),
            "input_hash": session.get("input_hash"),
        }

    def _session_detail(session: dict, *, with_correction: bool = False) -> dict:
        out = _session_brief(session)
        out.update(
            {
                "notes": session["notes"],
                "wavelength_nm": session["wavelength_nm"],
                "instrument_offset": session["instrument_offset"],
                "thresholds": session["thresholds"],
                "options": session["options"],
                "n_plan_slots": len(session["plan"]),
                "plan": session["plan"],
                "readings": [
                    {
                        "id": r["id"],
                        "seq": r["seq"],
                        "attempt": r["attempt"],
                        "zone_index": r["zone_index"],
                        "direction": r["direction"],
                        "kind": r["kind"],
                        "knife_position": r["knife_position"],
                        "knife_position_original": r["knife_position_original"],
                        "collected_at": r["collected_at"],
                        "excluded": bool(r["excluded"]),
                        "exclude_reason": r["exclude_reason"],
                        "locked": bool(r["locked"]),
                        "frozen": bool(r["frozen"]),
                    }
                    for r in session["readings"]
                ],
            }
        )
        if with_correction:
            out["quality"] = _session_correction(session)
        return out

    @app.post("/api/sessions", status_code=201)
    def create_session(payload: SessionCreateIn):
        mv = _get_mask_version_for_session(
            payload.mask_scheme_id, payload.mask_version_no
        )
        mp = mv["params"]
        zone_count = len(mv["layout"]["zones"])
        if payload.reference_zone_index >= zone_count:
            raise _err(
                422,
                f"参考区序号 {payload.reference_zone_index} 超出遮罩分区数 {zone_count}",
            )
        factor = UNIT_TO_MM[payload.unit]
        try:
            plan = build_plan(
                zone_count,
                payload.repeats_per_zone,
                payload.start_direction,
                payload.reference_zone_index,
                payload.reference_refresh,
            )
        except SessionError as exc:
            raise _err(422, str(exc)) from exc
        resolution_mm = mp["knife_resolution"]
        thresholds_in = payload.thresholds
        thresholds = {
            mapped: (
                getattr(thresholds_in, key) * factor
                if getattr(thresholds_in, key) is not None
                else resolution_mm
            )
            for key, mapped in (
                ("max_dispersion", "max_dispersion_mm"),
                ("max_direction_diff", "max_direction_diff_mm"),
                ("max_drift_residual", "max_drift_residual_mm"),
            )
        }
        if payload.options is not None:
            options = payload.options.model_dump()
        else:
            options = AnalysisOptionsIn(
                min_readings_per_zone=2 * payload.repeats_per_zone
            ).model_dump()
        deadline = iso_dt(payload.collect_deadline)
        collected_check = payload.collect_deadline
        if collected_check.tzinfo is None:
            collected_check = collected_check.replace(tzinfo=timezone.utc)
        else:
            collected_check = collected_check.astimezone(timezone.utc)
        if collected_check <= datetime.now(timezone.utc):
            raise _err(422, "采集时限必须晚于当前时间")
        record = {
            "name": payload.name,
            "notes": payload.notes,
            "deadline": deadline,
            "unit": payload.unit,
            "repeats_per_zone": payload.repeats_per_zone,
            "start_direction": payload.start_direction,
            "reference_zone_index": payload.reference_zone_index,
            "reference_refresh": payload.reference_refresh,
            "mask_scheme_id": mv["scheme_id"],
            "mask_version_no": mv["version_no"],
            "wavelength_nm": payload.wavelength_nm,
            "instrument_offset": payload.instrument_offset * factor,
            "thresholds": thresholds,
            "options": options,
        }
        session_id = db.create_session(record, plan)
        session = db.get_session(session_id)
        return {"session": _session_detail(session)}

    @app.get("/api/sessions")
    def list_sessions():
        return {"sessions": db.list_sessions()}

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: int, correction: bool = False):
        session = db.get_session(session_id)
        return {"session": _session_detail(session, with_correction=correction)}

    @app.get("/api/sessions/{session_id}/quality")
    def session_quality(session_id: int):
        session = db.get_session(session_id)
        report = _session_correction(session)
        report["session_id"] = session_id
        report["thresholds"] = session["thresholds"]
        report["status"] = session["status"]
        return report

    @app.post("/api/sessions/{session_id}/readings", status_code=201)
    def submit_reading(session_id: int, payload: ReadingSubmitIn):
        session = db.get_session(session_id)
        _require_writable(session, payload.seq)
        slot = _slot_of(session, payload.seq)
        expected = _next_expected_seq(session)
        if expected is not None and payload.seq != expected:
            kind = "跳步" if payload.seq > expected else "重复提交"
            raise _err(
                409,
                f"{kind}：下一个待采测次为 #{expected}，收到测次 #{payload.seq}；"
                f"漏测测次请在采完后通过补测接口补回",
            )
        _validate_reading_payload(session, slot, payload, retest=False)
        factor = UNIT_TO_MM[session["unit"]]
        reading = db.add_session_reading(
            session_id,
            slot,
            {
                "knife_position_mm": payload.knife_position * factor,
                "knife_position_original": payload.knife_position,
                "collected_at": iso_dt(payload.collected_at),
            },
        )
        if session["status"] == "confirmed":
            db.set_session_status(session_id, "collecting")
        return {
            "reading_id": reading["id"],
            "seq": slot["seq"],
            "attempt": reading["attempt"],
            "next_expected_seq": _next_expected_seq(db.get_session(session_id)),
        }

    @app.post("/api/sessions/{session_id}/readings/retest", status_code=201)
    def retest_reading(session_id: int, payload: RetestIn):
        session = db.get_session(session_id)
        _require_writable(session, payload.seq)
        slot = _slot_of(session, payload.seq)
        current = _current_readings_map(session)
        has_existing = current.get(payload.seq) is not None
        expected = _next_expected_seq(session)
        # 补测只允许两类测次：已有有效读数的测次（覆盖旧 attempt），
        # 或当前应采测次（漏测后补在计划位置）。其他一律视为乱序。
        if not has_existing and expected != payload.seq:
            if expected is None:
                raise _err(
                    409,
                    f"测次 #{payload.seq} 尚无读数，但计划已全部采集完毕；"
                    "补测只能针对已有读数的测次",
                )
            raise _err(
                409,
                f"乱序补测：测次 #{payload.seq} 尚无读数，当前应采测次为 "
                f"#{expected}，请先补该测次",
            )
        # 锁定测位需先解锁才能补测（当前实现锁定即视为可信，不提供单独解锁）
        locked = [
            r for r in session["readings"]
            if r["seq"] == payload.seq and r["locked"] and not r["excluded"]
        ]
        if locked:
            raise _err(
                409,
                f"测次 #{payload.seq} 读数已锁定为可信，不能补测覆盖",
            )
        _validate_reading_payload(session, slot, payload, retest=True)
        factor = UNIT_TO_MM[session["unit"]]
        reading = db.add_session_reading(
            session_id,
            slot,
            {
                "knife_position_mm": payload.knife_position * factor,
                "knife_position_original": payload.knife_position,
                "collected_at": iso_dt(payload.collected_at),
            },
        )
        if session["status"] == "confirmed":
            db.set_session_status(session_id, "collecting")
        return {
            "reading_id": reading["id"],
            "seq": slot["seq"],
            "attempt": reading["attempt"],
        }

    @app.post("/api/sessions/{session_id}/readings/exclude")
    def exclude_session_readings(session_id: int, payload: SessionExcludeIn):
        session = db.get_session(session_id)
        _require_writable(session)
        targets = []
        for item in payload.items:
            r = db.get_session_reading(item.reading_id)
            if r["session_id"] != session_id:
                raise _err(
                    409, f"读数 {item.reading_id} 不属于会话 {session_id}"
                )
            if r["frozen"]:
                raise _err(409, f"读数 {item.reading_id} 已随定稿冻结，不能剔除")
            if r["locked"]:
                raise _err(409, f"测次 #{r['seq']} 读数已锁定为可信，不能剔除")
            targets.append((item, r))
        for item, r in targets:
            db.set_session_reading_excluded(item.reading_id, True, item.reason)
        if session["status"] == "confirmed":
            db.set_session_status(session_id, "collecting")
        session = db.get_session(session_id)
        return {
            "excluded": len(targets),
            "quality": _session_correction(session),
        }

    @app.post("/api/sessions/{session_id}/lock")
    def lock_session(session_id: int, payload: SessionLockIn):
        session = db.get_session(session_id)
        _require_writable(session)
        if payload.all_collected:
            count = db.lock_session_slots(session_id, None)
        elif payload.seqs:
            plan_seqs = {s["seq"] for s in session["plan"]}
            bad = [n for n in payload.seqs if n not in plan_seqs]
            if bad:
                raise _err(422, f"测次 {bad} 不在本会话计划内")
            current = _current_readings_map(session)
            missing = [n for n in payload.seqs if current.get(n) is None]
            if missing:
                raise _err(409, f"测次 {missing} 尚无有效读数，不能锁定")
            count = db.lock_session_slots(session_id, payload.seqs)
        else:
            raise _err(422, "请提供 seqs 或设置 all_collected=true")
        return {"locked": count}

    @app.post("/api/sessions/{session_id}/confirm")
    def confirm_session(session_id: int):
        session = db.get_session(session_id)
        _require_writable(session)
        report = _session_correction(session)
        if not report["can_finalize"]:
            raise _err(
                409,
                "存在阻断性测次问题，不能确认",
                [v["message"] for v in report["blocking_violations"]],
            )
        db.set_session_status(session_id, "confirmed")
        return {"session": _session_brief(db.get_session(session_id)), "quality": report}

    def _build_freeze_bundle(session: dict, report: dict, mv: dict) -> dict:
        """定稿冻结包：会话参数、遮罩版本指纹、计划、全部原始 attempt、校正参数。"""
        return {
            "session": {
                "id": session["id"],
                "name": session["name"],
                "unit": session["unit"],
                "repeats_per_zone": session["repeats_per_zone"],
                "start_direction": session["start_direction"],
                "reference_zone_index": session["reference_zone_index"],
                "reference_refresh": session["reference_refresh"],
                "wavelength_nm": session["wavelength_nm"],
                "instrument_offset_mm": session["instrument_offset"],
                "thresholds_mm": session["thresholds"],
                "options": session["options"],
            },
            "mask": {
                "scheme_id": mv["scheme_id"],
                "version_no": mv["version_no"],
                "params": mv["params"],
                "params_hash": mv["params_hash"],
            },
            "plan": session["plan"],
            "raw_readings": [
                {
                    "id": r["id"],
                    "seq": r["seq"],
                    "attempt": r["attempt"],
                    "zone_index": r["zone_index"],
                    "direction": r["direction"],
                    "kind": r["kind"],
                    "knife_position_mm": r["knife_position"],
                    "knife_position_original": r["knife_position_original"],
                    "collected_at": r["collected_at"],
                    "excluded": bool(r["excluded"]),
                    "exclude_reason": r["exclude_reason"],
                    "locked": bool(r["locked"]),
                }
                for r in session["readings"]
            ],
            "correction": {
                "drift": {
                    "status": report["drift"]["status"],
                    "t0_epoch": report["drift"]["t0_epoch"],
                    "slope_mm_per_s": report["drift"]["slope_mm_per_s"],
                    "intercept_mm": report["drift"]["intercept_mm"],
                    "fit_observations": report["drift"]["fit_observations"],
                },
                "backlash_mm": report["backlash"]["global_mm"],
            },
        }

    @app.post("/api/sessions/{session_id}/finalize")
    def finalize_session(session_id: int):
        session = db.get_session(session_id)
        # 幂等：已定稿（顺序或并发重复）返回同一批次
        if session["status"] == "finalized":
            return {
                "session": _session_brief(session),
                "test_id": session["test_id"],
                "reused": True,
            }
        # 定稿前必须先进入待确认状态（confirm 已做过阻断性检查）
        if session["status"] != "confirmed":
            raise _err(
                409,
                f"会话 {session_id} 当前为 {session['status']} 状态，"
                "定稿前必须先调用 /confirm 进入待确认状态",
            )
        report = _session_correction(session)
        if not report["can_finalize"]:
            raise _err(
                409,
                "存在阻断性测次问题，不能定稿（请补测、排除异常或锁定可信读数）",
                [v["message"] for v in report["blocking_violations"]],
            )
        mv = db.get_mask_version(session["mask_scheme_id"], session["mask_version_no"])
        mp = mv["params"]
        mzones = mv["layout"]["zones"]
        # 每区校正读数（正反向均值已对齐）作为批次读数，顺序与遮罩环带一致
        zone_readings_mm = []
        for row, mz in zip(report["zones"], mzones):
            vals = [e["corrected_mm"] for e in row["forward"] + row["reverse"]]
            if not vals:
                raise _err(409, f"分区 {row['zone_index']} 无有效读数，不能定稿")
            zone_readings_mm.append(vals)
        freeze = _build_freeze_bundle(session, report, mv)
        input_hash = compute_input_hash(freeze)
        freeze_str = json.dumps(freeze, sort_keys=True, ensure_ascii=True)
        record = {
            "name": session["name"] or f"会话 {session_id} 定稿批次",
            "notes": session["notes"],
            "unit": "mm",  # 校正读数内部已是 mm
            "source_mode": mp["source_mode"],
            "diameter": mp["diameter"],
            "radius_of_curvature": mp["radius_of_curvature"],
            "conic_constant": mp["conic_constant"],
            "wavelength_nm": session["wavelength_nm"],
            "instrument_offset": session["instrument_offset"],
            "options": session["options"],
            "mask_scheme_id": mv["scheme_id"],
            "mask_version_no": mv["version_no"],
            "session_id": session_id,
        }
        zones = [
            {
                "inner_radius": mz["inner_radius"],
                "outer_radius": mz["outer_radius"],
                "readings": [(v, v) for v in vals],
            }
            for mz, vals in zip(mzones, zone_readings_mm)
        ]
        # 建批次 → 分析 → 原子 CAS 翻转状态。并发请求只有一个能翻转成功；
        # 落败者删除自己刚建的批次并返回胜出者的同一 test_id，不留孤儿。
        test_id = db.create_test(record, zones)
        try:
            version = _run_analysis(db, test_id)
        except Exception:
            db.delete_test(test_id)
            raise
        outcome = db.finalize_session(session_id, input_hash, freeze_str, test_id)
        if outcome["status"] != "finalized":
            db.delete_test(test_id)
            if outcome["status"] == "reused":
                settled = db.get_session(session_id)
                return {
                    "session": _session_brief(settled),
                    "test_id": outcome["test_id"],
                    "reused": True,
                }
            # stale：会话在 confirm 之后又被改回采集中
            raise _err(
                409,
                f"会话 {session_id} 在定稿前已离开待确认状态，请重新确认后再定稿",
            )
        return {
            "session": _session_brief(db.get_session(session_id)),
            "test_id": test_id,
            "version": _version_brief(version),
            "input_hash": input_hash,
            "reused": False,
        }

    @app.get("/api/sessions/{session_id}/freeze")
    def get_session_freeze(session_id: int):
        session = db.get_session(session_id)
        if session["status"] != "finalized":
            raise _err(409, f"会话 {session_id} 尚未定稿，无冻结包")
        return {
            "input_hash": session["input_hash"],
            "freeze": json.loads(session["freeze_json"]) if session.get("freeze_json") else None,
        }

    # ---------------- 子午线复测研究 ----------------
    #
    # 研究把同一面镜 4~16 份**冻结分析版本**组成独立档案，逐来源记录镜面
    # 旋转角、刀口扫描直径方位与采样时间；来源引用（test_id+version_no）一旦
    # 追加不改，排除只置标记（注明原因）不删行。每次求解写一条不可变研究
    # 版本（输入哈希、参与来源、排除快照、二次谐波结果、留一报告）。定稿冻结
    # 来源版本、角度约定、拟合参数与输入哈希，此后追加/排除/重解均被拒绝。

    def _resolve_source_ref(s):
        """来源引用 → (version, test, sampled_at ISO|None)；版本缺省取最新。"""
        if s.version_no is not None:
            v = db.get_version(s.test_id, s.version_no)
        else:
            v = db.get_latest_version(s.test_id)
        test = db.get_test(s.test_id)
        sampled_at = iso_dt(s.sampled_at) if s.sampled_at is not None else None
        return v, test, sampled_at

    def _check_source_consistency(resolved):
        """resolved: [(version, test), ...]；校验口径/曲率半径/遮罩分区一致。"""
        errs: list[str] = []
        base_v, base_test = resolved[0]
        base_zones = sorted(base_v["result"]["zones"], key=lambda z: z["index"])
        for k, (v, test) in enumerate(resolved[1:], start=1):
            label = f"来源 {k}（批次 {test['id']} v{v['version_no']}）"
            if abs(test["diameter"] - base_test["diameter"]) > 1e-6:
                errs.append(
                    f"{label}：口径 {test['diameter']} mm 与首份来源 "
                    f"{base_test['diameter']} mm 不一致"
                )
            if abs(test["radius_of_curvature"]
                   - base_test["radius_of_curvature"]) > 1e-6:
                errs.append(
                    f"{label}：曲率半径 {test['radius_of_curvature']} mm 与首份来源 "
                    f"{base_test['radius_of_curvature']} mm 不一致"
                )
            zones = sorted(v["result"]["zones"], key=lambda z: z["index"])
            if len(zones) != len(base_zones):
                errs.append(
                    f"{label}：遮罩分区数 {len(zones)} 与首份来源 "
                    f"{len(base_zones)} 不一致"
                )
                continue
            for j, (z, bz) in enumerate(zip(zones, base_zones)):
                if (
                    abs(z["inner_radius"] - bz["inner_radius"]) > 1e-9
                    or abs(z["outer_radius"] - bz["outer_radius"]) > 1e-9
                    or abs(z["r_mean"] - bz["r_mean"]) > 1e-9
                ):
                    errs.append(
                        f"{label}：分区 {j} 边界（{z['inner_radius']}~"
                        f"{z['outer_radius']}）与首份来源（{bz['inner_radius']}~"
                        f"{bz['outer_radius']}）不一致"
                    )
        return errs

    def _study_detail(study: dict) -> dict:
        return {
            "id": study["id"],
            "name": study["name"],
            "notes": study["notes"],
            "status": study["status"],
            "created_at": study["created_at"],
            "finalized_at": study.get("finalized_at"),
            "signal_field": study["signal_field"],
            "confidence_level": study["confidence_level"],
            "axis_stability_deg": study["axis_stability_deg"],
            "angle_convention": study["angle_convention"],
            "input_hash": study.get("input_hash"),
            "sources": [
                {
                    "id": s["id"],
                    "source_index": s["source_index"],
                    "test_id": s["test_id"],
                    "version_no": s["version_no"],
                    "mirror_rotation_deg": s["mirror_rotation_deg"],
                    "knife_diameter_azimuth_deg": s["knife_diameter_azimuth_deg"],
                    "sampled_at": s["sampled_at"],
                    "excluded": s["excluded"],
                    "exclude_reason": s["exclude_reason"],
                }
                for s in study["sources"]
            ],
            "versions": study["versions"],
        }

    def _gather_study_matrix(study: dict):
        """有效来源 → (matrix, errors)；errors 非空时 matrix 为 None。"""
        active = [s for s in study["sources"] if not s["excluded"]]
        if len(active) < 4:
            return None, [
                f"有效来源仅 {len(active)} 份（排除后仍需 ≥4 份才能分离两类二次"
                "像散），请追加来源后再求解"
            ]
        resolved = []
        for s in active:
            v = db.get_version(s["test_id"], s["version_no"])
            test = db.get_test(s["test_id"])
            resolved.append((v, test, s))
        errs = _check_source_consistency([(v, t) for v, t, _ in resolved])
        if errs:
            return None, errs
        base_v, base_test, _ = resolved[0]
        base_zones = sorted(base_v["result"]["zones"], key=lambda z: z["index"])
        radii = [z["r_mean"] for z in base_zones]
        inner = [z["inner_radius"] for z in base_zones]
        outer = [z["outer_radius"] for z in base_zones]
        field = study["signal_field"]
        psi, alpha, signals, source_signals = [], [], [], []
        for v, _test, s in resolved:
            sig = extract_zone_signals(v["result"], field)
            psi.append(s["mirror_rotation_deg"])
            alpha.append(s["knife_diameter_azimuth_deg"])
            signals.append(sig)
            source_signals.append(
                {
                    "test_id": s["test_id"],
                    "version_no": s["version_no"],
                    "source_input_hash": v["input_hash"],
                    "mirror_rotation_deg": s["mirror_rotation_deg"],
                    "knife_diameter_azimuth_deg": s["knife_diameter_azimuth_deg"],
                    "signals_mm": sig,
                }
            )
        matrix = {
            "psi": psi,
            "alpha": alpha,
            "signals": signals,
            "radii": radii,
            "inner": inner,
            "outer": outer,
            "R": base_test["radius_of_curvature"],
            "rim": base_test["diameter"] / 2.0,
            "wavelength_nm": base_test["wavelength_nm"],
            "diameter": base_test["diameter"],
            "source_signals": source_signals,
            "active": active,
        }
        return matrix, []

    def _study_input_payload(study: dict, matrix: dict) -> dict:
        """参与求解的有效输入（哈希只覆盖真正参与拟合的数据）。"""
        return {
            "signal_field": study["signal_field"],
            "confidence_level": study["confidence_level"],
            "axis_stability_deg": study["axis_stability_deg"],
            "angle_convention": study["angle_convention"],
            "constants": {
                "diameter": matrix["diameter"],
                "radius_of_curvature": matrix["R"],
                "wavelength_nm": matrix["wavelength_nm"],
            },
            "zones": [
                {"inner_radius": i, "outer_radius": o, "r_mean": r}
                for i, o, r in zip(matrix["inner"], matrix["outer"], matrix["radii"])
            ],
            "sources": [
                {
                    "test_id": s["test_id"],
                    "version_no": s["version_no"],
                    "source_input_hash": s["source_input_hash"],
                    "mirror_rotation_deg": s["mirror_rotation_deg"],
                    "knife_diameter_azimuth_deg": s["knife_diameter_azimuth_deg"],
                    "signals_mm": [round(x, 12) for x in s["signals_mm"]],
                }
                for s in matrix["source_signals"]
            ],
        }

    def _require_open_study(study: dict) -> None:
        if study["status"] == "finalized":
            raise _err(
                409,
                f"研究 {study['id']} 已定稿冻结（输入哈希 {study['input_hash']}），"
                "来源版本、角度约定、拟合参数与既有结果均不可改写；"
                "如需追加新批次请先复制研究（/copy）",
            )

    def _study_version_brief(v: dict) -> dict:
        r = v["result"]
        return {
            "study_id": v["study_id"],
            "version_no": v["version_no"],
            "created_at": v["created_at"],
            "input_hash": v["input_hash"],
            "reused": v.get("reused", False),
            "summary": {
                "n_sources": r["n_sources"],
                "n_zones": r["n_zones"],
                "rank": r["rank"],
                "residual_dof": r["residual_dof"],
                "residual_sigma_mm": r["residual_sigma_mm"],
                "mirror_axis_deg": r["astigmatism"]["mirror"][
                    "principal_axis_mirror_deg"
                ],
                "mirror_rim_amplitude_nm": r["astigmatism"]["mirror"][
                    "rim_wavefront_amplitude_nm"
                ],
                "lab_axis_deg": r["astigmatism"]["lab_fixed"][
                    "principal_axis_lab_deg"
                ],
                "lab_rim_amplitude_nm": r["astigmatism"]["lab_fixed"][
                    "rim_wavefront_amplitude_nm"
                ],
                "total_wavefront_rms_waves": r["astigmatism"][
                    "total_wavefront_rms_waves"
                ],
                "most_suspicious_zone": r["most_suspicious_zone"]["index"],
                "loo_stable": (r.get("loo") or {}).get("stable"),
            },
        }

    def _run_study_solve(study_id: int, include_loo: bool = True) -> dict:
        """用当前有效来源求解；输入与最新版本一致时幂等复用。"""
        study = db.get_study(study_id)
        matrix, errs = _gather_study_matrix(study)
        if errs:
            raise _err(422, "子午线研究无法求解", errs)
        input_hash = compute_input_hash(_study_input_payload(study, matrix))
        try:
            latest = db.get_latest_study_version(study_id)
        except NotFoundError:
            latest = None
        if latest is not None and latest["input_hash"] == input_hash:
            latest["reused"] = True
            return latest
        result = fit_harmonics(
            matrix["psi"],
            matrix["alpha"],
            matrix["signals"],
            matrix["radii"],
            matrix["inner"],
            matrix["outer"],
            matrix["R"],
            matrix["rim"],
            matrix["wavelength_nm"],
            confidence_level=study["confidence_level"],
        )
        loo = None
        if include_loo:
            loo = leave_one_out(
                matrix["psi"],
                matrix["alpha"],
                matrix["signals"],
                matrix["radii"],
                matrix["inner"],
                matrix["outer"],
                matrix["R"],
                matrix["rim"],
                matrix["wavelength_nm"],
                confidence_level=study["confidence_level"],
                axis_stability_deg=study["axis_stability_deg"],
            )
            result["loo"] = loo
        active_ids = [s["id"] for s in matrix["active"]]
        excluded_snapshot = [
            {
                "source_index": s["source_index"],
                "test_id": s["test_id"],
                "version_no": s["version_no"],
                "reason": s["exclude_reason"],
            }
            for s in study["sources"]
            if s["excluded"]
        ]
        version = db.create_study_version(
            study_id, input_hash, active_ids, excluded_snapshot, result, loo
        )
        version["reused"] = False
        return version

    def _append_sources(study_id: int, sources) -> list[tuple]:
        """校验并追加来源；返回解析出的 (version,test,s,sampled_at)。"""
        study = db.get_study(study_id)
        _require_open_study(study)
        seen = set()
        resolved = []
        for k, s in enumerate(sources):
            v, test, sampled_at = _resolve_source_ref(s)
            key = (test["id"], v["version_no"])
            if key in seen:
                raise _err(
                    422,
                    f"来源 {k}：批次 {test['id']} v{v['version_no']} 在请求中重复",
                )
            seen.add(key)
            resolved.append((v, test, s, sampled_at))
        existing = {(src["test_id"], src["version_no"]) for src in study["sources"]}
        dups = [
            f"批次 {t['id']} v{v['version_no']} 已是研究来源"
            for v, t, _s, _sa in resolved
            if (t["id"], v["version_no"]) in existing
        ]
        if dups:
            raise _err(409, "不能重复追加同一冻结分析版本", dups)
        projected = len(study["sources"]) + len(resolved)
        if projected > 16:
            raise _err(
                422,
                f"追加后来源总数 {projected} 超过研究档案上限 16"
                "（每份来源为一个冻结分析版本）",
            )
        # 口径/曲率半径/遮罩分区一致性：新来源须与既有来源一致（先校验再落库）
        all_refs = [(db.get_version(s["test_id"], s["version_no"]),
                     db.get_test(s["test_id"])) for s in study["sources"]]
        all_refs += [(v, t) for v, t, _s, _sa in resolved]
        consistency = _check_source_consistency(all_refs)
        if consistency:
            raise _err(
                422, "子午线研究来源不一致（口径/曲率半径/遮罩分区必须相同）",
                consistency,
            )
        for v, _test, s, sampled_at in resolved:
            db.add_study_source(
                study_id,
                {
                    "test_id": v["test_id"],
                    "version_no": v["version_no"],
                    "mirror_rotation_deg": s.mirror_rotation_deg,
                    "knife_diameter_azimuth_deg": s.knife_diameter_azimuth_deg,
                    "sampled_at": sampled_at,
                },
            )
        return resolved

    @app.post("/api/meridian-studies", status_code=201)
    def create_study(payload: StudyCreateIn):
        # 解析 + 一致性/重复校验先于落库（失败不留档案）
        seen = set()
        resolved = []
        for k, s in enumerate(payload.sources):
            v, test, sampled_at = _resolve_source_ref(s)
            key = (test["id"], v["version_no"])
            if key in seen:
                raise _err(
                    422,
                    f"来源 {k}：批次 {test['id']} v{v['version_no']} 在请求中重复",
                )
            seen.add(key)
            resolved.append((v, test, s, sampled_at))
        errs = _check_source_consistency([(v, t) for v, t, _s, _sa in resolved])
        if errs:
            raise _err(
                422, "子午线研究来源不一致（口径/曲率半径/遮罩分区必须相同）", errs
            )
        study_id = db.create_study(
            {
                "name": payload.name,
                "notes": payload.notes,
                "signal_field": payload.signal_field,
                "confidence_level": payload.confidence_level,
                "axis_stability_deg": payload.axis_stability_deg,
                "angle_convention": ANGLE_CONVENTION,
            }
        )
        for v, _test, s, sampled_at in resolved:
            db.add_study_source(
                study_id,
                {
                    "test_id": v["test_id"],
                    "version_no": v["version_no"],
                    "mirror_rotation_deg": s.mirror_rotation_deg,
                    "knife_diameter_azimuth_deg": s.knife_diameter_azimuth_deg,
                    "sampled_at": sampled_at,
                },
            )
        study = db.get_study(study_id)
        # 首次求解欠秩：档案保留（待补角度），以 422 返回角度缺口
        try:
            version = _run_study_solve(study_id, include_loo=payload.loo)
        except MeridianError as exc:
            return JSONResponse(
                status_code=422,
                content={
                    "study": _study_detail(study),
                    "solve_error": {"message": str(exc), "errors": exc.gaps},
                },
            )
        return {"study": _study_detail(study), "version": _study_version_brief(version)}

    @app.get("/api/meridian-studies")
    def list_studies():
        return {"studies": db.list_studies()}

    @app.get("/api/meridian-studies/{study_id}")
    def get_study(study_id: int):
        return {"study": _study_detail(db.get_study(study_id))}

    @app.post("/api/meridian-studies/{study_id}/sources", status_code=201)
    def append_study_sources(study_id: int, payload: StudyAppendIn):
        _append_sources(study_id, payload.sources)
        study = db.get_study(study_id)
        out = {"study": _study_detail(study)}
        if payload.solve:
            out["version"] = _study_version_brief(_run_study_solve(study_id))
        return out

    @app.post("/api/meridian-studies/{study_id}/sources/exclude")
    def exclude_study_source(study_id: int, payload: StudyExcludeIn):
        study = db.get_study(study_id)
        _require_open_study(study)
        src = db.get_study_source(study_id, payload.source_index)
        if src["excluded"]:
            raise _err(
                409,
                f"来源 #{payload.source_index} 已被排除"
                f"（原因：{src['exclude_reason']}）",
            )
        active_now = sum(1 for s in study["sources"] if not s["excluded"]) - 1
        if active_now < 4:
            raise _err(
                409,
                f"排除后有效来源仅 {active_now} 份，少于分离两类二次像散所需的 4 份",
            )
        db.set_study_source_excluded(study_id, payload.source_index, payload.reason)
        study = db.get_study(study_id)
        out = {
            "study": _study_detail(study),
            "excluded": {"source_index": payload.source_index, "reason": payload.reason},
        }
        if payload.solve:
            out["version"] = _study_version_brief(
                _run_study_solve(study_id, include_loo=payload.loo)
            )
        return out

    @app.post("/api/meridian-studies/{study_id}/solve", status_code=201)
    def solve_study(study_id: int, payload: StudySolveIn):
        study = db.get_study(study_id)
        _require_open_study(study)
        return {
            "version": _study_version_brief(
                _run_study_solve(study_id, include_loo=payload.loo)
            )
        }

    @app.get("/api/meridian-studies/{study_id}/loo")
    def study_loo(study_id: int, version_no: int | None = None):
        db.get_study(study_id)
        v = (
            db.get_study_version(study_id, version_no)
            if version_no is not None
            else db.get_latest_study_version(study_id)
        )
        # 严格使用该版本冻结的来源与信号复算，不读取研究当前来源/排除状态
        if v["loo"] is not None:
            loo = v["loo"]
            on_demand = False
        else:
            loo = loo_from_fit_result(
                v["result"], axis_stability_deg=db.get_study(study_id)["axis_stability_deg"]
            )
            on_demand = True
        return {
            "study_id": study_id,
            "version_no": v["version_no"],
            "computed_on_demand": on_demand,
            "loo": loo,
        }

    @app.get("/api/meridian-studies/{study_id}/versions/{version_no}")
    def get_study_version(study_id: int, version_no: int):
        return db.get_study_version(study_id, version_no)

    @app.post("/api/meridian-studies/{study_id}/copy", status_code=201)
    def copy_study(study_id: int, payload: StudyCopyIn):
        src_study = db.get_study(study_id)
        base_name = src_study["name"] or f"研究 {study_id}"
        new_id = db.create_study(
            {
                "name": payload.name or f"{base_name}（副本）",
                "notes": payload.notes,
                "signal_field": src_study["signal_field"],
                "confidence_level": src_study["confidence_level"],
                "axis_stability_deg": src_study["axis_stability_deg"],
                "angle_convention": src_study["angle_convention"],
            }
        )
        db.copy_study_sources(new_id, src_study["sources"])
        new_study = db.get_study(new_id)
        version = _run_study_solve(new_id)
        return {
            "study": _study_detail(new_study),
            "copied_from_study_id": study_id,
            "version": _study_version_brief(version),
        }

    @app.post("/api/meridian-studies/{study_id}/finalize")
    def finalize_study(study_id: int, payload: StudyFinalizeIn):
        study = db.get_study(study_id)
        if study["status"] == "finalized":
            return {
                "study": _study_detail(study),
                "reused": True,
                "input_hash": study["input_hash"],
            }
        _require_open_study(study)
        v = (
            db.get_study_version(study_id, payload.version_no)
            if payload.version_no is not None
            else db.get_latest_study_version(study_id)
        )
        # 严格按所选版本冻结：参与求解的来源取该版本 active_source_ids，
        # 排除状态取该版本写入时的排除快照，不读取研究当前来源状态。
        active_ids = set(v["active_source_ids"])
        by_id = {s["id"]: s for s in study["sources"]}
        excluded_by_index = {
            e["source_index"]: e for e in v["excluded_snapshot"]
        }
        sources = []
        for s in sorted(study["sources"], key=lambda x: x["source_index"]):
            sv = db.get_version(s["test_id"], s["version_no"])
            was_active = s["id"] in active_ids
            snap = excluded_by_index.get(s["source_index"])
            sources.append(
                {
                    "source_index": s["source_index"],
                    "test_id": s["test_id"],
                    "version_no": s["version_no"],
                    "version_input_hash": sv["input_hash"],
                    "mirror_rotation_deg": s["mirror_rotation_deg"],
                    "knife_diameter_azimuth_deg": s["knife_diameter_azimuth_deg"],
                    "sampled_at": s["sampled_at"],
                    "active_in_frozen_version": was_active,
                    "excluded": not was_active,
                    "exclude_reason": snap["reason"] if snap else None,
                }
            )
        n_active = sum(1 for x in sources if x["active_in_frozen_version"])
        freeze = {
            "study_id": study_id,
            "frozen_version_no": v["version_no"],
            "n_active_sources": n_active,
            "sources": sources,
            "angle_convention": study["angle_convention"],
            "fit_params": {
                "signal_field": study["signal_field"],
                "confidence_level": study["confidence_level"],
                "axis_stability_deg": study["axis_stability_deg"],
            },
            "solve_input_hash": v["input_hash"],
        }
        freeze_str = json.dumps(freeze, sort_keys=True, ensure_ascii=True)
        input_hash = compute_input_hash(freeze)
        outcome = db.finalize_study(study_id, input_hash, freeze_str, v["version_no"])
        if outcome["status"] != "finalized":
            settled = db.get_study(study_id)
            return {
                "study": _study_detail(settled),
                "reused": True,
                "input_hash": outcome.get("input_hash"),
            }
        return {
            "study": _study_detail(db.get_study(study_id)),
            "reused": False,
            "version_no": v["version_no"],
            "input_hash": input_hash,
        }

    @app.get("/api/meridian-studies/{study_id}/freeze")
    def get_study_freeze(study_id: int):
        study = db.get_study(study_id)
        if study["status"] != "finalized":
            raise _err(409, f"研究 {study_id} 尚未定稿，无冻结包")
        return {
            "input_hash": study["input_hash"],
            "freeze": json.loads(study["freeze_json"])
            if study.get("freeze_json")
            else None,
        }

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
