"""FastAPI 路由：测试批次管理、读数冻结/剔除、分析版本、批次对比、修正搜索。"""
from __future__ import annotations

import math
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import __version__
from .correction import CorrectionError, search_corrections
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
    CompareIn,
    CorrectionSearchIn,
    ExcludeIn,
    FreezeIn,
    RestoreIn,
    TestCreateIn,
)
from .storage import ConflictError, Database, NotFoundError

DEFAULT_DB = os.environ.get("FOUCAULT_DB", "foucault.db")


def _err(status: int, message: str, details: list | None = None) -> HTTPException:
    return HTTPException(
        status_code=status,
        detail={"message": message, "errors": details or []},
    )


def _validate_payload(p: TestCreateIn) -> list[str]:
    """业务校验：返回错误列表（空 = 通过）。类型/字面量错误由 Pydantic 拦截。"""
    errs: list[str] = []
    for name, val in (
        ("diameter", p.diameter),
        ("radius_of_curvature", p.radius_of_curvature),
        ("conic_constant", p.conic_constant),
        ("wavelength_nm", p.wavelength_nm),
        ("instrument_offset", p.instrument_offset),
    ):
        if not math.isfinite(val):
            errs.append(f"{name} 必须是有限数值")
    if p.diameter <= 0:
        errs.append("镜面口径必须为正数")
    if p.radius_of_curvature <= 0:
        errs.append("曲率半径必须为正数")
    if p.wavelength_nm <= 0:
        errs.append("检测波长必须为正数")
    if abs(p.conic_constant) > 10:
        errs.append("目标圆锥常数超出合理范围 |K| <= 10")
    if errs:
        return errs

    rim = p.diameter / 2.0
    min_n = p.options.min_readings_per_zone
    zones = sorted(p.zones, key=lambda z: (z.inner_radius, z.outer_radius))
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
    """版本快照：常量 + 算法选项 + 有效读数（冻结原始输入）。"""
    zones = []
    for z in sorted(test["zones"], key=lambda x: x["zone_index"]):
        readings = [
            {
                "id": r["id"],
                "value": r["value"],
                "value_original": r["value_original"],
            }
            for r in z["readings"]
            if not r["excluded"]
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
    return {
        "constants": snapshot["constants"],
        "options": snapshot["options"],
        "zones": [
            {
                "inner_radius": z["inner_radius"],
                "outer_radius": z["outer_radius"],
                "readings": [r["value"] for r in z["readings"]],
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
        "读数冻结与剔除、批次对比和分区修正量搜索。",
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

    # ---------------- 测试批次 ----------------

    @app.post("/api/tests", status_code=201)
    def create_test(payload: TestCreateIn):
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

    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
