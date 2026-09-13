"""SQLite 持久化：测试批次、分区、读数、分析版本。

分析版本（versions）冻结创建时刻的有效读数快照、常量、算法选项与输入哈希，
结果 JSON 一旦写入不再修改，保证重复读取结果不变。
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    notes TEXT,
    created_at TEXT NOT NULL,
    unit TEXT NOT NULL,
    source_mode TEXT NOT NULL,
    diameter REAL NOT NULL,
    radius_of_curvature REAL NOT NULL,
    conic_constant REAL NOT NULL,
    wavelength_nm REAL NOT NULL,
    instrument_offset REAL NOT NULL DEFAULT 0,
    options_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS zones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
    zone_index INTEGER NOT NULL,
    inner_radius REAL NOT NULL,
    outer_radius REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
    zone_id INTEGER NOT NULL REFERENCES zones(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    value REAL NOT NULL,            -- mm（分析用规范值）
    value_original REAL NOT NULL,   -- 用户提交的原始值（声明单位）
    excluded INTEGER NOT NULL DEFAULT 0,
    exclude_reason TEXT,
    frozen INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id INTEGER NOT NULL REFERENCES tests(id) ON DELETE CASCADE,
    version_no INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    UNIQUE (test_id, version_no)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NotFoundError(Exception):
    pass


class ConflictError(Exception):
    pass


class Database:
    """单连接 + 锁的轻量封装（本机服务，并发量低）。"""

    def __init__(self, path: str = "foucault.db"):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------- 创建 ----------------

    def create_test(self, record: dict, zones: list[dict]) -> int:
        """record 为常量（mm）+ options_json；zones 含 readings（mm 与原始值）。"""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO tests
                   (name, notes, created_at, unit, source_mode, diameter,
                    radius_of_curvature, conic_constant, wavelength_nm,
                    instrument_offset, options_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.get("name"),
                    record.get("notes"),
                    _now(),
                    record["unit"],
                    record["source_mode"],
                    record["diameter"],
                    record["radius_of_curvature"],
                    record["conic_constant"],
                    record["wavelength_nm"],
                    record["instrument_offset"],
                    json.dumps(record["options"], ensure_ascii=True),
                ),
            )
            test_id = cur.lastrowid
            for zi, zone in enumerate(zones):
                zcur = self._conn.execute(
                    "INSERT INTO zones (test_id, zone_index, inner_radius, outer_radius)"
                    " VALUES (?, ?, ?, ?)",
                    (test_id, zi, zone["inner_radius"], zone["outer_radius"]),
                )
                zone_id = zcur.lastrowid
                for seq, (val_mm, val_orig) in enumerate(zone["readings"]):
                    self._conn.execute(
                        "INSERT INTO readings (test_id, zone_id, seq, value, value_original)"
                        " VALUES (?, ?, ?, ?, ?)",
                        (test_id, zone_id, seq, val_mm, val_orig),
                    )
            self._conn.commit()
            return int(test_id)

    # ---------------- 读取 ----------------

    def get_test(self, test_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM tests WHERE id = ?", (test_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"测试批次 {test_id} 不存在")
            test = dict(row)
            test["options"] = json.loads(test.pop("options_json"))
            zones = self._conn.execute(
                "SELECT * FROM zones WHERE test_id = ? ORDER BY zone_index", (test_id,)
            ).fetchall()
            test["zones"] = []
            for z in zones:
                zd = dict(z)
                readings = self._conn.execute(
                    "SELECT * FROM readings WHERE zone_id = ? ORDER BY seq", (z["id"],)
                ).fetchall()
                zd["readings"] = [dict(r) for r in readings]
                test["zones"].append(zd)
            return test

    def list_tests(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, created_at, unit, source_mode, diameter,"
                " radius_of_curvature, conic_constant, wavelength_nm FROM tests"
                " ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    # ---------------- 读数状态 ----------------

    def get_reading(self, reading_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM readings WHERE id = ?", (reading_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"读数 {reading_id} 不存在")
            return dict(row)

    def set_excluded(self, reading_id: int, excluded: bool, reason: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE readings SET excluded = ?, exclude_reason = ? WHERE id = ?",
                (1 if excluded else 0, reason if excluded else None, reading_id),
            )
            self._conn.commit()

    def set_frozen(self, test_id: int, reading_ids: list[int] | None) -> int:
        """冻结读数；reading_ids 为 None 时冻结该测试全部有效读数。"""
        with self._lock:
            if reading_ids is None:
                cur = self._conn.execute(
                    "UPDATE readings SET frozen = 1 WHERE test_id = ? AND excluded = 0",
                    (test_id,),
                )
            else:
                cur = self._conn.executemany(
                    "UPDATE readings SET frozen = 1 WHERE id = ? AND test_id = ?",
                    [(rid, test_id) for rid in reading_ids],
                )
            self._conn.commit()
            return cur.rowcount

    # ---------------- 版本 ----------------

    def create_version(
        self, test_id: int, input_hash: str, snapshot: dict, result: dict
    ) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version_no), 0) AS v FROM versions WHERE test_id = ?",
                (test_id,),
            ).fetchone()
            version_no = int(row["v"]) + 1
            self._conn.execute(
                "INSERT INTO versions (test_id, version_no, created_at, input_hash,"
                " snapshot_json, result_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    test_id,
                    version_no,
                    _now(),
                    input_hash,
                    json.dumps(snapshot, ensure_ascii=True),
                    json.dumps(result, ensure_ascii=True),
                ),
            )
            self._conn.commit()
            return self.get_version(test_id, version_no)

    def get_version(self, test_id: int, version_no: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM versions WHERE test_id = ? AND version_no = ?",
                (test_id, version_no),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"测试 {test_id} 的版本 {version_no} 不存在")
            return self._version_dict(row)

    def get_latest_version(self, test_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM versions WHERE test_id = ? ORDER BY version_no DESC LIMIT 1",
                (test_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"测试 {test_id} 尚无分析版本")
            return self._version_dict(row)

    def list_versions(self, test_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, test_id, version_no, created_at, input_hash, result_json"
                " FROM versions WHERE test_id = ? ORDER BY version_no",
                (test_id,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                result = json.loads(d.pop("result_json"))
                d["summary"] = result.get("summary", {})
                out.append(d)
            return out

    @staticmethod
    def _version_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["snapshot"] = json.loads(d.pop("snapshot_json"))
        d["result"] = json.loads(d.pop("result_json"))
        return d
