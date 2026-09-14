"""SQLite 持久化：测试批次、分区、读数、分析版本、遮罩方案版本。

分析版本（versions）冻结创建时刻的有效读数快照、常量、算法选项与输入哈希，
结果 JSON 一旦写入不再修改，保证重复读取结果不变。
遮罩方案（mask_schemes + mask_scheme_versions）同样不可变：每个版本冻结
常量、制作限制与计算出的环带/开窗布局；测试批次引用遮罩版本时复制其环带
边界，此后遮罩另建版本不影响既有批次与分析。

往返测量会话（measurement_sessions + session_readings）：会话冻结对遮罩
版本的引用与采集计划（测次序列），逐笔读数（含补测 attempt、剔除原因、
锁定/冻结标记）只增不改；定稿时写入冻结包哈希并关联自动生成的测试批次，
此后原始记录全部冻结，重复定稿幂等返回同一批次。
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
CREATE TABLE IF NOT EXISTS mask_schemes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    notes TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS mask_scheme_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scheme_id INTEGER NOT NULL REFERENCES mask_schemes(id) ON DELETE CASCADE,
    version_no INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    params_json TEXT NOT NULL,   -- 冻结的常量与制作限制（mm）
    layout_json TEXT NOT NULL,   -- 环带边界 / 开窗 / 刀口位移预测 / 指标
    params_hash TEXT NOT NULL,
    UNIQUE (scheme_id, version_no)
);
CREATE TABLE IF NOT EXISTS measurement_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    notes TEXT,
    status TEXT NOT NULL DEFAULT 'collecting',  -- collecting | confirmed | finalized
    created_at TEXT NOT NULL,
    deadline TEXT,                               -- 采集时限（UTC ISO）
    unit TEXT NOT NULL,
    repeats_per_zone INTEGER NOT NULL,
    start_direction TEXT NOT NULL,
    reference_zone_index INTEGER NOT NULL,
    reference_refresh INTEGER NOT NULL,
    mask_scheme_id INTEGER NOT NULL REFERENCES mask_schemes(id),
    mask_version_no INTEGER NOT NULL,
    wavelength_nm REAL NOT NULL,
    instrument_offset REAL NOT NULL DEFAULT 0,  -- mm
    thresholds_json TEXT NOT NULL,             -- 定稿质量阈值（mm）
    options_json TEXT NOT NULL,                -- 定稿批次的分析选项
    plan_json TEXT NOT NULL,                   -- 不可变测站计划（测次序列）
    freeze_json TEXT,                          -- 定稿冻结包（原始记录 + 校正参数）
    input_hash TEXT,                           -- 定稿冻结包哈希
    finalized_at TEXT,
    test_id INTEGER REFERENCES tests(id)
);
CREATE TABLE IF NOT EXISTS session_readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES measurement_sessions(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    attempt INTEGER NOT NULL,                  -- 同一测次补测逐次递增
    zone_index INTEGER NOT NULL,
    direction TEXT NOT NULL,
    kind TEXT NOT NULL,                        -- reading | reference
    knife_position REAL NOT NULL,              -- 规范值（mm）
    knife_position_original REAL NOT NULL,     -- 声明单位原始值
    collected_at TEXT NOT NULL,
    excluded INTEGER NOT NULL DEFAULT 0,
    exclude_reason TEXT,
    locked INTEGER NOT NULL DEFAULT 0,
    frozen INTEGER NOT NULL DEFAULT 0,         -- 定稿后冻结
    created_at TEXT NOT NULL,
    UNIQUE (session_id, seq, attempt)
);
CREATE INDEX IF NOT EXISTS idx_session_readings_session
    ON session_readings(session_id);
"""

# tests 表后加列（老库迁移）：批次引用的遮罩方案与版本
_TEST_EXTRA_COLUMNS = {
    "mask_scheme_id": "ALTER TABLE tests ADD COLUMN mask_scheme_id INTEGER",
    "mask_version_no": "ALTER TABLE tests ADD COLUMN mask_version_no INTEGER",
    "session_id": "ALTER TABLE tests ADD COLUMN session_id INTEGER",
}

# measurement_sessions 表后加列（老库迁移）
_SESSION_EXTRA_COLUMNS = {
    "freeze_json": "ALTER TABLE measurement_sessions ADD COLUMN freeze_json TEXT",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NotFoundError(Exception):
    pass


class ConflictError(Exception):
    pass


class Database:
    """单连接 + 可重入锁的轻量封装（本机服务，并发量低）。

    使用 RLock：create_version 等方法在持锁期间会调用其他加锁方法
    （如 get_version），不可重入锁会导致请求线程自我死锁。
    """

    def __init__(self, path: str = "foucault.db"):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            cols = {
                r[1] for r in self._conn.execute("PRAGMA table_info(tests)").fetchall()
            }
            for col, ddl in _TEST_EXTRA_COLUMNS.items():
                if col not in cols:
                    self._conn.execute(ddl)
            scols = {
                r[1]
                for r in self._conn.execute(
                    "PRAGMA table_info(measurement_sessions)"
                ).fetchall()
            }
            for col, ddl in _SESSION_EXTRA_COLUMNS.items():
                if scols and col not in scols:
                    self._conn.execute(ddl)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------- 创建 ----------------

    def create_test(self, record: dict, zones: list[dict]) -> int:
        """record 为常量（mm）+ options_json；zones 含 readings（mm 与原始值）。

        record 可带 mask_scheme_id / mask_version_no：批次冻结所引用遮罩
        版本的 ID，环带边界已在 zones 中复制，此后遮罩变更不影响本批次。
        """
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO tests
                   (name, notes, created_at, unit, source_mode, diameter,
                    radius_of_curvature, conic_constant, wavelength_nm,
                    instrument_offset, options_json, mask_scheme_id,
                    mask_version_no, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                    record.get("mask_scheme_id"),
                    record.get("mask_version_no"),
                    record.get("session_id"),
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

    def delete_test(self, test_id: int) -> None:
        """删除批次及其分区/读数/版本（定稿失败回调用；外键级联）。"""
        with self._lock:
            self._conn.execute("DELETE FROM tests WHERE id = ?", (test_id,))
            self._conn.commit()

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

    # ---------------- 遮罩方案 ----------------

    def create_mask_scheme(self, name: str | None, notes: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO mask_schemes (name, notes, created_at) VALUES (?, ?, ?)",
                (name, notes, _now()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_mask_scheme(self, scheme_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM mask_schemes WHERE id = ?", (scheme_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"遮罩方案 {scheme_id} 不存在")
            scheme = dict(row)
            scheme["versions"] = self.list_mask_versions(scheme_id)
            return scheme

    def list_mask_schemes(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.id, s.name, s.notes, s.created_at, "
                " COUNT(v.id) AS n_versions, MAX(v.version_no) AS latest_version_no"
                " FROM mask_schemes s"
                " LEFT JOIN mask_scheme_versions v ON v.scheme_id = s.id"
                " GROUP BY s.id ORDER BY s.id"
            ).fetchall()
            return [dict(r) for r in rows]

    def create_mask_version(
        self, scheme_id: int, params: dict, layout: dict, params_hash: str
    ) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(version_no), 0) AS v"
                " FROM mask_scheme_versions WHERE scheme_id = ?",
                (scheme_id,),
            ).fetchone()
            version_no = int(row["v"]) + 1
            self._conn.execute(
                "INSERT INTO mask_scheme_versions"
                " (scheme_id, version_no, created_at, params_json, layout_json,"
                "  params_hash) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    scheme_id,
                    version_no,
                    _now(),
                    json.dumps(params, ensure_ascii=True),
                    json.dumps(layout, ensure_ascii=True),
                    params_hash,
                ),
            )
            self._conn.commit()
            return self.get_mask_version(scheme_id, version_no)

    def get_mask_version(self, scheme_id: int, version_no: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM mask_scheme_versions"
                " WHERE scheme_id = ? AND version_no = ?",
                (scheme_id, version_no),
            ).fetchone()
            if row is None:
                raise NotFoundError(
                    f"遮罩方案 {scheme_id} 的版本 {version_no} 不存在"
                )
            return self._mask_version_dict(row)

    def get_latest_mask_version(self, scheme_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM mask_scheme_versions WHERE scheme_id = ?"
                " ORDER BY version_no DESC LIMIT 1",
                (scheme_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"遮罩方案 {scheme_id} 尚无版本")
            return self._mask_version_dict(row)

    def list_mask_versions(self, scheme_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, scheme_id, version_no, created_at, params_hash,"
                " layout_json FROM mask_scheme_versions"
                " WHERE scheme_id = ? ORDER BY version_no",
                (scheme_id,),
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                layout = json.loads(d.pop("layout_json"))
                d["metrics"] = layout.get("metrics", {})
                out.append(d)
            return out

    @staticmethod
    def _mask_version_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["params"] = json.loads(d.pop("params_json"))
        d["layout"] = json.loads(d.pop("layout_json"))
        return d

    # ---------------- 往返测量会话 ----------------

    def create_session(self, record: dict, plan: list[dict]) -> int:
        """创建采集会话；record 含会话参数（mm 规范值）、阈值、分析选项。"""
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO measurement_sessions
                   (name, notes, status, created_at, deadline, unit,
                    repeats_per_zone, start_direction, reference_zone_index,
                    reference_refresh, mask_scheme_id, mask_version_no,
                    wavelength_nm, instrument_offset, thresholds_json,
                    options_json, plan_json)
                   VALUES (?, ?, 'collecting', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.get("name"),
                    record.get("notes"),
                    _now(),
                    record.get("deadline"),
                    record["unit"],
                    record["repeats_per_zone"],
                    record["start_direction"],
                    record["reference_zone_index"],
                    record["reference_refresh"],
                    record["mask_scheme_id"],
                    record["mask_version_no"],
                    record["wavelength_nm"],
                    record["instrument_offset"],
                    json.dumps(record["thresholds"], ensure_ascii=True),
                    json.dumps(record["options"], ensure_ascii=True),
                    json.dumps(plan, ensure_ascii=True),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def _session_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        d["thresholds"] = json.loads(d.pop("thresholds_json"))
        d["options"] = json.loads(d.pop("options_json"))
        d["plan"] = json.loads(d.pop("plan_json"))
        return d

    def get_session(self, session_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM measurement_sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"测量会话 {session_id} 不存在")
            s = self._session_dict(row)
            rows = self._conn.execute(
                "SELECT * FROM session_readings WHERE session_id = ? ORDER BY seq, attempt",
                (session_id,),
            ).fetchall()
            s["readings"] = [dict(r) for r in rows]
            return s

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, status, created_at, deadline, unit,"
                " repeats_per_zone, start_direction, reference_zone_index,"
                " reference_refresh, mask_scheme_id, mask_version_no,"
                " finalized_at, test_id FROM measurement_sessions ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_slot_readings(self, session_id: int, seq: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM session_readings WHERE session_id = ? AND seq = ?"
                " ORDER BY attempt",
                (session_id, seq),
            ).fetchall()
            return [dict(r) for r in rows]

    def add_session_reading(self, session_id: int, slot: dict, reading: dict) -> dict:
        """按计划测位追加一次读数（attempt 自动递增）。

        slot 为计划测位；reading 含 knife_position（mm）、原始值、collected_at。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(attempt), 0) AS a FROM session_readings"
                " WHERE session_id = ? AND seq = ?",
                (session_id, slot["seq"]),
            ).fetchone()
            attempt = int(row["a"]) + 1
            self._conn.execute(
                """INSERT INTO session_readings
                   (session_id, seq, attempt, zone_index, direction, kind,
                    knife_position, knife_position_original, collected_at,
                    created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    slot["seq"],
                    attempt,
                    slot["zone_index"],
                    slot["direction"],
                    slot["kind"],
                    reading["knife_position_mm"],
                    reading["knife_position_original"],
                    reading["collected_at"],
                    _now(),
                ),
            )
            self._conn.commit()
            return self._current_session_reading(session_id, slot["seq"])

    def _current_session_reading(self, session_id: int, seq: int) -> dict:
        """某测位当前有效读数：最大 attempt 且未剔除；无则返回 None。"""
        row = self._conn.execute(
            "SELECT * FROM session_readings WHERE session_id = ? AND seq = ?"
            " AND excluded = 0 ORDER BY attempt DESC LIMIT 1",
            (session_id, seq),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_session_reading(self, reading_id: int) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM session_readings WHERE id = ?", (reading_id,)
            ).fetchone()
            if row is None:
                raise NotFoundError(f"会话读数 {reading_id} 不存在")
            return dict(row)

    def set_session_reading_excluded(
        self, reading_id: int, excluded: bool, reason: str | None
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE session_readings SET excluded = ?, exclude_reason = ?"
                " WHERE id = ?",
                (1 if excluded else 0, reason if excluded else None, reading_id),
            )
            self._conn.commit()

    def lock_session_slots(self, session_id: int, seqs: list[int] | None) -> int:
        """锁定测位的当前有效读数；seqs=None 时锁定全部已采集测位。"""
        with self._lock:
            if seqs is None:
                cur = self._conn.execute(
                    "UPDATE session_readings SET locked = 1 WHERE session_id = ?"
                    " AND excluded = 0",
                    (session_id,),
                )
            else:
                cur = self._conn.executemany(
                    "UPDATE session_readings SET locked = 1"
                    " WHERE session_id = ? AND seq = ? AND excluded = 0",
                    [(session_id, seq) for seq in seqs],
                )
            self._conn.commit()
            return cur.rowcount

    def set_session_status(self, session_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE measurement_sessions SET status = ? WHERE id = ?",
                (status, session_id),
            )
            self._conn.commit()

    def finalize_session(
        self,
        session_id: int,
        input_hash: str,
        freeze_json: str,
        test_id: int,
    ) -> dict:
        """原子定稿：仅当会话仍处于非定稿状态时关联批次并冻结。

        返回 {"status": ..., "test_id": ...}：
        - "finalized"：本次完成状态翻转并冻结全部原始读数；
        - "reused"：已被其他请求定稿（并发/重复），返回既有 test_id；
        - "stale"：会话状态已被改动（不再是调用方预期的 confirmed）。

        分析失败时由调用方先删除刚创建的批次，本方法不被调用，
        因而不会留下无会话关联的孤儿批次。
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT status, test_id FROM measurement_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"测量会话 {session_id} 不存在")
            if row["status"] == "finalized":
                return {"status": "reused", "test_id": row["test_id"]}
            if row["status"] != "confirmed":
                return {"status": "stale", "test_id": None}
            self._conn.execute(
                "UPDATE measurement_sessions SET status = 'finalized',"
                " input_hash = ?, freeze_json = ?, finalized_at = ?, test_id = ?"
                " WHERE id = ? AND status = 'confirmed'",
                (input_hash, freeze_json, _now(), test_id, session_id),
            )
            self._conn.execute(
                "UPDATE session_readings SET frozen = 1 WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            return {"status": "finalized", "test_id": test_id}
