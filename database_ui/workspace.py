"""Durable sidecar storage for manual corpus curation and training labels."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


class AnnotationWorkspace:
    """A rebuild-safe SQLite sidecar; all keys refer back to immutable corpus IDs."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_annotations (
                    pmid TEXT PRIMARY KEY,
                    training_status TEXT NOT NULL DEFAULT 'not_selected',
                    review_status TEXT NOT NULL DEFAULT 'not_started',
                    reviewer TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS section_revisions (
                    pmid TEXT NOT NULL,
                    section_name TEXT NOT NULL,
                    base_content_sha256 TEXT NOT NULL DEFAULT '',
                    original_content TEXT NOT NULL DEFAULT '',
                    revised_content TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'draft',
                    reviewer TEXT NOT NULL DEFAULT '',
                    rationale TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (pmid, section_name)
                );

                CREATE TABLE IF NOT EXISTS task_annotations (
                    occurrence_key TEXT PRIMARY KEY,
                    pmid TEXT NOT NULL,
                    reviewed_task_name TEXT NOT NULL DEFAULT '',
                    reviewed_description TEXT NOT NULL DEFAULT '',
                    reviewed_task_role TEXT NOT NULL DEFAULT '',
                    in_scanner_status TEXT NOT NULL DEFAULT 'unknown',
                    review_status TEXT NOT NULL DEFAULT 'draft',
                    reviewer TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS activation_table_annotations (
                    table_key TEXT PRIMARY KEY,
                    pmid TEXT NOT NULL,
                    section_name TEXT NOT NULL,
                    table_index INTEGER NOT NULL,
                    source_markdown TEXT NOT NULL DEFAULT '',
                    is_activation_table TEXT NOT NULL DEFAULT 'unknown',
                    table_label TEXT NOT NULL DEFAULT '',
                    contrast TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'draft',
                    reviewer TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS task_table_links (
                    link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pmid TEXT NOT NULL,
                    occurrence_key TEXT NOT NULL,
                    table_key TEXT NOT NULL,
                    contrast TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'draft',
                    reviewer TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    UNIQUE (pmid, occurrence_key, table_key)
                );

                CREATE TABLE IF NOT EXISTS coordinate_annotations (
                    coordinate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pmid TEXT NOT NULL,
                    occurrence_key TEXT NOT NULL,
                    table_key TEXT NOT NULL,
                    contrast TEXT NOT NULL DEFAULT '',
                    region TEXT NOT NULL DEFAULT '',
                    hemisphere TEXT NOT NULL DEFAULT '',
                    x TEXT NOT NULL DEFAULT '',
                    y TEXT NOT NULL DEFAULT '',
                    z TEXT NOT NULL DEFAULT '',
                    coordinate_space TEXT NOT NULL DEFAULT '',
                    statistic TEXT NOT NULL DEFAULT '',
                    evidence TEXT NOT NULL DEFAULT '',
                    review_status TEXT NOT NULL DEFAULT 'draft',
                    reviewer TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotation_history (
                    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pmid TEXT NOT NULL DEFAULT '',
                    entity_type TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    action TEXT NOT NULL,
                    reviewer TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_workspace_doc_status ON document_annotations(training_status, review_status);
                CREATE INDEX IF NOT EXISTS idx_workspace_section_pmid ON section_revisions(pmid);
                CREATE INDEX IF NOT EXISTS idx_workspace_task_pmid ON task_annotations(pmid);
                CREATE INDEX IF NOT EXISTS idx_workspace_table_pmid ON activation_table_annotations(pmid);
                CREATE INDEX IF NOT EXISTS idx_workspace_link_pmid ON task_table_links(pmid);
                CREATE INDEX IF NOT EXISTS idx_workspace_coordinate_pmid ON coordinate_annotations(pmid);
                """
            )

    def _history(self, connection: sqlite3.Connection, pmid: str, entity_type: str, entity_key: str, action: str, reviewer: str, payload: Dict[str, Any]) -> None:
        connection.execute(
            """
            INSERT INTO annotation_history(pmid, entity_type, entity_key, action, reviewer, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (pmid, entity_type, entity_key, action, reviewer, json_text(payload), utc_now()),
        )

    def document_annotation(self, pmid: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM document_annotations WHERE pmid=?", (pmid,)).fetchone()
        return dict(row) if row else {}

    def upsert_document(self, pmid: str, values: Dict[str, Any]) -> None:
        now = utc_now()
        record = {
            "training_status": str(values.get("training_status", "not_selected")),
            "review_status": str(values.get("review_status", "not_started")),
            "reviewer": str(values.get("reviewer", "")),
            "note": str(values.get("note", "")),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO document_annotations(pmid, training_status, review_status, reviewer, note, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid) DO UPDATE SET
                    training_status=excluded.training_status,
                    review_status=excluded.review_status,
                    reviewer=excluded.reviewer,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (pmid, record["training_status"], record["review_status"], record["reviewer"], record["note"], now),
            )
            self._history(connection, pmid, "document", pmid, "upsert", record["reviewer"], record)

    def section_revision(self, pmid: str, section_name: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM section_revisions WHERE pmid=? AND section_name=?", (pmid, section_name)
            ).fetchone()
        return dict(row) if row else {}

    def upsert_section_revision(self, pmid: str, section_name: str, values: Dict[str, Any]) -> None:
        now = utc_now()
        record = {
            "base_content_sha256": str(values.get("base_content_sha256", "")),
            "original_content": str(values.get("original_content", "")),
            "revised_content": str(values.get("revised_content", "")),
            "review_status": str(values.get("review_status", "draft")),
            "reviewer": str(values.get("reviewer", "")),
            "rationale": str(values.get("rationale", "")),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO section_revisions(pmid, section_name, base_content_sha256, original_content, revised_content, review_status, reviewer, rationale, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid, section_name) DO UPDATE SET
                    base_content_sha256=excluded.base_content_sha256,
                    original_content=excluded.original_content,
                    revised_content=excluded.revised_content,
                    review_status=excluded.review_status,
                    reviewer=excluded.reviewer,
                    rationale=excluded.rationale,
                    updated_at=excluded.updated_at
                """,
                (pmid, section_name, *record.values(), now),
            )
            self._history(connection, pmid, "section_revision", f"{pmid}:{section_name}", "upsert", record["reviewer"], record)

    def task_annotation(self, occurrence_key: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM task_annotations WHERE occurrence_key=?", (occurrence_key,)).fetchone()
        return dict(row) if row else {}

    def upsert_task_annotation(self, occurrence_key: str, pmid: str, values: Dict[str, Any]) -> None:
        now = utc_now()
        record = {
            "reviewed_task_name": str(values.get("reviewed_task_name", "")),
            "reviewed_description": str(values.get("reviewed_description", "")),
            "reviewed_task_role": str(values.get("reviewed_task_role", "")),
            "in_scanner_status": str(values.get("in_scanner_status", "unknown")),
            "review_status": str(values.get("review_status", "draft")),
            "reviewer": str(values.get("reviewer", "")),
            "note": str(values.get("note", "")),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO task_annotations(occurrence_key, pmid, reviewed_task_name, reviewed_description, reviewed_task_role, in_scanner_status, review_status, reviewer, note, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(occurrence_key) DO UPDATE SET
                    reviewed_task_name=excluded.reviewed_task_name,
                    reviewed_description=excluded.reviewed_description,
                    reviewed_task_role=excluded.reviewed_task_role,
                    in_scanner_status=excluded.in_scanner_status,
                    review_status=excluded.review_status,
                    reviewer=excluded.reviewer,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (occurrence_key, pmid, *record.values(), now),
            )
            self._history(connection, pmid, "task_annotation", occurrence_key, "upsert", record["reviewer"], record)

    def upsert_table_annotation(self, table_key: str, pmid: str, values: Dict[str, Any]) -> None:
        now = utc_now()
        record = {
            "section_name": str(values.get("section_name", "")),
            "table_index": int(values.get("table_index", 0)),
            "source_markdown": str(values.get("source_markdown", "")),
            "is_activation_table": str(values.get("is_activation_table", "unknown")),
            "table_label": str(values.get("table_label", "")),
            "contrast": str(values.get("contrast", "")),
            "review_status": str(values.get("review_status", "draft")),
            "reviewer": str(values.get("reviewer", "")),
            "note": str(values.get("note", "")),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO activation_table_annotations(table_key, pmid, section_name, table_index, source_markdown, is_activation_table, table_label, contrast, review_status, reviewer, note, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(table_key) DO UPDATE SET
                    is_activation_table=excluded.is_activation_table,
                    table_label=excluded.table_label,
                    contrast=excluded.contrast,
                    review_status=excluded.review_status,
                    reviewer=excluded.reviewer,
                    note=excluded.note,
                    updated_at=excluded.updated_at
                """,
                (table_key, pmid, *record.values(), now),
            )
            self._history(connection, pmid, "activation_table", table_key, "upsert", record["reviewer"], record)

    def upsert_task_table_link(self, pmid: str, values: Dict[str, Any]) -> None:
        now = utc_now()
        record = {
            "occurrence_key": str(values.get("occurrence_key", "")),
            "table_key": str(values.get("table_key", "")),
            "contrast": str(values.get("contrast", "")),
            "evidence": str(values.get("evidence", "")),
            "review_status": str(values.get("review_status", "draft")),
            "reviewer": str(values.get("reviewer", "")),
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO task_table_links(pmid, occurrence_key, table_key, contrast, evidence, review_status, reviewer, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pmid, occurrence_key, table_key) DO UPDATE SET
                    contrast=excluded.contrast,
                    evidence=excluded.evidence,
                    review_status=excluded.review_status,
                    reviewer=excluded.reviewer,
                    updated_at=excluded.updated_at
                """,
                (pmid, *record.values(), now),
            )
            self._history(connection, pmid, "task_table_link", f"{record['occurrence_key']}->{record['table_key']}", "upsert", record["reviewer"], record)

    def add_coordinate(self, pmid: str, values: Dict[str, Any]) -> int:
        now = utc_now()
        fields = ["occurrence_key", "table_key", "contrast", "region", "hemisphere", "x", "y", "z", "coordinate_space", "statistic", "evidence", "review_status", "reviewer"]
        record = {field: str(values.get(field, "")) for field in fields}
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO coordinate_annotations(pmid, occurrence_key, table_key, contrast, region, hemisphere, x, y, z, coordinate_space, statistic, evidence, review_status, reviewer, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (pmid, *[record[field] for field in fields], now),
            )
            coordinate_id = int(cursor.lastrowid)
            self._history(connection, pmid, "coordinate", str(coordinate_id), "insert", record["reviewer"], record)
        return coordinate_id

    def delete_coordinate(self, coordinate_id: int) -> None:
        with self._connect() as connection:
            row = connection.execute("SELECT pmid, reviewer FROM coordinate_annotations WHERE coordinate_id=?", (coordinate_id,)).fetchone()
            connection.execute("DELETE FROM coordinate_annotations WHERE coordinate_id=?", (coordinate_id,))
            if row:
                self._history(connection, str(row["pmid"]), "coordinate", str(coordinate_id), "delete", str(row["reviewer"]), {})

    def rows(self, query: str, parameters: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(query, tuple(parameters)).fetchall()]

