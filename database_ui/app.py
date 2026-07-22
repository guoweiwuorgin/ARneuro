"""Flask application for exploring and curating the ARneuro SQLite corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from flask import Flask, jsonify, request, send_from_directory

from .workspace import AnnotationWorkspace, utc_now


STEP3_DIR = Path(r"D:/language_template/reviewer/current_data/step3_library")
DEFAULT_DATABASE = STEP3_DIR / "structured_text_database" / "language_neuro_corpus.sqlite3"
DEFAULT_WORKSPACE = STEP3_DIR / "structured_text_database" / "annotation_workspace" / "annotation_workspace.sqlite3"
TABLE_LINE_RE = re.compile(r"^\s*\|.+\|\s*$")
HTML_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table\s*>", re.IGNORECASE | re.DOTALL)


class HtmlTableParser(HTMLParser):
    """Extract display-safe row and cell text from one HTML table fragment."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[str]] = []
        self._current_row: Optional[List[str]] = None
        self._current_cell: Optional[List[str]] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() == "tr":
            self._current_row = []
        elif tag.lower() in {"td", "th"} and self._current_row is not None:
            self._current_cell = []
        elif tag.lower() == "br" and self._current_cell is not None:
            self._current_cell.append(" ")

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            self._current_row.append(" ".join("".join(self._current_cell).split()))
            self._current_cell = None
        elif tag.lower() == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None


def parse_html_table_rows(fragment: str) -> List[List[str]]:
    parser = HtmlTableParser()
    parser.feed(fragment)
    parser.close()
    return parser.rows


def parse_json(value: Any, default: Any) -> Any:
    try:
        return json.loads(value) if value else default
    except (TypeError, json.JSONDecodeError):
        return default


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_body() -> Dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object request body.")
    return payload


class CorpusRepository:
    def __init__(self, database_path: Path, workspace: AnnotationWorkspace) -> None:
        self.database_path = Path(database_path)
        self.workspace = workspace
        if not self.database_path.exists():
            raise FileNotFoundError(f"Corpus database not found: {self.database_path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def stats(self) -> Dict[str, Any]:
        with self._connect() as connection:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            sections = connection.execute("SELECT COUNT(*) FROM sections").fetchone()[0]
            passages = connection.execute("SELECT COUNT(*) FROM passages").fetchone()[0]
            tasks = connection.execute("SELECT COUNT(*) FROM document_tasks WHERE exclude_from_all_analysis=0").fetchone()[0]
            excluded_tasks = connection.execute("SELECT COUNT(*) FROM document_tasks WHERE exclude_from_all_analysis=1").fetchone()[0]
            raw_tasks = connection.execute("SELECT COUNT(*) FROM task_occurrences WHERE exclude_from_all_analysis=0").fetchone()[0]
            methods = connection.execute("SELECT COUNT(DISTINCT pmid) FROM sections WHERE lower(section_name)='methods'").fetchone()[0]
            results = connection.execute("SELECT COUNT(DISTINCT pmid) FROM sections WHERE lower(section_name)='results'").fetchone()[0]
            tables = connection.execute("SELECT COUNT(DISTINCT pmid) FROM sections WHERE lower(section_name)='tables'").fetchone()[0]
        workspace_counts = self.workspace.rows(
            "SELECT training_status, COUNT(*) AS count FROM document_annotations GROUP BY training_status"
        )
        return {
            "documents": documents,
            "sections": sections,
            "passages": passages,
            "active_task_occurrences": tasks,
            "excluded_task_occurrences": excluded_tasks,
            "raw_active_task_occurrences": raw_tasks,
            "documents_with_methods": methods,
            "documents_with_results": results,
            "documents_with_tables": tables,
            "training_status_counts": {row["training_status"]: row["count"] for row in workspace_counts},
            "workspace_path": str(self.workspace.path),
        }

    def _document_rows(self) -> Tuple[List[sqlite3.Row], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        with self._connect() as connection:
            docs = connection.execute("SELECT * FROM documents ORDER BY pmid").fetchall()
            task_rows = connection.execute(
                """
                SELECT pmid, COUNT(*) AS task_count,
                       GROUP_CONCAT(DISTINCT normalized_task_name) AS task_names,
                       GROUP_CONCAT(DISTINCT task_role) AS task_roles
                FROM document_tasks
                WHERE exclude_from_all_analysis=0
                GROUP BY pmid
                """
            ).fetchall()
            section_rows = connection.execute(
                "SELECT pmid, GROUP_CONCAT(DISTINCT lower(section_name)) AS section_names FROM sections GROUP BY pmid"
            ).fetchall()
        task_index = {str(row["pmid"]): dict(row) for row in task_rows}
        section_index = {str(row["pmid"]): dict(row) for row in section_rows}
        return docs, task_index, section_index

    def documents(self, filters: Dict[str, str]) -> Dict[str, Any]:
        docs, task_index, section_index = self._document_rows()
        annotation_rows = self.workspace.rows("SELECT * FROM document_annotations")
        annotations = {str(row["pmid"]): row for row in annotation_rows}
        query = str(filters.get("q", "")).strip().casefold()
        pmid_filter = str(filters.get("pmid", "")).strip()
        task_filter = str(filters.get("task", "")).strip().casefold()
        role_filter = str(filters.get("task_role", "")).strip()
        fMRI_filter = str(filters.get("fmri", "")).strip()
        scanner_filter = str(filters.get("scanner", "")).strip().casefold()
        training_filter = str(filters.get("training_status", "")).strip()
        needs_filter = str(filters.get("needs", "")).strip()
        records: List[Dict[str, Any]] = []
        for row in docs:
            pmid = str(row["pmid"])
            title = str(row["title"] or "")
            method_info = parse_json(row["method_info_json"], {})
            sample_groups = method_info.get("sample_groups", []) if isinstance(method_info, dict) else []
            if not isinstance(sample_groups, list):
                sample_groups = []
            sample_summary = "; ".join(
                " ".join(
                    part
                    for part in (
                        str(group.get("name", "")).strip(),
                        f"n={group.get('n', '')}" if str(group.get("n", "")).strip() else "",
                    )
                    if part
                ).strip()
                for group in sample_groups
                if isinstance(group, dict)
            )
            task_info = task_index.get(pmid, {})
            sections = set(filter(None, str(section_index.get(pmid, {}).get("section_names", "")).split(",")))
            annotation = annotations.get(pmid, {})
            indexed_names = str(task_info.get("task_names", "")).casefold()
            indexed_roles = str(task_info.get("task_roles", ""))
            acquisition = method_info.get("acquisition", {}) if isinstance(method_info, dict) else {}
            scanner = str(
                method_info.get("scanner_type", "")
                or (acquisition.get("scanner_or_instrument", "") if isinstance(acquisition, dict) else "")
            )
            fmri_values = " | ".join(
                [
                    str(method_info.get("typical_human_all_participated_in_fMRI_task", "unknown")),
                    str(method_info.get("patient_all_participated_in_fMRI_task", "unknown")),
                    " ".join(
                        str(group.get("in_scanner", "unknown"))
                        for group in sample_groups
                        if isinstance(group, dict)
                    ),
                ]
            )
            if pmid_filter and pmid_filter not in pmid:
                continue
            if query and query not in (pmid + " " + title + " " + indexed_names).casefold():
                continue
            if task_filter and task_filter not in indexed_names:
                continue
            if role_filter and role_filter not in indexed_roles:
                continue
            if fMRI_filter and fMRI_filter.casefold() not in fmri_values.casefold():
                continue
            if scanner_filter and scanner_filter not in scanner.casefold():
                continue
            if training_filter and annotation.get("training_status", "not_selected") != training_filter:
                continue
            if needs_filter == "methods" and "methods" not in sections:
                continue
            if needs_filter == "results" and "results" not in sections:
                continue
            if needs_filter == "tables" and "tables" not in sections:
                continue
            records.append(
                {
                    "pmid": pmid,
                    "title": title,
                    "typical_human_total": str(row["typical_human_total_raw"] or ""),
                    "patient_total": str(row["patient_total_raw"] or ""),
                    "sample_summary": sample_summary,
                    "task_count": int(task_info.get("task_count", 0) or 0),
                    "task_roles": [item for item in indexed_roles.split(",") if item],
                    "source_groups": [],
                    "has_methods": "methods" in sections,
                    "has_results": "results" in sections,
                    "has_tables": "tables" in sections,
                    "scanner_type": scanner,
                    "typical_human_fmri": method_info.get("typical_human_all_participated_in_fMRI_task", "unknown"),
                    "patient_fmri": method_info.get("patient_all_participated_in_fMRI_task", "unknown"),
                    "training_status": annotation.get("training_status", "not_selected"),
                    "review_status": annotation.get("review_status", "not_started"),
                    "updated_at": annotation.get("updated_at", ""),
                }
            )
        records.sort(key=lambda item: (item["training_status"] != "in_progress", item["pmid"]))
        page = max(1, int(filters.get("page", "1") or 1))
        page_size = min(100, max(10, int(filters.get("page_size", "40") or 40)))
        start = (page - 1) * page_size
        return {"items": records[start : start + page_size], "total": len(records), "page": page, "page_size": page_size}

    @staticmethod
    def _markdown_tables(pmid: str, sections: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        for section in sections:
            section_name = str(section["section_name"])
            section_text = str(section["content"] or "")
            lines = section_text.splitlines()
            group: List[str] = []
            table_index = 0

            def add_candidate(source_text: str, source_format: str, table_rows: List[List[str]]) -> None:
                nonlocal table_index
                table_index += 1
                digest = hashlib.sha1(
                    f"{pmid}|{section_name}|{source_format}|{source_text}".encode("utf-8")
                ).hexdigest()[:16]
                header = " | ".join(table_rows[0]) if table_rows else source_text.splitlines()[0].strip()
                candidates.append(
                    {
                        "table_key": f"{pmid}:{section_name}:{table_index}:{digest}",
                        "pmid": pmid,
                        "section_name": section_name,
                        "table_index": table_index,
                        "source_markdown": source_text,
                        "table_format": source_format,
                        "table_rows": table_rows,
                        "header": header,
                    }
                )

            def flush() -> None:
                nonlocal group
                if len(group) < 2:
                    group = []
                    return
                markdown = "\n".join(group).strip()
                if markdown.count("|") < 4:
                    group = []
                    return
                add_candidate(markdown, "markdown", [])
                group = []
            for line in lines:
                if TABLE_LINE_RE.match(line):
                    group.append(line)
                else:
                    flush()
            flush()
            for html_table in HTML_TABLE_RE.findall(section_text):
                rows = parse_html_table_rows(html_table)
                if rows:
                    add_candidate(html_table, "html", rows)
        return candidates

    def detail(self, pmid: str) -> Dict[str, Any]:
        with self._connect() as connection:
            document = connection.execute("SELECT * FROM documents WHERE pmid=?", (pmid,)).fetchone()
            if document is None:
                raise KeyError(f"PMID not found: {pmid}")
            sections = [dict(row) for row in connection.execute("SELECT * FROM sections WHERE pmid=? ORDER BY section_order", (pmid,))]
            tasks = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM document_tasks WHERE pmid=? ORDER BY exclude_from_all_analysis, final_task_name, document_task_key", (pmid,)
                )
            ]
        document_record = dict(document)
        document_record["method_info"] = parse_json(document_record.pop("method_info_json"), {})
        section_revisions = {
            row["section_name"]: row
            for row in self.workspace.rows("SELECT * FROM section_revisions WHERE pmid=?", (pmid,))
        }
        task_revisions = {
            row["occurrence_key"]: row for row in self.workspace.rows("SELECT * FROM task_annotations WHERE pmid=?", (pmid,))
        }
        for section in sections:
            section["revision"] = section_revisions.get(section["section_name"], {})
        for task in tasks:
            task["occurrence_key"] = task["document_task_key"]
            task["member_occurrence_keys"] = parse_json(task.pop("member_occurrence_keys_json", "[]"), [])
            task["member_task_indices"] = parse_json(task.pop("member_task_indices_json", "[]"), [])
            task["description_variants"] = parse_json(task.pop("description_variants_json", "[]"), [])
            task["clue_sentence_variants"] = parse_json(task.pop("clue_sentence_variants_json", "[]"), [])
            task["review"] = task_revisions.get(task["occurrence_key"], {})
        candidates = self._markdown_tables(pmid, sections)
        table_reviews = {
            row["table_key"]: row
            for row in self.workspace.rows("SELECT * FROM activation_table_annotations WHERE pmid=?", (pmid,))
        }
        for candidate in candidates:
            candidate["review"] = table_reviews.get(candidate["table_key"], {})
        links = self.workspace.rows("SELECT * FROM task_table_links WHERE pmid=? ORDER BY link_id", (pmid,))
        coordinates = self.workspace.rows("SELECT * FROM coordinate_annotations WHERE pmid=? ORDER BY coordinate_id", (pmid,))
        history = self.workspace.rows(
            "SELECT * FROM annotation_history WHERE pmid=? ORDER BY history_id DESC LIMIT 60", (pmid,)
        )
        document_annotation = self.workspace.document_annotation(pmid)
        qcs = self._quality_flags(sections, tasks, candidates, links, coordinates)
        return {
            "document": document_record,
            "document_annotation": document_annotation,
            "sections": sections,
            "tasks": tasks,
            "table_candidates": candidates,
            "links": links,
            "coordinates": coordinates,
            "history": history,
            "quality_flags": qcs,
        }

    @staticmethod
    def _quality_flags(sections: List[Dict[str, Any]], tasks: List[Dict[str, Any]], tables: List[Dict[str, Any]], links: List[Dict[str, Any]], coordinates: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        names = {str(section["section_name"]).casefold() for section in sections}
        active_tasks = [task for task in tasks if not int(task.get("exclude_from_all_analysis", 0))]
        linked_keys = {str(link["occurrence_key"]) for link in links}
        flags: List[Dict[str, str]] = []
        if "methods" not in names:
            flags.append({"level": "error", "message": "Methods section is unavailable."})
        if "results" not in names:
            flags.append({"level": "warning", "message": "Results section is unavailable."})
        if not active_tasks:
            flags.append({"level": "warning", "message": "No active task occurrence is indexed."})
        if not tables:
            flags.append({"level": "warning", "message": "No Markdown table candidate was detected in the segmented document."})
        unlinked = len([task for task in active_tasks if task["occurrence_key"] not in linked_keys])
        if active_tasks and unlinked:
            flags.append({"level": "info", "message": f"{unlinked} active task occurrence(s) are not linked to an activation table."})
        invalid_coordinates = [row for row in coordinates if not (row.get("x") and row.get("y") and row.get("z"))]
        if invalid_coordinates:
            flags.append({"level": "warning", "message": f"{len(invalid_coordinates)} coordinate annotation(s) lack one or more x/y/z values."})
        return flags

    def training_export(self, output_dir: Path) -> Dict[str, Any]:
        output_dir.mkdir(parents=True, exist_ok=True)
        gold_rows = self.workspace.rows("SELECT * FROM document_annotations WHERE training_status='gold' ORDER BY pmid")
        task_path = output_dir / "task_extraction_gold.jsonl"
        link_path = output_dir / "task_table_linking_gold.jsonl"
        coordinate_path = output_dir / "coordinate_extraction_gold.jsonl"
        counts = {"documents": 0, "task_examples": 0, "link_examples": 0, "coordinate_examples": 0}
        with task_path.open("w", encoding="utf-8") as task_file, link_path.open("w", encoding="utf-8") as link_file, coordinate_path.open("w", encoding="utf-8") as coordinate_file:
            for annotation in gold_rows:
                pmid = str(annotation["pmid"])
                detail = self.detail(pmid)
                section_map = {str(section["section_name"]).casefold(): section for section in detail["sections"]}
                methods = section_map.get("methods", {})
                results = section_map.get("results", {})
                methods_text = str(methods.get("revision", {}).get("revised_content") or methods.get("content", ""))
                results_text = str(results.get("revision", {}).get("revised_content") or results.get("content", ""))
                active_tasks = [task for task in detail["tasks"] if not int(task.get("exclude_from_all_analysis", 0))]
                target_tasks = []
                for task in active_tasks:
                    review = task.get("review", {})
                    target_tasks.append(
                        {
                            "occurrence_key": task["occurrence_key"],
                            "task_name": review.get("reviewed_task_name") or task["final_task_name"],
                            "description": review.get("reviewed_description") or task["description"],
                            "task_role": review.get("reviewed_task_role") or task["task_role"],
                            "in_scanner_status": review.get("in_scanner_status") or "unknown",
                            "evidence": task["clue_sentences"],
                        }
                    )
                base = {"pmid": pmid, "methods": methods_text, "results": results_text, "annotation_status": "gold"}
                task_file.write(json.dumps({**base, "target": {"tasks": target_tasks}}, ensure_ascii=False) + "\n")
                counts["task_examples"] += 1
                linked = [link for link in detail["links"]]
                link_file.write(json.dumps({**base, "target": {"task_table_links": linked}}, ensure_ascii=False) + "\n")
                counts["link_examples"] += 1
                coordinate_file.write(json.dumps({**base, "target": {"coordinates": detail["coordinates"]}}, ensure_ascii=False) + "\n")
                counts["coordinate_examples"] += 1
                counts["documents"] += 1
        return {**counts, "task_path": str(task_path), "link_path": str(link_path), "coordinate_path": str(coordinate_path), "generated_at": utc_now()}


def create_app(database_path: Path = DEFAULT_DATABASE, workspace_path: Path = DEFAULT_WORKSPACE) -> Flask:
    static_dir = Path(__file__).resolve().parent / "static"
    app = Flask(__name__, static_folder=str(static_dir), static_url_path="/static")
    repository = CorpusRepository(database_path, AnnotationWorkspace(workspace_path))

    @app.get("/")
    def index() -> Any:
        return send_from_directory(static_dir, "index.html")

    @app.get("/api/stats")
    def api_stats() -> Any:
        return jsonify(repository.stats())

    @app.get("/api/documents")
    def api_documents() -> Any:
        return jsonify(repository.documents(dict(request.args)))

    @app.get("/api/document/<pmid>")
    def api_document(pmid: str) -> Any:
        try:
            return jsonify(repository.detail(pmid))
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404

    @app.post("/api/document/<pmid>/annotation")
    def api_document_annotation(pmid: str) -> Any:
        payload = json_body()
        repository.workspace.upsert_document(pmid, payload)
        return jsonify({"ok": True})

    @app.post("/api/document/<pmid>/section/<section_name>")
    def api_section_revision(pmid: str, section_name: str) -> Any:
        payload = json_body()
        repository.workspace.upsert_section_revision(pmid, section_name, payload)
        return jsonify({"ok": True})

    @app.post("/api/document/<pmid>/task/<path:occurrence_key>")
    def api_task_annotation(pmid: str, occurrence_key: str) -> Any:
        payload = json_body()
        repository.workspace.upsert_task_annotation(occurrence_key, pmid, payload)
        return jsonify({"ok": True})

    @app.post("/api/document/<pmid>/table/<path:table_key>")
    def api_table_annotation(pmid: str, table_key: str) -> Any:
        payload = json_body()
        repository.workspace.upsert_table_annotation(table_key, pmid, payload)
        return jsonify({"ok": True})

    @app.post("/api/document/<pmid>/link")
    def api_task_table_link(pmid: str) -> Any:
        payload = json_body()
        repository.workspace.upsert_task_table_link(pmid, payload)
        return jsonify({"ok": True})

    @app.post("/api/document/<pmid>/coordinate")
    def api_coordinate(pmid: str) -> Any:
        payload = json_body()
        coordinate_id = repository.workspace.add_coordinate(pmid, payload)
        return jsonify({"ok": True, "coordinate_id": coordinate_id})

    @app.delete("/api/coordinate/<int:coordinate_id>")
    def api_delete_coordinate(coordinate_id: int) -> Any:
        repository.workspace.delete_coordinate(coordinate_id)
        return jsonify({"ok": True})

    @app.post("/api/export")
    def api_export() -> Any:
        payload = request.get_json(silent=True)
        output_dir = payload.get("output_dir") if isinstance(payload, dict) else None
        target_dir = Path(output_dir) if output_dir else repository.workspace.path.parent / "exports"
        return jsonify(repository.training_export(target_dir))

    @app.errorhandler(ValueError)
    def value_error(exc: ValueError) -> Any:
        return jsonify({"error": str(exc)}), 400

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ARneuro corpus annotation workspace.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.database, args.workspace)
    print(f"ARneuro Corpus Workspace: http://{args.host}:{args.port}")
    print(f"Corpus database: {args.database}")
    print(f"Annotation workspace: {args.workspace}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
