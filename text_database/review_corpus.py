"""Build a generic, lightweight ARneuro SQLite review corpus.

The resulting database deliberately retains the stable document/section/passage
tables used by the annotation UI while adding domain-neutral ``study_metadata``
and ``experimental_tasks`` tables.  It therefore supports both broad review
projects and later human correction / LoRA export without a server dependency.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .structured_corpus import (
    StructuredCorpusBuilder,
    content_sha256,
    extract_sections,
    normalize_task_name,
    safe_json,
    split_section_to_passages,
    utc_now,
)


@dataclass(frozen=True)
class ReviewCorpusConfig:
    canonical_articles_csv: Path
    segmented_dir: Path
    study_info_dir: Path
    output_database: Path
    passage_max_chars: int = 1800
    rebuild: bool = True


class ReviewCorpusBuilder:
    """Create a project-local SQLite corpus from segmentation and generic JSON."""

    def __init__(self, config: ReviewCorpusConfig) -> None:
        self.config = config

    @staticmethod
    def _records(path: Path) -> Dict[str, Dict[str, str]]:
        import csv

        with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
            return {str(row.get("PMID", "")).strip(): dict(row) for row in csv.DictReader(file_obj) if str(row.get("PMID", "")).strip()}

    @staticmethod
    def _content_paths(directory: Path) -> Dict[str, Path]:
        paths: Dict[str, Path] = {}
        for path in directory.glob("paper_*_structured_content.json"):
            parts = path.stem.split("_")
            if len(parts) >= 4 and parts[1].isdigit():
                paths[parts[1]] = path
        return paths

    @staticmethod
    def _study_info(path: Path, pmid: str) -> Dict[str, Any]:
        candidates = (path / f"paper_{pmid}_study_info.json", path / f"{pmid}_study_info.json")
        for candidate in candidates:
            if candidate.exists():
                try:
                    value = json.loads(candidate.read_text(encoding="utf-8"))
                    if isinstance(value, dict) and isinstance(value.get("data"), dict):
                        return dict(value["data"])
                    return value if isinstance(value, dict) else {}
                except json.JSONDecodeError:
                    return {}
        return {}

    @staticmethod
    def _task_rows(pmid: str, study_info: Mapping[str, Any]) -> List[Dict[str, str]]:
        tasks = study_info.get("experimental_tasks", [])
        if not isinstance(tasks, list):
            return []
        rows: List[Dict[str, str]] = []
        for index, task in enumerate(tasks, start=1):
            if not isinstance(task, dict):
                continue
            name = str(task.get("name", "") or "").strip()
            if not name:
                continue
            key = "generic:" + hashlib.sha1(f"{pmid}|{index}|{name}".encode("utf-8")).hexdigest()[:20]
            rows.append(
                {
                    "occurrence_key": key,
                    "pmid": pmid,
                    "task_index": str(index),
                    "base_task_name": name,
                    "final_task_name": name,
                    "normalized_task_name": normalize_task_name(name),
                    "description": str(task.get("description", "") or ""),
                    "clue_sentences": str(task.get("evidence", "") or ""),
                    "task_role": str(task.get("category", "experimental_task") or "experimental_task"),
                    "stimulus_modality": ", ".join(str(value) for value in task.get("stimulus_modality", []) if value) if isinstance(task.get("stimulus_modality"), list) else "",
                    "response_mode": str(task.get("response_mode", "") or ""),
                    "source_json": safe_json(task),
                }
            )
        return rows

    def build(self) -> Dict[str, Any]:
        config = self.config
        if not config.canonical_articles_csv.exists():
            raise FileNotFoundError(f"Canonical article CSV not found: {config.canonical_articles_csv}")
        if not config.segmented_dir.exists():
            raise FileNotFoundError(f"Segmented directory not found: {config.segmented_dir}")
        config.output_database.parent.mkdir(parents=True, exist_ok=True)
        if config.output_database.exists() and not config.rebuild:
            raise FileExistsError(f"Database already exists: {config.output_database}")
        temporary = config.output_database.with_suffix(".building.sqlite3")
        if temporary.exists():
            temporary.unlink()
        records = self._records(config.canonical_articles_csv)
        content_paths = self._content_paths(config.segmented_dir)
        connection = StructuredCorpusBuilder._connect(temporary)
        counts = {"documents": 0, "sections": 0, "passages": 0, "experimental_tasks": 0, "missing_segmented": []}
        try:
            StructuredCorpusBuilder._create_schema(connection)
            connection.executescript(
                """
                CREATE TABLE study_metadata (
                    pmid TEXT PRIMARY KEY REFERENCES documents(pmid) ON DELETE CASCADE,
                    study_type TEXT NOT NULL DEFAULT 'unknown',
                    research_modalities_json TEXT NOT NULL DEFAULT '[]',
                    study_design TEXT NOT NULL DEFAULT '',
                    sample_groups_json TEXT NOT NULL DEFAULT '[]',
                    acquisition_json TEXT NOT NULL DEFAULT '{}',
                    analysis_json TEXT NOT NULL DEFAULT '{}',
                    key_outcomes_json TEXT NOT NULL DEFAULT '[]',
                    missing_or_uncertain_fields_json TEXT NOT NULL DEFAULT '[]',
                    evidence_json TEXT NOT NULL DEFAULT '{}',
                    extraction_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE source_records (
                    pmid TEXT PRIMARY KEY REFERENCES documents(pmid) ON DELETE CASCADE,
                    title TEXT NOT NULL DEFAULT '', abstract TEXT NOT NULL DEFAULT '', doi TEXT NOT NULL DEFAULT '',
                    pmcid TEXT NOT NULL DEFAULT '', authors TEXT NOT NULL DEFAULT '', journal TEXT NOT NULL DEFAULT '',
                    year TEXT NOT NULL DEFAULT '', source_row INTEGER NOT NULL DEFAULT 0
                );
                CREATE VIEW experimental_tasks AS
                SELECT document_task_key AS task_key, pmid, final_task_name AS task_name,
                       task_role AS category, description, clue_sentences AS evidence,
                       stimulus_modality, response_mode, source_groups_json AS source_json
                FROM document_tasks;
                """
            )
            for pmid, record in sorted(records.items()):
                content_path = content_paths.get(pmid)
                if content_path is None:
                    counts["missing_segmented"].append(pmid)
                    continue
                try:
                    content = json.loads(content_path.read_text(encoding="utf-8"))
                    if not isinstance(content, dict):
                        raise ValueError("structured content must be a JSON object")
                    study_info = self._study_info(config.study_info_dir, pmid)
                    sections = extract_sections(content)
                    title = str(content.get("Title", "") or record.get("title", "")).strip()
                    source_markdown = str((config.segmented_dir.parent / "03_markdown" / f"paper_{pmid}.md"))
                    connection.execute(
                        """INSERT INTO documents(pmid,title,method_info_json,content_json_path,meta_json_path,source_markdown_path,content_sha256)
                           VALUES(?,?,?,?,?,?,?)""",
                        (pmid, title, safe_json(study_info), str(content_path), "", source_markdown, content_sha256(content_path)),
                    )
                    connection.execute(
                        """INSERT INTO source_records(pmid,title,abstract,doi,pmcid,authors,journal,year,source_row)
                           VALUES(?,?,?,?,?,?,?,?,?)""",
                        (pmid, record.get("title", ""), record.get("abstract", ""), record.get("doi", ""), record.get("pmcid", ""), record.get("authors", ""), record.get("journal", ""), record.get("year", ""), int(record.get("source_row", 0) or 0)),
                    )
                    connection.execute(
                        """INSERT INTO study_metadata(pmid,study_type,research_modalities_json,study_design,sample_groups_json,
                           acquisition_json,analysis_json,key_outcomes_json,missing_or_uncertain_fields_json,evidence_json,extraction_json)
                           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (pmid, str(study_info.get("study_type", "unknown")), safe_json(study_info.get("research_modalities", [])),
                         str(study_info.get("study_design", "")), safe_json(study_info.get("sample_groups", [])),
                         safe_json(study_info.get("acquisition", {})), safe_json(study_info.get("analysis", {})),
                         safe_json(study_info.get("key_outcomes", [])), safe_json(study_info.get("missing_or_uncertain_fields", [])),
                         safe_json(study_info.get("evidence", {})), safe_json(study_info)),
                    )
                    passage_count = 0
                    for section_order, (section_name, section_text) in enumerate(sections):
                        connection.execute("INSERT INTO sections(pmid,section_name,section_order,content,char_count) VALUES(?,?,?,?,?)", (pmid, section_name, section_order, section_text, len(section_text)))
                        for heading_path, start, end, text in split_section_to_passages(section_text, config.passage_max_chars):
                            cursor = connection.execute("INSERT INTO passages(pmid,section_name,section_order,heading_path,passage_order,char_start,char_end,text) VALUES(?,?,?,?,?,?,?,?)", (pmid, section_name, section_order, heading_path, passage_count, start, end, text))
                            connection.execute("INSERT INTO passage_fts(rowid,text,heading_path,section_name,pmid) VALUES(?,?,?,?,?)", (int(cursor.lastrowid), text, heading_path, section_name, pmid))
                            passage_count += 1
                    connection.execute("UPDATE documents SET section_count=?, passage_count=? WHERE pmid=?", (len(sections), passage_count, pmid))
                    task_rows = self._task_rows(pmid, study_info)
                    for task in task_rows:
                        connection.execute(
                            """INSERT INTO task_occurrences(occurrence_key,pmid,task_index,base_task_name,final_task_name,normalized_task_name,description,clue_sentences,task_role,stimulus_modality,response_mode,source_json)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                            tuple(task[key] for key in ("occurrence_key", "pmid", "task_index", "base_task_name", "final_task_name", "normalized_task_name", "description", "clue_sentences", "task_role", "stimulus_modality", "response_mode", "source_json")),
                        )
                    for task in StructuredCorpusBuilder._aggregate_document_tasks(task_rows):
                        connection.execute(
                            """INSERT INTO document_tasks(document_task_key,pmid,normalized_task_name,final_task_name,representative_occurrence_key,description,clue_sentences,task_role,stimulus_modality,response_mode,member_occurrence_count,source_group_count,source_groups_json,member_occurrence_keys_json,member_task_indices_json,description_variants_json,clue_sentence_variants_json,task_roles_json,exclude_from_all_analysis,analysis_exclusion_reason)
                               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                            (task["document_task_key"], task["pmid"], task["normalized_task_name"], task["final_task_name"], task["representative_occurrence_key"], task["description"], task["clue_sentences"], task["task_role"], task["stimulus_modality"], task["response_mode"], task["member_occurrence_count"], task["source_group_count"], task["source_groups_json"], task["member_occurrence_keys_json"], task["member_task_indices_json"], task["description_variants_json"], task["clue_sentence_variants_json"], task["task_roles_json"], task["exclude_from_all_analysis"], task["analysis_exclusion_reason"]),
                        )
                    counts["documents"] += 1
                    counts["sections"] += len(sections)
                    counts["passages"] += passage_count
                    counts["experimental_tasks"] += len(task_rows)
                except Exception:
                    connection.rollback()
                    raise
            metadata = {"schema_version": "2.0", "built_at_utc": utc_now(), "corpus_type": "generic_review", **counts}
            connection.executemany("INSERT INTO corpus_metadata(key,value) VALUES(?,?)", [(key, safe_json(value) if isinstance(value, (list, dict)) else str(value)) for key, value in metadata.items()])
            connection.commit()
        finally:
            connection.close()
        temporary.replace(config.output_database)
        return {**counts, "database_path": str(config.output_database), "built_at_utc": utc_now()}
