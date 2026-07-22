"""Build and query a portable, provenance-preserving literature text corpus.

The corpus is intentionally SQLite-only.  It stores complete segmented sections,
heading-aware passages, task annotations, and an FTS5 passage index in one file.
This keeps the database easy to copy, inspect, and reuse in later graph or LoRA
work without requiring a server process.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


SCHEMA_VERSION = "1.4"
COUNT_SEQUENCE_RE = re.compile(r"^\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?)*$")
MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
CONTENT_NAME_RE = re.compile(r"^paper_(?P<pmid>\d+)_structured_content\.json$", re.IGNORECASE)
META_NAME_RE = re.compile(r"^paper_(?P<pmid>\d+)_structured_meta\.json$", re.IGNORECASE)

SECTION_ORDER = [
    "Title",
    "Author",
    "Keywords",
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "Conclusion",
    "Acknowledgements",
    "Other",
    "References",
    "Tables",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_space(value: Any) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def normalize_match_text(value: Any) -> str:
    return normalize_space(value).casefold()


def normalize_task_name(value: Any) -> str:
    """Return the stable, lowercase task-name key used for all task lookups.

    The display/original task name remains untouched in ``final_task_name``.  This
    key only standardizes whitespace and letter case, so it never silently merges
    semantically different tasks.
    """

    return normalize_space(value).casefold()


def is_valid_participant_count(value: Any) -> bool:
    """Accept one numeric count or an ordered numeric count sequence.

    Ordered sequences such as ``9, 8, 7`` are valid because they represent
    separate experiments in the existing method-information convention.
    """

    if value is None:
        return False
    text = str(value).strip()
    if not text or text.casefold() in {"na", "nan", "null", "none", "unknown", "not reported"}:
        return False
    return bool(COUNT_SEQUENCE_RE.fullmatch(text))


def parse_bool(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def detect_csv_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:16384]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;").delimiter
    except csv.Error:
        return ";" if sample.splitlines() and sample.splitlines()[0].count(";") > sample.splitlines()[0].count(",") else ","


def read_csv_records(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    delimiter = detect_csv_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return [dict(row) for row in csv.DictReader(file_obj, delimiter=delimiter)]


def content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_heading(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().strip("# ")).strip()


def text_from_table_list(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    chunks: List[str] = []
    for item in value:
        if isinstance(item, dict):
            content = item.get("content", "")
            if content:
                chunks.append(str(content))
        elif item:
            chunks.append(str(item))
    return "\n\n".join(chunks)


def extract_sections(content: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract known and future text sections while avoiding internal metadata."""

    sections: List[Tuple[str, str]] = []
    consumed: Set[str] = set()
    for name in SECTION_ORDER:
        if name not in content:
            continue
        value = content[name]
        text = text_from_table_list(value) if name == "Tables" else (value if isinstance(value, str) else "")
        text = str(text or "").strip()
        if text:
            sections.append((name, text))
        consumed.add(name)

    ignored = {"Methods_Hierarchy", "Results_Hierarchy", "blocks", "metadata", "images"}
    for name, value in content.items():
        if name in consumed or name in ignored or not isinstance(value, str):
            continue
        text = value.strip()
        if text:
            sections.append((str(name), text))
    return sections


def split_text_at_boundary(text: str, max_chars: int) -> Iterator[Tuple[int, int, str]]:
    """Yield compact pieces while preserving offsets into the original section."""

    if len(text) <= max_chars:
        yield 0, len(text), text
        return
    start = 0
    length = len(text)
    while start < length:
        end = min(length, start + max_chars)
        if end < length:
            candidates = [
                text.rfind("\n\n", start + max_chars // 2, end),
                text.rfind(". ", start + max_chars // 2, end),
                text.rfind(" ", start + max_chars // 2, end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + (2 if text.startswith(". ", boundary) else 0)
        chunk = text[start:end].strip()
        if chunk:
            left = text.find(chunk, start, end)
            yield left, left + len(chunk), chunk
        start = max(end, start + 1)


def split_section_to_passages(section_text: str, max_chars: int) -> Iterator[Tuple[str, int, int, str]]:
    """Split Markdown-like text into heading-aware, offset-preserving passages."""

    headings: List[Tuple[int, str]] = []
    chunks: List[Tuple[int, int, str, str]] = []
    paragraph_start: Optional[int] = None
    current_heading = ""
    cursor = 0
    lines = section_text.splitlines(keepends=True)

    def flush_paragraph(end_pos: int) -> None:
        nonlocal paragraph_start
        if paragraph_start is None:
            return
        raw = section_text[paragraph_start:end_pos]
        stripped = raw.strip()
        if stripped:
            relative = raw.find(stripped)
            begin = paragraph_start + relative
            chunks.append((begin, begin + len(stripped), current_heading, stripped))
        paragraph_start = None

    for line in lines:
        line_start = cursor
        cursor += len(line)
        heading_match = MARKDOWN_HEADING_RE.match(line.strip())
        if heading_match:
            flush_paragraph(line_start)
            level = len(heading_match.group(1))
            heading = clean_heading(heading_match.group(2))
            headings[:] = [(old_level, old_heading) for old_level, old_heading in headings if old_level < level]
            headings.append((level, heading))
            current_heading = " > ".join(item[1] for item in headings)
            continue
        if not line.strip():
            flush_paragraph(line_start)
            continue
        if paragraph_start is None:
            paragraph_start = line_start
    flush_paragraph(len(section_text))

    if not chunks and section_text.strip():
        stripped = section_text.strip()
        begin = section_text.find(stripped)
        chunks.append((begin, begin + len(stripped), "", stripped))

    for begin, end, heading_path, text in chunks:
        for relative_start, relative_end, part in split_text_at_boundary(text, max_chars):
            yield heading_path, begin + relative_start, begin + relative_end, part


def phrase_pattern(term: str) -> re.Pattern[str]:
    words = [part for part in re.split(r"\s+", normalize_space(term)) if part]
    if not words:
        return re.compile(r"(?!x)x")
    expression = r"\s+".join(re.escape(word) for word in words)
    return re.compile(r"(?<!\w)" + expression + r"(?!\w)", flags=re.IGNORECASE)


def count_term_occurrences(text: str, term: str) -> int:
    return len(phrase_pattern(term).findall(text))


def fts_phrase(term: str) -> str:
    cleaned = normalize_space(term).replace('"', " ")
    if not cleaned:
        raise ValueError("Search term cannot be empty.")
    return '"' + cleaned + '"'


@dataclass
class CorpusBuildConfig:
    method_info_csv: Path
    segmented_dir: Path
    output_dir: Path
    task_occurrences_csv: Optional[Path] = None
    task_review_csv: Optional[Path] = None
    task_exclusion_overrides_csv: Optional[Path] = None
    database_name: str = "language_neuro_corpus.sqlite3"
    passage_max_chars: int = 1800
    rebuild: bool = True

    @property
    def database_path(self) -> Path:
        return self.output_dir / self.database_name


@dataclass
class CorpusBuildResult:
    database_path: Path
    manifest_path: Path
    eligible_pmids: int
    indexed_documents: int
    indexed_sections: int
    indexed_passages: int
    indexed_tasks: int
    indexed_document_tasks: int = 0
    missing_content_pmids: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)


class StructuredCorpusBuilder:
    """Build the SQLite corpus from method information and segmented JSON files."""

    def __init__(self, config: CorpusBuildConfig) -> None:
        self.config = config

    def _eligible_method_rows(self) -> Tuple[Dict[str, Dict[str, str]], List[Dict[str, str]]]:
        rows = read_csv_records(self.config.method_info_csv)
        eligible: Dict[str, Dict[str, str]] = {}
        duplicates: List[Dict[str, str]] = []
        for row in rows:
            pmid = str(row.get("PMID", "")).strip()
            if not pmid:
                continue
            if not (is_valid_participant_count(row.get("typical_human_total")) or is_valid_participant_count(row.get("patient_total"))):
                continue
            if pmid in eligible:
                duplicates.append({"PMID": pmid, "reason": "duplicate eligible method-info row; last row retained"})
            eligible[pmid] = row
        return eligible, duplicates

    def _segmented_paths(self) -> Tuple[Dict[str, Path], Dict[str, Path]]:
        content_paths: Dict[str, Path] = {}
        meta_paths: Dict[str, Path] = {}
        for path in self.config.segmented_dir.glob("paper_*_structured_content.json"):
            match = CONTENT_NAME_RE.match(path.name)
            if match:
                content_paths[match.group("pmid")] = path
        for path in self.config.segmented_dir.glob("paper_*_structured_meta.json"):
            match = META_NAME_RE.match(path.name)
            if match:
                meta_paths[match.group("pmid")] = path
        return content_paths, meta_paths

    @staticmethod
    def _task_key(row: Dict[str, Any], cluster_id: Optional[Any] = None) -> str:
        pmid = str(row.get("PMID", "")).strip()
        task_index = str(row.get("task_index", "")).strip()
        task_id = str(row.get("task_id", "")).strip()
        source_group = str(row.get("source_group", "")).strip()
        if cluster_id is None:
            cluster_id = row.get("cluster_id", row.get("hier_cluster_k258", ""))
        cluster_text = str(cluster_id or "").strip()
        try:
            cluster_text = f"{int(float(cluster_text)):03d}"
        except Exception:
            pass
        return "|".join([cluster_text, pmid, task_index, task_id, source_group])

    def _load_task_exclusion_overrides(self) -> Dict[str, str]:
        """Load durable occurrence-level exclusions without altering source annotations."""

        overrides: Dict[str, str] = {}
        for row in read_csv_records(self.config.task_exclusion_overrides_csv):
            occurrence_key = str(row.get("occurrence_key", "")).strip()
            if not occurrence_key or not parse_bool(row.get("exclude_from_all_analysis", True)):
                continue
            overrides[occurrence_key] = str(
                row.get("analysis_exclusion_reason") or row.get("reason") or "external analysis exclusion"
            ).strip()
        return overrides

    def _load_task_annotations(self, eligible_pmids: Set[str]) -> List[Dict[str, Any]]:
        base_rows = read_csv_records(self.config.task_occurrences_csv)
        review_rows = read_csv_records(self.config.task_review_csv)
        exclusion_overrides = self._load_task_exclusion_overrides()
        review_by_key = {
            str(row.get("membership_key", "")).strip(): row
            for row in review_rows
            if str(row.get("membership_key", "")).strip()
        }
        fallback_reviews: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        for row in review_rows:
            fallback = "|".join(
                [str(row.get(key, "")).strip() for key in ("PMID", "task_index", "task_id", "source_group")]
            )
            if fallback.strip("|"):
                fallback_reviews[fallback].append(row)

        annotations: List[Dict[str, Any]] = []
        for row in base_rows:
            pmid = str(row.get("PMID", "")).strip()
            if pmid not in eligible_pmids:
                continue
            membership_key = self._task_key(row)
            review = review_by_key.get(membership_key, {})
            if not review:
                fallback = "|".join(
                    [str(row.get(key, "")).strip() for key in ("PMID", "task_index", "task_id", "source_group")]
                )
                candidates = fallback_reviews.get(fallback, [])
                if len(candidates) == 1:
                    review = candidates[0]

            decision = str(review.get("membership_decision", "")).strip()
            reclassified = str(review.get("reclassified_task_name", "")).strip()
            reviewed_name = str(review.get("reviewed_task_name", "")).strip()
            base_name = (
                str(row.get("base_task_name", "")).strip()
                or str(row.get("structured_task_name", "")).strip()
                or str(row.get("canonical_task_name", "")).strip()
                or str(row.get("standardized_task_name", "")).strip()
                or str(row.get("raw_task_name", "")).strip()
            )
            precomputed_final_name = str(row.get("final_task_name", "")).strip()
            final_name = (
                reclassified
                if decision == "reclassify_to_new_task" and reclassified
                else (reviewed_name or precomputed_final_name or base_name)
            )
            source_excluded = parse_bool(row.get("exclude_from_all_analysis"))
            review_excluded = parse_bool(review.get("exclude_from_all_analysis")) or decision == "exclude_from_all_analysis"
            override_reason = exclusion_overrides.get(membership_key, "")
            if override_reason:
                analysis_exclusion_reason = override_reason
            elif review_excluded:
                analysis_exclusion_reason = str(review.get("exclusion_reason", "") or "manual review exclusion").strip()
            elif source_excluded:
                analysis_exclusion_reason = "final task annotation exclusion"
            else:
                analysis_exclusion_reason = ""
            annotations.append(
                {
                    "occurrence_key": membership_key,
                    "pmid": pmid,
                    "task_index": str(row.get("task_index", "")).strip(),
                    "task_id": str(row.get("task_id", "")).strip(),
                    "source_group": str(row.get("source_group", "")).strip(),
                    "cluster_id": str(review.get("cluster_id") or row.get("hier_cluster_k258", "")).strip(),
                    "base_task_name": base_name,
                    "final_task_name": final_name,
                    "normalized_task_name": normalize_task_name(final_name),
                    "description": str(row.get("description", "")).strip(),
                    "clue_sentences": str(row.get("clue_sentences", "")).strip(),
                    "normalized_task_description": str(row.get("normalized_task_description", "")).strip(),
                    "task_existence": str(row.get("task_existence", "")).strip(),
                    "task_existence_evidence": str(row.get("task_existence_evidence", "")).strip(),
                    "task_naming_rationale": str(row.get("task_naming_rationale", "")).strip(),
                    "supported_task_contrast": str(row.get("supported_task_contrast", "")).strip(),
                    "annotation_source": str(row.get("annotation_source", "")).strip() or ("manual_review" if review else "base"),
                    "task_role": str(row.get("task_role", "")).strip(),
                    "is_language_task": str(row.get("is_language_task", "")).strip(),
                    "is_control_task": str(row.get("is_control_task", "")).strip(),
                    "linguistic_domain": str(row.get("linguistic_domain", "")).strip(),
                    "processing_operation": str(row.get("processing_operation", "")).strip(),
                    "stimulus_modality": str(row.get("stimulus_modality", "")).strip(),
                    "stimulus_unit": str(row.get("stimulus_unit", "")).strip(),
                    "response_mode": str(row.get("response_mode", "")).strip(),
                    "membership_decision": decision,
                    "exclude_from_all_analysis": source_excluded or review_excluded or bool(override_reason),
                    "analysis_exclusion_reason": analysis_exclusion_reason,
                    "review_json": safe_json(review) if review else "",
                    "source_json": safe_json(row),
                }
            )
        return annotations

    @staticmethod
    def _aggregate_document_tasks(task_annotations: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create one analysis task per PMID and normalized task name.

        Raw group-specific occurrences remain in ``task_occurrences``. This layer
        provides the default unit for article-level task displays and analyses.
        """

        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
        for task in task_annotations:
            normalized_name = str(task.get("normalized_task_name", "")).strip()
            group_name = normalized_name or "__raw__" + str(task["occurrence_key"])
            grouped[(str(task["pmid"]), group_name)].append(task)

        def rank(task: Dict[str, Any]) -> Tuple[int, int, int, str, str]:
            return (
                int(bool(task.get("exclude_from_all_analysis"))),
                int(not bool(str(task.get("description", "")).strip())),
                int(not bool(str(task.get("clue_sentences", "")).strip())),
                str(task.get("task_index", "")),
                str(task.get("occurrence_key", "")),
            )

        document_tasks: List[Dict[str, Any]] = []
        for (pmid, group_name), members in sorted(grouped.items()):
            ordered_members = sorted(members, key=rank)
            representative = ordered_members[0]
            normalized_name = str(representative.get("normalized_task_name", "")).strip()
            key_source = f"{pmid}|{group_name}"
            document_task_key = "document_task:" + hashlib.sha1(key_source.encode("utf-8")).hexdigest()[:20]
            active_members = [member for member in members if not bool(member.get("exclude_from_all_analysis"))]
            all_excluded = not active_members
            descriptions = list(
                dict.fromkeys(str(member.get("description", "")).strip() for member in ordered_members if str(member.get("description", "")).strip())
            )
            clues = list(
                dict.fromkeys(str(member.get("clue_sentences", "")).strip() for member in ordered_members if str(member.get("clue_sentences", "")).strip())
            )
            roles = list(
                dict.fromkeys(str(member.get("task_role", "")).strip() for member in ordered_members if str(member.get("task_role", "")).strip())
            )
            source_groups = list(
                dict.fromkeys(str(member.get("source_group", "")).strip() for member in ordered_members if str(member.get("source_group", "")).strip())
            )
            reasons = list(
                dict.fromkeys(
                    str(member.get("analysis_exclusion_reason", "")).strip()
                    for member in ordered_members
                    if str(member.get("analysis_exclusion_reason", "")).strip()
                )
            )
            document_tasks.append(
                {
                    "document_task_key": document_task_key,
                    "pmid": pmid,
                    "normalized_task_name": normalized_name,
                    "final_task_name": str(representative.get("final_task_name", "")).strip(),
                    "representative_occurrence_key": str(representative["occurrence_key"]),
                    "description": str(representative.get("description", "")).strip(),
                    "clue_sentences": str(representative.get("clue_sentences", "")).strip(),
                    "task_role": str(representative.get("task_role", "")).strip(),
                    "is_language_task": str(representative.get("is_language_task", "")).strip(),
                    "is_control_task": str(representative.get("is_control_task", "")).strip(),
                    "linguistic_domain": str(representative.get("linguistic_domain", "")).strip(),
                    "processing_operation": str(representative.get("processing_operation", "")).strip(),
                    "stimulus_modality": str(representative.get("stimulus_modality", "")).strip(),
                    "stimulus_unit": str(representative.get("stimulus_unit", "")).strip(),
                    "response_mode": str(representative.get("response_mode", "")).strip(),
                    "member_occurrence_count": len(ordered_members),
                    "source_group_count": len(source_groups),
                    "source_groups_json": safe_json(source_groups),
                    "member_occurrence_keys_json": safe_json([member["occurrence_key"] for member in ordered_members]),
                    "member_task_indices_json": safe_json([str(member.get("task_index", "")) for member in ordered_members]),
                    "description_variants_json": safe_json(descriptions),
                    "clue_sentence_variants_json": safe_json(clues),
                    "task_roles_json": safe_json(roles),
                    "exclude_from_all_analysis": int(all_excluded),
                    "analysis_exclusion_reason": " | ".join(reasons) if all_excluded else "",
                }
            )
        return document_tasks

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE corpus_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE documents (
                pmid TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                typical_human_total_raw TEXT NOT NULL DEFAULT '',
                patient_total_raw TEXT NOT NULL DEFAULT '',
                method_info_json TEXT NOT NULL,
                content_json_path TEXT NOT NULL,
                meta_json_path TEXT NOT NULL DEFAULT '',
                source_markdown_path TEXT NOT NULL DEFAULT '',
                content_sha256 TEXT NOT NULL,
                section_count INTEGER NOT NULL DEFAULT 0,
                passage_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE sections (
                section_id INTEGER PRIMARY KEY,
                pmid TEXT NOT NULL REFERENCES documents(pmid) ON DELETE CASCADE,
                section_name TEXT NOT NULL,
                section_order INTEGER NOT NULL,
                content TEXT NOT NULL,
                char_count INTEGER NOT NULL,
                UNIQUE(pmid, section_name)
            );

            CREATE TABLE passages (
                passage_id INTEGER PRIMARY KEY,
                pmid TEXT NOT NULL REFERENCES documents(pmid) ON DELETE CASCADE,
                section_name TEXT NOT NULL,
                section_order INTEGER NOT NULL,
                heading_path TEXT NOT NULL DEFAULT '',
                passage_order INTEGER NOT NULL,
                char_start INTEGER NOT NULL,
                char_end INTEGER NOT NULL,
                text TEXT NOT NULL
            );

            CREATE INDEX idx_sections_pmid_name ON sections(pmid, section_name);
            CREATE INDEX idx_passages_pmid_section ON passages(pmid, section_name, passage_order);

            CREATE VIRTUAL TABLE passage_fts USING fts5(
                text,
                heading_path,
                section_name UNINDEXED,
                pmid UNINDEXED,
                content='passages',
                content_rowid='passage_id',
                tokenize='porter unicode61 remove_diacritics 2'
            );

            CREATE TABLE task_occurrences (
                occurrence_key TEXT PRIMARY KEY,
                pmid TEXT NOT NULL REFERENCES documents(pmid) ON DELETE CASCADE,
                task_index TEXT NOT NULL DEFAULT '',
                task_id TEXT NOT NULL DEFAULT '',
                source_group TEXT NOT NULL DEFAULT '',
                cluster_id TEXT NOT NULL DEFAULT '',
                base_task_name TEXT NOT NULL DEFAULT '',
                final_task_name TEXT NOT NULL DEFAULT '',
                normalized_task_name TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                clue_sentences TEXT NOT NULL DEFAULT '',
                normalized_task_description TEXT NOT NULL DEFAULT '',
                task_existence TEXT NOT NULL DEFAULT '',
                task_existence_evidence TEXT NOT NULL DEFAULT '',
                task_naming_rationale TEXT NOT NULL DEFAULT '',
                supported_task_contrast TEXT NOT NULL DEFAULT '',
                annotation_source TEXT NOT NULL DEFAULT '',
                task_role TEXT NOT NULL DEFAULT '',
                is_language_task TEXT NOT NULL DEFAULT '',
                is_control_task TEXT NOT NULL DEFAULT '',
                linguistic_domain TEXT NOT NULL DEFAULT '',
                processing_operation TEXT NOT NULL DEFAULT '',
                stimulus_modality TEXT NOT NULL DEFAULT '',
                stimulus_unit TEXT NOT NULL DEFAULT '',
                response_mode TEXT NOT NULL DEFAULT '',
                membership_decision TEXT NOT NULL DEFAULT '',
                exclude_from_all_analysis INTEGER NOT NULL DEFAULT 0,
                analysis_exclusion_reason TEXT NOT NULL DEFAULT '',
                review_json TEXT NOT NULL DEFAULT '',
                source_json TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX idx_task_pmid ON task_occurrences(pmid);
            CREATE INDEX idx_task_final_name ON task_occurrences(final_task_name);
            CREATE INDEX idx_task_normalized_name ON task_occurrences(normalized_task_name);

            CREATE TABLE document_tasks (
                document_task_key TEXT PRIMARY KEY,
                pmid TEXT NOT NULL REFERENCES documents(pmid) ON DELETE CASCADE,
                normalized_task_name TEXT NOT NULL DEFAULT '',
                final_task_name TEXT NOT NULL DEFAULT '',
                representative_occurrence_key TEXT NOT NULL DEFAULT '',
                description TEXT NOT NULL DEFAULT '',
                clue_sentences TEXT NOT NULL DEFAULT '',
                task_role TEXT NOT NULL DEFAULT '',
                is_language_task TEXT NOT NULL DEFAULT '',
                is_control_task TEXT NOT NULL DEFAULT '',
                linguistic_domain TEXT NOT NULL DEFAULT '',
                processing_operation TEXT NOT NULL DEFAULT '',
                stimulus_modality TEXT NOT NULL DEFAULT '',
                stimulus_unit TEXT NOT NULL DEFAULT '',
                response_mode TEXT NOT NULL DEFAULT '',
                member_occurrence_count INTEGER NOT NULL DEFAULT 1,
                source_group_count INTEGER NOT NULL DEFAULT 0,
                source_groups_json TEXT NOT NULL DEFAULT '[]',
                member_occurrence_keys_json TEXT NOT NULL DEFAULT '[]',
                member_task_indices_json TEXT NOT NULL DEFAULT '[]',
                description_variants_json TEXT NOT NULL DEFAULT '[]',
                clue_sentence_variants_json TEXT NOT NULL DEFAULT '[]',
                task_roles_json TEXT NOT NULL DEFAULT '[]',
                exclude_from_all_analysis INTEGER NOT NULL DEFAULT 0,
                analysis_exclusion_reason TEXT NOT NULL DEFAULT ''
            );
            CREATE INDEX idx_document_task_pmid ON document_tasks(pmid);
            CREATE INDEX idx_document_task_normalized_name ON document_tasks(normalized_task_name);
            CREATE INDEX idx_document_task_active ON document_tasks(exclude_from_all_analysis, normalized_task_name);
            """
        )

    def build(self) -> CorpusBuildResult:
        config = self.config
        config.output_dir.mkdir(parents=True, exist_ok=True)
        if not config.method_info_csv.exists():
            raise FileNotFoundError(f"Method information CSV not found: {config.method_info_csv}")
        if not config.segmented_dir.exists():
            raise FileNotFoundError(f"Segmented content directory not found: {config.segmented_dir}")

        eligible_rows, duplicate_rows = self._eligible_method_rows()
        content_paths, meta_paths = self._segmented_paths()
        missing_content_pmids = sorted(set(eligible_rows) - set(content_paths))
        task_annotations = self._load_task_annotations(set(eligible_rows))

        database_path = config.database_path
        temporary_path = database_path.with_name(database_path.stem + ".building.sqlite3")
        if temporary_path.exists():
            temporary_path.unlink()
        if database_path.exists() and not config.rebuild:
            raise FileExistsError(f"Database already exists and rebuild=False: {database_path}")

        errors: List[Dict[str, str]] = list(duplicate_rows)
        counts: Counter[str] = Counter()
        connection = self._connect(temporary_path)
        try:
            self._create_schema(connection)
            for pmid in sorted(eligible_rows):
                content_path = content_paths.get(pmid)
                if content_path is None:
                    continue
                row = eligible_rows[pmid]
                connection.execute("SAVEPOINT document_index")
                try:
                    content = read_json(content_path)
                    if not isinstance(content, dict):
                        raise ValueError("structured content JSON is not an object")
                    meta_path = meta_paths.get(pmid)
                    meta = read_json(meta_path) if meta_path and meta_path.exists() else {}
                    if not isinstance(meta, dict):
                        meta = {}
                    sections = extract_sections(content)
                    title = normalize_space(content.get("Title", ""))
                    source_markdown = str(meta.get("source_file", "") or "")
                    connection.execute(
                        """
                        INSERT INTO documents(
                            pmid, title, typical_human_total_raw, patient_total_raw,
                            method_info_json, content_json_path, meta_json_path,
                            source_markdown_path, content_sha256
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            pmid,
                            title,
                            str(row.get("typical_human_total", "")),
                            str(row.get("patient_total", "")),
                            safe_json(row),
                            str(content_path),
                            str(meta_path or ""),
                            source_markdown,
                            content_sha256(content_path),
                        ),
                    )
                    passage_count = 0
                    for section_order, (section_name, section_text) in enumerate(sections):
                        connection.execute(
                            """
                            INSERT INTO sections(pmid, section_name, section_order, content, char_count)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (pmid, section_name, section_order, section_text, len(section_text)),
                        )
                        for heading_path, char_start, char_end, passage_text in split_section_to_passages(
                            section_text, config.passage_max_chars
                        ):
                            cursor = connection.execute(
                                """
                                INSERT INTO passages(
                                    pmid, section_name, section_order, heading_path,
                                    passage_order, char_start, char_end, text
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    pmid,
                                    section_name,
                                    section_order,
                                    heading_path,
                                    passage_count,
                                    char_start,
                                    char_end,
                                    passage_text,
                                ),
                            )
                            passage_id = int(cursor.lastrowid)
                            connection.execute(
                                """
                                INSERT INTO passage_fts(rowid, text, heading_path, section_name, pmid)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (passage_id, passage_text, heading_path, section_name, pmid),
                            )
                            passage_count += 1
                    connection.execute(
                        "UPDATE documents SET section_count=?, passage_count=? WHERE pmid=?",
                        (len(sections), passage_count, pmid),
                    )
                    counts["documents"] += 1
                    counts["sections"] += len(sections)
                    counts["passages"] += passage_count
                    connection.execute("RELEASE SAVEPOINT document_index")
                except Exception as exc:
                    connection.execute("ROLLBACK TO SAVEPOINT document_index")
                    connection.execute("RELEASE SAVEPOINT document_index")
                    errors.append({"PMID": pmid, "reason": f"content indexing failed: {type(exc).__name__}: {exc}"})

            indexed_pmids = {str(item[0]) for item in connection.execute("SELECT pmid FROM documents")}
            for annotation in task_annotations:
                if annotation["pmid"] not in indexed_pmids:
                    continue
                connection.execute(
                    """
                    INSERT OR REPLACE INTO task_occurrences(
                        occurrence_key, pmid, task_index, task_id, source_group, cluster_id,
                        base_task_name, final_task_name, normalized_task_name, description, clue_sentences,
                        normalized_task_description, task_existence, task_existence_evidence,
                        task_naming_rationale, supported_task_contrast, annotation_source,
                        task_role, is_language_task, is_control_task, linguistic_domain,
                        processing_operation, stimulus_modality, stimulus_unit, response_mode,
                        membership_decision, exclude_from_all_analysis, analysis_exclusion_reason, review_json, source_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        annotation["occurrence_key"],
                        annotation["pmid"],
                        annotation["task_index"],
                        annotation["task_id"],
                        annotation["source_group"],
                        annotation["cluster_id"],
                        annotation["base_task_name"],
                        annotation["final_task_name"],
                        annotation["normalized_task_name"],
                        annotation["description"],
                        annotation["clue_sentences"],
                        annotation["normalized_task_description"],
                        annotation["task_existence"],
                        annotation["task_existence_evidence"],
                        annotation["task_naming_rationale"],
                        annotation["supported_task_contrast"],
                        annotation["annotation_source"],
                        annotation["task_role"],
                        annotation["is_language_task"],
                        annotation["is_control_task"],
                        annotation["linguistic_domain"],
                        annotation["processing_operation"],
                        annotation["stimulus_modality"],
                        annotation["stimulus_unit"],
                        annotation["response_mode"],
                        annotation["membership_decision"],
                        int(annotation["exclude_from_all_analysis"]),
                        annotation["analysis_exclusion_reason"],
                        annotation["review_json"],
                        annotation["source_json"],
                    ),
                )
                counts["tasks"] += 1

            document_tasks = self._aggregate_document_tasks(
                [annotation for annotation in task_annotations if annotation["pmid"] in indexed_pmids]
            )
            for task in document_tasks:
                connection.execute(
                    """
                    INSERT INTO document_tasks(
                        document_task_key, pmid, normalized_task_name, final_task_name,
                        representative_occurrence_key, description, clue_sentences, task_role,
                        is_language_task, is_control_task, linguistic_domain, processing_operation,
                        stimulus_modality, stimulus_unit, response_mode, member_occurrence_count,
                        source_group_count, source_groups_json, member_occurrence_keys_json,
                        member_task_indices_json, description_variants_json, clue_sentence_variants_json,
                        task_roles_json, exclude_from_all_analysis, analysis_exclusion_reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        task["document_task_key"],
                        task["pmid"],
                        task["normalized_task_name"],
                        task["final_task_name"],
                        task["representative_occurrence_key"],
                        task["description"],
                        task["clue_sentences"],
                        task["task_role"],
                        task["is_language_task"],
                        task["is_control_task"],
                        task["linguistic_domain"],
                        task["processing_operation"],
                        task["stimulus_modality"],
                        task["stimulus_unit"],
                        task["response_mode"],
                        task["member_occurrence_count"],
                        task["source_group_count"],
                        task["source_groups_json"],
                        task["member_occurrence_keys_json"],
                        task["member_task_indices_json"],
                        task["description_variants_json"],
                        task["clue_sentence_variants_json"],
                        task["task_roles_json"],
                        task["exclude_from_all_analysis"],
                        task["analysis_exclusion_reason"],
                    ),
                )
                counts["document_tasks"] += 1

            metadata = {
                "schema_version": SCHEMA_VERSION,
                "built_at_utc": utc_now(),
                "method_info_csv": str(config.method_info_csv),
                "segmented_dir": str(config.segmented_dir),
                "task_occurrences_csv": str(config.task_occurrences_csv or ""),
                "task_review_csv": str(config.task_review_csv or ""),
                "task_exclusion_overrides_csv": str(config.task_exclusion_overrides_csv or ""),
                "eligible_pmids": len(eligible_rows),
                "missing_content_pmids": len(missing_content_pmids),
                "indexed_documents": counts["documents"],
                "indexed_sections": counts["sections"],
                "indexed_passages": counts["passages"],
                "indexed_tasks": counts["tasks"],
                "indexed_document_tasks": counts["document_tasks"],
            }
            connection.executemany(
                "INSERT INTO corpus_metadata(key, value) VALUES (?, ?)",
                [(key, safe_json(value)) for key, value in metadata.items()],
            )
            connection.commit()
            connection.execute("VACUUM")
        finally:
            connection.close()

        temporary_path.replace(database_path)
        audit_dir = config.output_dir / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        (audit_dir / "eligible_pmids.json").write_text(
            json.dumps(sorted(eligible_rows), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (audit_dir / "missing_content_pmids.json").write_text(
            json.dumps(missing_content_pmids, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (audit_dir / "build_issues.csv").open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=["PMID", "reason"], delimiter=";")
            writer.writeheader()
            writer.writerows(errors)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "build_config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
            "built_at_utc": utc_now(),
            "eligible_pmids": len(eligible_rows),
            "indexed_documents": counts["documents"],
            "indexed_sections": counts["sections"],
            "indexed_passages": counts["passages"],
            "indexed_tasks": counts["tasks"],
            "indexed_document_tasks": counts["document_tasks"],
            "missing_content_pmids": missing_content_pmids,
            "issue_count": len(errors),
            "database_sha256": content_sha256(database_path),
        }
        manifest_path = config.output_dir / "corpus_manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return CorpusBuildResult(
            database_path=database_path,
            manifest_path=manifest_path,
            eligible_pmids=len(eligible_rows),
            indexed_documents=counts["documents"],
            indexed_sections=counts["sections"],
            indexed_passages=counts["passages"],
            indexed_tasks=counts["tasks"],
            indexed_document_tasks=counts["document_tasks"],
            missing_content_pmids=missing_content_pmids,
            errors=errors,
        )


class StructuredCorpus:
    """Read/query interface for a database produced by :class:`StructuredCorpusBuilder`."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        if not self.database_path.exists():
            raise FileNotFoundError(f"Corpus database not found: {self.database_path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _dict(row: Optional[sqlite3.Row]) -> Optional[Dict[str, Any]]:
        return dict(row) if row is not None else None

    @staticmethod
    def _parse_document(record: Dict[str, Any]) -> Dict[str, Any]:
        if record.get("method_info_json"):
            record["method_info"] = json.loads(record["method_info_json"])
        return record

    def metadata(self) -> Dict[str, Any]:
        with self._connect() as connection:
            rows = connection.execute("SELECT key, value FROM corpus_metadata ORDER BY key").fetchall()
        result: Dict[str, Any] = {}
        for row in rows:
            try:
                result[row["key"]] = json.loads(row["value"])
            except json.JSONDecodeError:
                result[row["key"]] = row["value"]
        return result

    def get_document(self, pmid: str, include_sections: bool = False) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM documents WHERE pmid=?", (str(pmid),)).fetchone()
            record = self._dict(row)
            if record is None:
                return None
            record = self._parse_document(record)
            if include_sections:
                section_rows = connection.execute(
                    "SELECT section_name, content FROM sections WHERE pmid=? ORDER BY section_order", (str(pmid),)
                ).fetchall()
                record["sections"] = {item["section_name"]: item["content"] for item in section_rows}
        return record

    def get_section(self, pmid: str, section_name: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM sections WHERE pmid=? AND lower(section_name)=lower(?)", (str(pmid), section_name)
            ).fetchone()
        return self._dict(row)

    def get_passage(self, passage_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM passages WHERE passage_id=?", (int(passage_id),)).fetchone()
        return self._dict(row)

    def search(
        self,
        fts_query: str,
        section_name: Optional[str] = None,
        pmid: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        clauses = ["passage_fts MATCH ?"]
        parameters: List[Any] = [fts_query]
        if section_name:
            clauses.append("p.section_name = ?")
            parameters.append(section_name)
        if pmid:
            clauses.append("p.pmid = ?")
            parameters.append(str(pmid))
        parameters.append(max(1, int(limit)))
        sql = f"""
            SELECT p.*, bm25(passage_fts, 1.0, 0.25, 0.0, 0.0) AS bm25_score,
                   snippet(passage_fts, 0, '[', ']', '...', 24) AS snippet
            FROM passage_fts
            JOIN passages AS p ON p.passage_id = passage_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY bm25_score, p.pmid, p.section_order, p.passage_order
            LIMIT ?
        """
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def search_phrase(
        self, phrase: str, section_name: Optional[str] = None, pmid: Optional[str] = None, limit: int = 20
    ) -> List[Dict[str, Any]]:
        return self.search(fts_phrase(phrase), section_name=section_name, pmid=pmid, limit=limit)

    def find_task_occurrences(
        self,
        task_name: str,
        exact: bool = False,
        include_excluded: bool = False,
        limit: int = 100,
        deduplicated: bool = True,
    ) -> List[Dict[str, Any]]:
        """Find task records, using one logical task per PMID by default.

        Set ``deduplicated=False`` only when group-specific provenance rows are
        explicitly needed from ``task_occurrences``.
        """

        normalized_name = normalize_task_name(task_name)
        if not normalized_name:
            raise ValueError("task_name cannot be empty")
        comparator = "= ?" if exact else "LIKE ?"
        value = normalized_name if exact else f"%{normalized_name}%"
        clauses = [f"normalized_task_name {comparator}"]
        parameters: List[Any] = [value]
        if not include_excluded:
            clauses.append("exclude_from_all_analysis=0")
        parameters.append(max(1, int(limit)))
        table_name = "document_tasks" if deduplicated else "task_occurrences"
        order_clause = "pmid, final_task_name, document_task_key" if deduplicated else "pmid, task_index, occurrence_key"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM {table_name} WHERE {' AND '.join(clauses)} ORDER BY {order_clause} LIMIT ?",
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _sentences(text: str) -> Iterator[Tuple[int, int, str]]:
        cursor = 0
        for match in SENTENCE_RE.finditer(text):
            raw = text[cursor : match.start()]
            sentence = raw.strip()
            if sentence:
                begin = cursor + raw.find(sentence)
                yield begin, begin + len(sentence), sentence
            cursor = match.end()
        raw = text[cursor:]
        sentence = raw.strip()
        if sentence:
            begin = cursor + raw.find(sentence)
            yield begin, begin + len(sentence), sentence

    def locate_clue_sentence(self, pmid: str, clue_sentence: str) -> Optional[Dict[str, Any]]:
        clue_normalized = normalize_match_text(clue_sentence)
        if not clue_normalized:
            raise ValueError("clue_sentence cannot be empty")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM passages WHERE pmid=? ORDER BY section_order, passage_order", (str(pmid),)
            ).fetchall()
        best: Optional[Dict[str, Any]] = None
        best_score = -1.0
        for row in rows:
            record = dict(row)
            text = record["text"]
            normalized_text = normalize_match_text(text)
            exact_offset = normalized_text.find(clue_normalized)
            if exact_offset >= 0:
                record.update({"match_type": "normalized_exact", "similarity": 1.0, "matched_sentence": clue_sentence})
                return record
            for sentence_start, sentence_end, sentence in self._sentences(text):
                score = SequenceMatcher(None, clue_normalized, normalize_match_text(sentence)).ratio()
                if score > best_score:
                    best_score = score
                    best = record.copy()
                    best.update(
                        {
                            "match_type": "nearest_sentence",
                            "similarity": round(score, 6),
                            "matched_sentence": sentence,
                            "sentence_char_start": record["char_start"] + sentence_start,
                            "sentence_char_end": record["char_start"] + sentence_end,
                        }
                    )
        return best

    def _candidate_passages(self, term: str, section_name: Optional[str] = None) -> List[Dict[str, Any]]:
        try:
            return self.search_phrase(term, section_name=section_name, limit=1_000_000)
        except sqlite3.OperationalError:
            # Fall back to a regular SQL scan for unusual text that FTS cannot parse.
            clauses = []
            parameters: List[Any] = []
            if section_name:
                clauses.append("section_name=?")
                parameters.append(section_name)
            sql = "SELECT * FROM passages"
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)
            with self._connect() as connection:
                return [dict(row) for row in connection.execute(sql, parameters).fetchall()]

    def _unit_texts(self, unit: str, section_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        if unit not in {"document", "section", "passage"}:
            raise ValueError("unit must be one of: document, section, passage")
        with self._connect() as connection:
            if unit == "passage":
                clauses = ["1=1"]
                parameters: List[Any] = []
                if section_name:
                    clauses.append("section_name=?")
                    parameters.append(section_name)
                rows = connection.execute(
                    f"SELECT passage_id, pmid, section_name, text FROM passages WHERE {' AND '.join(clauses)}", parameters
                ).fetchall()
                return {
                    f"passage:{row['passage_id']}": {
                        "unit_id": f"passage:{row['passage_id']}",
                        "pmid": row["pmid"],
                        "section_name": row["section_name"],
                        "text": row["text"],
                    }
                    for row in rows
                }
            if unit == "section":
                clauses = ["1=1"]
                parameters = []
                if section_name:
                    clauses.append("section_name=?")
                    parameters.append(section_name)
                rows = connection.execute(
                    f"SELECT pmid, section_name, content FROM sections WHERE {' AND '.join(clauses)}", parameters
                ).fetchall()
                return {
                    f"section:{row['pmid']}:{row['section_name']}": {
                        "unit_id": f"section:{row['pmid']}:{row['section_name']}",
                        "pmid": row["pmid"],
                        "section_name": row["section_name"],
                        "text": row["content"],
                    }
                    for row in rows
                }
            clauses = ["1=1"]
            parameters = []
            if section_name:
                clauses.append("s.section_name=?")
                parameters.append(section_name)
            rows = connection.execute(
                f"""
                SELECT d.pmid, GROUP_CONCAT(s.content, '\n\n') AS content
                FROM documents AS d
                JOIN sections AS s ON s.pmid=d.pmid
                WHERE {' AND '.join(clauses)}
                GROUP BY d.pmid
                """,
                parameters,
            ).fetchall()
            return {
                f"document:{row['pmid']}": {
                    "unit_id": f"document:{row['pmid']}",
                    "pmid": row["pmid"],
                    "section_name": section_name or "all",
                    "text": row["content"],
                }
                for row in rows
            }

    def _matching_units(self, term: str, unit: str, section_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        pattern = phrase_pattern(term)
        units = self._unit_texts(unit, section_name=section_name)
        return {key: value for key, value in units.items() if pattern.search(value["text"])}

    def term_tfidf(
        self, term: str, unit: str = "document", section_name: Optional[str] = None, limit: int = 100
    ) -> Dict[str, Any]:
        """Compute conventional TF-IDF with smooth IDF over corpus units."""

        all_units = self._unit_texts(unit, section_name=section_name)
        pattern = phrase_pattern(term)
        matches: List[Dict[str, Any]] = []
        for value in all_units.values():
            term_count = len(pattern.findall(value["text"]))
            if not term_count:
                continue
            token_count = max(1, len(WORD_RE.findall(value["text"])))
            matches.append(
                {
                    "unit_id": value["unit_id"],
                    "PMID": value["pmid"],
                    "section_name": value["section_name"],
                    "term_count": term_count,
                    "token_count": token_count,
                    "tf": term_count / token_count,
                }
            )
        total_units = len(all_units)
        document_frequency = len(matches)
        idf = math.log((total_units + 1) / (document_frequency + 1)) + 1 if total_units else 0.0
        for row in matches:
            row["idf"] = idf
            row["tfidf"] = row["tf"] * idf
        matches.sort(key=lambda item: (-item["tfidf"], item["PMID"], item["unit_id"]))
        return {
            "term": term,
            "unit": unit,
            "section_name": section_name or "all",
            "unit_count": total_units,
            "document_frequency": document_frequency,
            "idf": idf,
            "results": matches[: max(1, int(limit))],
        }

    def ppmi_terms(
        self, term_a: str, term_b: str, unit: str = "document", section_name: Optional[str] = None, evidence_limit: int = 100
    ) -> Dict[str, Any]:
        """Compute positive PMI from binary co-occurrence events over corpus units."""

        all_units = self._unit_texts(unit, section_name=section_name)
        pattern_a = phrase_pattern(term_a)
        pattern_b = phrase_pattern(term_b)
        keys_a = {key for key, value in all_units.items() if pattern_a.search(value["text"])}
        keys_b = {key for key, value in all_units.items() if pattern_b.search(value["text"])}
        shared = sorted(keys_a & keys_b)
        n = len(all_units)
        n_a, n_b, n_ab = len(keys_a), len(keys_b), len(shared)
        pmi = math.log2((n_ab * n) / (n_a * n_b)) if n and n_a and n_b and n_ab else float("-inf")
        evidence = [
            {
                "unit_id": key,
                "PMID": all_units[key]["pmid"],
                "section_name": all_units[key]["section_name"],
            }
            for key in shared[: max(1, int(evidence_limit))]
        ]
        return {
            "term_a": term_a,
            "term_b": term_b,
            "unit": unit,
            "section_name": section_name or "all",
            "unit_count": n,
            "term_a_unit_count": n_a,
            "term_b_unit_count": n_b,
            "cooccurrence_unit_count": n_ab,
            "pmi": pmi if math.isfinite(pmi) else None,
            "ppmi": max(0.0, pmi) if math.isfinite(pmi) else 0.0,
            "evidence": evidence,
        }

    def ppmi_task_term(self, task_name: str, term: str, section_name: Optional[str] = None, evidence_limit: int = 100) -> Dict[str, Any]:
        """Compute document-level PPMI for an annotated task and a text term."""

        all_units = self._unit_texts("document", section_name=section_name)
        with self._connect() as connection:
            task_rows = connection.execute(
                """
                SELECT DISTINCT pmid FROM document_tasks
                WHERE normalized_task_name LIKE ? AND exclude_from_all_analysis=0
                """,
                (f"%{normalize_task_name(task_name)}%",),
            ).fetchall()
        task_pmids = {str(row["pmid"]) for row in task_rows}
        term_pattern = phrase_pattern(term)
        term_keys = {key for key, value in all_units.items() if term_pattern.search(value["text"])}
        task_keys = {key for key, value in all_units.items() if value["pmid"] in task_pmids}
        shared = sorted(task_keys & term_keys)
        n, n_task, n_term, n_shared = len(all_units), len(task_keys), len(term_keys), len(shared)
        pmi = math.log2((n_shared * n) / (n_task * n_term)) if n and n_task and n_term and n_shared else float("-inf")
        return {
            "task_name_query": task_name,
            "term": term,
            "unit": "document",
            "section_name": section_name or "all",
            "document_count": n,
            "task_document_count": n_task,
            "term_document_count": n_term,
            "cooccurrence_document_count": n_shared,
            "pmi": pmi if math.isfinite(pmi) else None,
            "ppmi": max(0.0, pmi) if math.isfinite(pmi) else 0.0,
            "evidence_pmids": [all_units[key]["pmid"] for key in shared[: max(1, int(evidence_limit))]],
        }

    def export_lora_jsonl(
        self,
        output_path: Path,
        mode: str = "passage",
        section_names: Optional[Sequence[str]] = None,
        max_records: Optional[int] = None,
    ) -> int:
        """Export provenance-preserving records for continued pretraining or SFT.

        ``passage`` creates raw text records suitable for continued pretraining/RAG.
        ``method_info_sft`` creates Methods-to-method-info message pairs; labels are
        the existing structured extraction, never fabricated by this exporter.
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        allowed = set(section_names or [])
        count = 0
        with self._connect() as connection, output_path.open("w", encoding="utf-8") as file_obj:
            if mode == "passage":
                rows = connection.execute(
                    """
                    SELECT p.passage_id, p.pmid, p.section_name, p.heading_path, p.char_start, p.char_end, p.text,
                           d.content_json_path
                    FROM passages AS p JOIN documents AS d ON d.pmid=p.pmid
                    ORDER BY p.pmid, p.section_order, p.passage_order
                    """
                )
                for row in rows:
                    if allowed and row["section_name"] not in allowed:
                        continue
                    record = {
                        "id": f"passage:{row['passage_id']}",
                        "text": row["text"],
                        "metadata": {
                            "PMID": row["pmid"],
                            "section_name": row["section_name"],
                            "heading_path": row["heading_path"],
                            "char_start": row["char_start"],
                            "char_end": row["char_end"],
                            "content_json_path": row["content_json_path"],
                        },
                    }
                    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if max_records and count >= max_records:
                        break
            elif mode == "method_info_sft":
                rows = connection.execute(
                    """
                    SELECT d.pmid, d.method_info_json, s.content
                    FROM documents AS d JOIN sections AS s ON s.pmid=d.pmid
                    WHERE lower(s.section_name)='methods'
                    ORDER BY d.pmid
                    """
                )
                for row in rows:
                    record = {
                        "id": f"method-info:{row['pmid']}",
                        "messages": [
                            {"role": "system", "content": "Extract only the requested structured study metadata from the supplied Methods text."},
                            {"role": "user", "content": f"PMID: {row['pmid']}\n\nMETHODS:\n{row['content']}"},
                            {"role": "assistant", "content": row["method_info_json"]},
                        ],
                    }
                    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1
                    if max_records and count >= max_records:
                        break
            else:
                raise ValueError("mode must be 'passage' or 'method_info_sft'")
        return count
