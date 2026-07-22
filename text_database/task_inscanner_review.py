"""Task-to-source-context retrieval and manual in-scanner evidence review.

The module keeps the review unit at ``PMID + normalized_task_name``.  It does
not assume that a paper-level fMRI participation flag applies to every task:
reviewers see the matched source text and decide whether that text is adequate
to determine the task's in-scanner status.
"""

from __future__ import annotations

import csv
import json
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SECTION_PRIORITY = {
    "Methods": 0,
    "Results": 1,
    "Other": 2,
    "Introduction": 3,
    "Abstract": 4,
    "Discussion": 5,
    "Conclusion": 6,
}
SKIPPED_SECTIONS = {"References", "Author", "Acknowledgements", "Tables", "Title", "Keywords"}
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['-][A-Za-z0-9]+)?")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
ELLIPSIS_RE = re.compile(r"(?:\.{3,}|…)+")


def utc_now() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


def normalize_tokens(value: Any) -> List[str]:
    return [token.casefold() for token in TOKEN_RE.findall(str(value or ""))]


def unique_texts(values: Iterable[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = normalize_space(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def split_clue_fragments(value: Any) -> List[str]:
    """Return sufficiently specific literal clue fragments without inventing text."""

    raw = normalize_space(value).strip(" \"'")
    if not raw:
        return []
    fragments = [raw]
    fragments.extend(part.strip(" \"'") for part in ELLIPSIS_RE.split(raw))
    fragments.extend(part.strip(" \"'") for part in re.split(r"(?<=[.!?;:])\s+", raw))
    return [part for part in unique_texts(fragments) if len(normalize_tokens(part)) >= 4 and len(part) >= 20]


def task_name_variants(value: Any) -> List[str]:
    name = normalize_space(value)
    if not name:
        return []
    variants = [name]
    no_task = re.sub(r"\s+task\s*$", "", name, flags=re.IGNORECASE).strip()
    if no_task and no_task.casefold() != name.casefold():
        variants.append(no_task)
    return [part for part in unique_texts(variants) if len(normalize_tokens(part)) >= 2]


def token_sequence_span(text: str, phrase: str) -> Optional[Tuple[int, int]]:
    """Locate a phrase by contiguous lexical tokens while retaining source offsets."""

    phrase_tokens = normalize_tokens(phrase)
    if len(phrase_tokens) < 2:
        return None
    text_tokens = [(match.group(0).casefold(), match.start(), match.end()) for match in TOKEN_RE.finditer(text)]
    target_length = len(phrase_tokens)
    for start_index in range(0, len(text_tokens) - target_length + 1):
        if [item[0] for item in text_tokens[start_index : start_index + target_length]] == phrase_tokens:
            return text_tokens[start_index][1], text_tokens[start_index + target_length - 1][2]
    return None


def sentence_spans(text: str) -> Iterable[Tuple[int, int, str]]:
    cursor = 0
    for match in SENTENCE_BOUNDARY_RE.finditer(text):
        raw = text[cursor : match.start()]
        sentence = raw.strip()
        if sentence:
            start = cursor + raw.find(sentence)
            yield start, start + len(sentence), sentence
        cursor = match.end()
    raw = text[cursor:]
    sentence = raw.strip()
    if sentence:
        start = cursor + raw.find(sentence)
        yield start, start + len(sentence), sentence


def context_window(text: str, start: int, end: int, radius: int) -> Tuple[str, int, int]:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    context = prefix + text[left:right] + suffix
    context_start = len(prefix) + start - left
    context_end = context_start + (end - start)
    return context, context_start, context_end


@dataclass
class TaskInScannerReviewConfig:
    database_path: Path
    output_dir: Path
    article_sample_size: int = 20
    random_seed: int = 20260721
    context_radius: int = 500
    include_excluded: bool = False


class TaskContextMatcher:
    """Retrieve source evidence for every de-duplicated task in an article."""

    def __init__(self, database_path: Path, context_radius: int = 500) -> None:
        self.database_path = Path(database_path)
        self.context_radius = max(100, int(context_radius))
        if not self.database_path.exists():
            raise FileNotFoundError(f"SQLite database not found: {self.database_path}")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database_path))
        connection.row_factory = sqlite3.Row
        return connection

    def available_pmids(self, include_excluded: bool = False) -> List[str]:
        where = "" if include_excluded else "AND exclude_from_all_analysis=0"
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT DISTINCT pmid
                FROM task_occurrences
                WHERE normalized_task_name<>'' {where}
                ORDER BY pmid
                """
            ).fetchall()
        return [str(row["pmid"]) for row in rows]

    def _document_metadata(self, connection: sqlite3.Connection, pmid: str) -> Dict[str, Any]:
        row = connection.execute("SELECT title, method_info_json FROM documents WHERE pmid=?", (pmid,)).fetchone()
        if row is None:
            return {"title": "", "method_info": {}}
        try:
            method_info = json.loads(row["method_info_json"] or "{}")
        except json.JSONDecodeError:
            method_info = {}
        return {"title": str(row["title"] or ""), "method_info": method_info}

    def _task_groups(self, connection: sqlite3.Connection, pmid: str, include_excluded: bool) -> List[Dict[str, Any]]:
        where = "" if include_excluded else "AND exclude_from_all_analysis=0"
        rows = connection.execute(
            f"""
            SELECT occurrence_key, task_index, task_id, source_group, base_task_name,
                   final_task_name, normalized_task_name, description, clue_sentences,
                   normalized_task_description, task_existence, task_existence_evidence,
                   task_role, exclude_from_all_analysis
            FROM task_occurrences
            WHERE pmid=? AND normalized_task_name<>'' {where}
            ORDER BY normalized_task_name, task_index, occurrence_key
            """,
            (pmid,),
        ).fetchall()
        grouped: Dict[str, List[sqlite3.Row]] = defaultdict(list)
        for row in rows:
            grouped[str(row["normalized_task_name"])].append(row)
        groups: List[Dict[str, Any]] = []
        for normalized_name, members in grouped.items():
            original_names = unique_texts(
                [member["final_task_name"] for member in members] + [member["base_task_name"] for member in members]
            )
            clues = unique_texts(
                [member["clue_sentences"] for member in members] + [member["task_existence_evidence"] for member in members]
            )
            descriptions = unique_texts(
                [member["description"] for member in members] + [member["normalized_task_description"] for member in members]
            )
            groups.append(
                {
                    "task_review_key": f"{pmid}|{normalized_name}",
                    "PMID": pmid,
                    "normalized_task_name": normalized_name,
                    "display_task_name": original_names[0] if original_names else normalized_name,
                    "original_task_names": original_names,
                    "clue_sentences": clues,
                    "descriptions": descriptions,
                    "source_groups": sorted({str(member["source_group"] or "") for member in members if member["source_group"]}),
                    "task_indices": sorted({str(member["task_index"] or "") for member in members if member["task_index"]}),
                    "occurrence_keys": [str(member["occurrence_key"]) for member in members],
                    "occurrence_count": len(members),
                    "task_roles": sorted({str(member["task_role"] or "") for member in members if member["task_role"]}),
                    "task_existence": sorted({str(member["task_existence"] or "") for member in members if member["task_existence"]}),
                }
            )
        return groups

    @staticmethod
    def _sections(connection: sqlite3.Connection, pmid: str) -> List[sqlite3.Row]:
        rows = connection.execute(
            "SELECT section_name, section_order, content FROM sections WHERE pmid=? ORDER BY section_order", (pmid,)
        ).fetchall()
        return [row for row in rows if str(row["section_name"]) not in SKIPPED_SECTIONS and str(row["content"] or "").strip()]

    @staticmethod
    def _ordered_sections(rows: Sequence[sqlite3.Row]) -> List[sqlite3.Row]:
        return sorted(rows, key=lambda row: (SECTION_PRIORITY.get(str(row["section_name"]), 99), int(row["section_order"])))

    def _candidate_phrases(self, task: Dict[str, Any]) -> List[Tuple[str, str]]:
        candidates: List[Tuple[str, str]] = []
        for clue in task["clue_sentences"]:
            candidates.extend(("clue_sentence", fragment) for fragment in split_clue_fragments(clue))
        for name in task["original_task_names"] + [task["normalized_task_name"]]:
            candidates.extend(("task_name", variant) for variant in task_name_variants(name))
        seen = set()
        ordered: List[Tuple[str, str]] = []
        for source, phrase in candidates:
            key = (source, phrase.casefold())
            if key not in seen:
                seen.add(key)
                ordered.append((source, phrase))
        return ordered

    def _locate_context(self, task: Dict[str, Any], sections: Sequence[sqlite3.Row]) -> Dict[str, Any]:
        candidates = self._candidate_phrases(task)
        ordered_sections = self._ordered_sections(sections)
        best_exact: Optional[Dict[str, Any]] = None
        for candidate_order, (source, phrase) in enumerate(candidates):
            for section_order, section in enumerate(ordered_sections):
                text = str(section["content"])
                span = token_sequence_span(text, phrase)
                if span is None:
                    continue
                rank = (0 if source == "clue_sentence" else 1, section_order, candidate_order, -len(normalize_tokens(phrase)))
                record = {
                    "match_method": f"exact_{source}",
                    "match_query": phrase,
                    "match_section": str(section["section_name"]),
                    "match_score": 1.0,
                    "match_start": span[0],
                    "match_end": span[1],
                    "_rank": rank,
                    "_text": text,
                }
                if best_exact is None or rank < best_exact["_rank"]:
                    best_exact = record
        if best_exact is not None:
            context, context_start, context_end = context_window(
                best_exact.pop("_text"), best_exact["match_start"], best_exact["match_end"], self.context_radius
            )
            best_exact.pop("_rank", None)
            best_exact.update(
                {
                    "context": context,
                    "context_match_start": context_start,
                    "context_match_end": context_end,
                }
            )
            return best_exact

        # A cautious fallback: only surface a high-similarity clue sentence and label it approximate.
        best_approximate: Optional[Dict[str, Any]] = None
        clue_candidates = [(source, phrase) for source, phrase in candidates if source == "clue_sentence"]
        for candidate_order, (_source, phrase) in enumerate(clue_candidates):
            normalized_phrase = " ".join(normalize_tokens(phrase))
            if len(normalized_phrase) < 24:
                continue
            for section_order, section in enumerate(ordered_sections):
                text = str(section["content"])
                for start, end, sentence in sentence_spans(text):
                    normalized_sentence = " ".join(normalize_tokens(sentence))
                    if len(normalized_sentence) < 20:
                        continue
                    similarity = SequenceMatcher(None, normalized_phrase, normalized_sentence).ratio()
                    if similarity < 0.70:
                        continue
                    rank = (-similarity, section_order, candidate_order)
                    candidate = {
                        "match_method": "approximate_clue_sentence",
                        "match_query": phrase,
                        "match_section": str(section["section_name"]),
                        "match_score": round(float(similarity), 4),
                        "match_start": start,
                        "match_end": end,
                        "_rank": rank,
                        "_text": text,
                    }
                    if best_approximate is None or rank < best_approximate["_rank"]:
                        best_approximate = candidate
        if best_approximate is not None:
            context, context_start, context_end = context_window(
                best_approximate.pop("_text"),
                best_approximate["match_start"],
                best_approximate["match_end"],
                self.context_radius,
            )
            best_approximate.pop("_rank", None)
            best_approximate.update(
                {
                    "context": context,
                    "context_match_start": context_start,
                    "context_match_end": context_end,
                }
            )
            return best_approximate

        return {
            "match_method": "unmatched",
            "match_query": "",
            "match_section": "",
            "match_score": 0.0,
            "match_start": None,
            "match_end": None,
            "context": "",
            "context_match_start": None,
            "context_match_end": None,
        }

    def collect_article(self, pmid: str, include_excluded: bool = False) -> Dict[str, Any]:
        pmid = str(pmid).strip()
        with self._connect() as connection:
            metadata = self._document_metadata(connection, pmid)
            sections = self._sections(connection, pmid)
            tasks = self._task_groups(connection, pmid, include_excluded)
        for task in tasks:
            task["source_match"] = self._locate_context(task, sections)
        method_info = metadata["method_info"]
        return {
            "PMID": pmid,
            "title": metadata["title"],
            "paper_level_fmri_metadata": {
                "typical_human_all_participated_in_fMRI_task": method_info.get(
                    "typical_human_all_participated_in_fMRI_task", "unknown"
                ),
                "patient_all_participated_in_fMRI_task": method_info.get(
                    "patient_all_participated_in_fMRI_task", "unknown"
                ),
                "scanner_type": method_info.get("scanner_type", ""),
            },
            "tasks": tasks,
        }


def build_review_sample(config: TaskInScannerReviewConfig, force: bool = False) -> Dict[str, Any]:
    """Create a fixed article-level review sample and its source-match audit."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = config.output_dir / "task_inscanner_context_review_sample.json"
    if sample_path.exists() and not force:
        with sample_path.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    matcher = TaskContextMatcher(config.database_path, context_radius=config.context_radius)
    pmids = matcher.available_pmids(include_excluded=config.include_excluded)
    if not pmids:
        raise RuntimeError("No indexed task occurrences are available for review.")
    chosen_count = min(max(1, int(config.article_sample_size)), len(pmids))
    selected_pmids = sorted(random.Random(config.random_seed).sample(pmids, chosen_count))
    articles = [matcher.collect_article(pmid, include_excluded=config.include_excluded) for pmid in selected_pmids]
    payload = {
        "generated_at": utc_now(),
        "database_path": str(config.database_path),
        "article_sample_size": chosen_count,
        "random_seed": config.random_seed,
        "context_radius_each_side": config.context_radius,
        "include_excluded_task_occurrences": config.include_excluded,
        "articles": articles,
    }
    sample_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_rows: List[Dict[str, Any]] = []
    for article in articles:
        for task in article["tasks"]:
            match = task["source_match"]
            audit_rows.append(
                {
                    "PMID": article["PMID"],
                    "title": article["title"],
                    "normalized_task_name": task["normalized_task_name"],
                    "display_task_name": task["display_task_name"],
                    "source_groups": "|".join(task["source_groups"]),
                    "task_indices": "|".join(task["task_indices"]),
                    "occurrence_count_merged": task["occurrence_count"],
                    "match_method": match["match_method"],
                    "match_section": match["match_section"],
                    "match_score": match["match_score"],
                    "match_query": match["match_query"],
                    "context": match["context"],
                }
            )
    audit_path = config.output_dir / "task_inscanner_context_match_audit.csv"
    fields = list(audit_rows[0]) if audit_rows else ["PMID", "normalized_task_name", "match_method"]
    with audit_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(audit_rows)
    (config.output_dir / "task_inscanner_context_review_summary.json").write_text(
        json.dumps(
            {
                "article_count": len(articles),
                "deduplicated_article_task_count": len(audit_rows),
                "exact_clue_matches": sum(row["match_method"] == "exact_clue_sentence" for row in audit_rows),
                "exact_task_name_matches": sum(row["match_method"] == "exact_task_name" for row in audit_rows),
                "approximate_clue_matches": sum(row["match_method"] == "approximate_clue_sentence" for row in audit_rows),
                "unmatched": sum(row["match_method"] == "unmatched" for row in audit_rows),
                "sample_path": str(sample_path),
                "audit_path": str(audit_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


class TaskInScannerContextReviewUI:
    """A paged article-level Tkinter review UI with autosave and provenance."""

    SCORE_LABELS = {
        "": "Not scored",
        "0": "0 - no task-specific source context",
        "1": "1 - task context found, but insufficient for scanner judgment",
        "2": "2 - enough evidence for a scanner judgment",
        "3": "3 - explicit in-scanner / out-of-scanner evidence",
    }
    STATUS_LABELS = {
        "": "Not decided",
        "in_scanner": "In-scanner",
        "not_in_scanner": "Not in-scanner",
        "unclear": "Unclear from available text",
    }

    def __init__(self, sample_path: Path, output_dir: Path, title: str = "Task In-Scanner Context Review") -> None:
        self.sample_path = Path(sample_path)
        self.output_dir = Path(output_dir)
        self.title = title
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with self.sample_path.open("r", encoding="utf-8") as file_obj:
            self.payload = json.load(file_obj)
        self.articles: List[Dict[str, Any]] = self.payload.get("articles", [])
        if not self.articles:
            raise RuntimeError(f"No articles found in review sample: {self.sample_path}")
        self.autosave_path = self.output_dir / "task_inscanner_context_review_autosave.json"
        self.final_json_path = self.output_dir / "task_inscanner_context_review_results.json"
        self.final_csv_path = self.output_dir / "task_inscanner_context_review_results.csv"
        self.reviews = self._load_reviews()
        self.article_index = self._first_unscored_article()
        self.root = None
        self.tk = None
        self.ttk = None
        self.canvas = None
        self.scroll_frame = None
        self.canvas_window = None
        self.status_var = None
        self.autosave_var = None
        self.jump_var = None
        self.score_vars: Dict[str, Any] = {}
        self.status_vars: Dict[str, Any] = {}
        self.note_widgets: Dict[str, Any] = {}
        self.autosave_after_id = None
        self.rendering = False

    def _load_reviews(self) -> Dict[str, Dict[str, Any]]:
        candidates = [path for path in (self.final_json_path, self.autosave_path) if path.exists()]
        if not candidates:
            return {}
        newest = max(candidates, key=lambda path: path.stat().st_mtime)
        try:
            payload = json.loads(newest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return {
            str(item.get("task_review_key", "")): item
            for item in payload.get("task_reviews", [])
            if item.get("task_review_key")
        }

    def _first_unscored_article(self) -> int:
        for index, article in enumerate(self.articles):
            keys = [task["task_review_key"] for task in article["tasks"]]
            if any(not self.reviews.get(key, {}).get("evidence_sufficiency_score", "") for key in keys):
                return index
        return 0

    def run(self) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.root = tk.Tk()
        self.root.title(self.title)
        self.root.geometry("1500x940")
        self.root.minsize(1160, 760)
        self.root.configure(bg="#f4f7fb")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_styles()
        self._build_shell()
        self._render_article()
        self.root.mainloop()

    def _configure_styles(self) -> None:
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#f4f7fb")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Header.TLabel", background="#f4f7fb", foreground="#12233d", font=("Microsoft YaHei", 16, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#263648", font=("Microsoft YaHei", 10))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#667085", font=("Microsoft YaHei", 9))
        style.configure("Metric.TLabel", background="#eaf2fb", foreground="#1d4f7a", font=("Microsoft YaHei", 9, "bold"), padding=(8, 4))
        style.configure("Evidence.TLabel", background="#ffffff", foreground="#1d4f7a", font=("Microsoft YaHei", 10))
        style.configure("Primary.TButton", font=("Microsoft YaHei", 10, "bold"), padding=(10, 6))
        style.configure("TCheckbutton", background="#ffffff", font=("Microsoft YaHei", 10))

    def _build_shell(self) -> None:
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(18, 14, 18, 14))
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer, style="App.TFrame")
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Task in-scanner evidence review", style="Header.TLabel").pack(side="left")
        controls = ttk.Frame(top, style="App.TFrame")
        controls.pack(side="right")
        ttk.Button(controls, text="Previous article", command=self._previous_article).pack(side="left", padx=3)
        ttk.Button(controls, text="Save", command=lambda: self._save(show_message=True), style="Primary.TButton").pack(side="left", padx=3)
        ttk.Button(controls, text="Save + next", command=self._save_and_next, style="Primary.TButton").pack(side="left", padx=3)
        ttk.Button(controls, text="Next article", command=self._next_article).pack(side="left", padx=3)

        nav = ttk.Frame(outer, style="App.TFrame")
        nav.pack(fill="x", pady=(0, 10))
        self.status_var = tk.StringVar(value="")
        self.autosave_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self.status_var, style="Header.TLabel", font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        ttk.Label(nav, textvariable=self.autosave_var, style="Header.TLabel", font=("Microsoft YaHei", 9)).pack(side="left", padx=(16, 0))
        ttk.Label(nav, text="Jump to PMID:", background="#f4f7fb", font=("Microsoft YaHei", 10)).pack(side="left", padx=(22, 6))
        self.jump_var = tk.StringVar()
        selector = ttk.Combobox(nav, textvariable=self.jump_var, width=14, state="readonly")
        selector["values"] = [article["PMID"] for article in self.articles]
        selector.bind("<<ComboboxSelected>>", lambda _event: self._jump_to_article())
        selector.pack(side="left")

        header = ttk.Frame(outer, style="Panel.TFrame", padding=(14, 10, 14, 10))
        header.pack(fill="x", pady=(0, 10))
        self.article_header = header

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg="#f4f7fb", highlightthickness=0)
        scrollbar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scroll_frame = ttk.Frame(self.canvas, style="App.TFrame")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")
        self.scroll_frame.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width))
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _on_mousewheel(self, event: Any) -> None:
        if self.canvas is not None:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    @staticmethod
    def _clear(frame: Any) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _current_article(self) -> Dict[str, Any]:
        return self.articles[self.article_index]

    def _render_article(self) -> None:
        ttk = self.ttk
        self.rendering = True
        article = self._current_article()
        self.score_vars = {}
        self.status_vars = {}
        self.note_widgets = {}
        self._clear(self.article_header)
        self._clear(self.scroll_frame)
        self.jump_var.set(article["PMID"])
        scored_articles = sum(
            all(self.reviews.get(task["task_review_key"], {}).get("evidence_sufficiency_score", "") for task in item["tasks"])
            for item in self.articles
        )
        self.status_var.set(
            f"Article {self.article_index + 1}/{len(self.articles)} | PMID {article['PMID']} | fully scored {scored_articles}/{len(self.articles)}"
        )
        self.autosave_var.set(f"Autosave: {self.autosave_path.name}")
        fmri = article["paper_level_fmri_metadata"]
        ttk.Label(self.article_header, text=f"PMID {article['PMID']}", style="Body.TLabel", font=("Microsoft YaHei", 13, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Label(self.article_header, text=article.get("title", "") or "Title unavailable", style="Muted.TLabel", wraplength=1200).grid(row=1, column=0, sticky="w", pady=(3, 8))
        metrics = ttk.Frame(self.article_header, style="Panel.TFrame")
        metrics.grid(row=2, column=0, sticky="w")
        for label, value in (
            ("Unique tasks", len(article["tasks"])),
            ("Typical human fMRI", fmri.get("typical_human_all_participated_in_fMRI_task", "unknown")),
            ("Patient fMRI", fmri.get("patient_all_participated_in_fMRI_task", "unknown")),
            ("Scanner", fmri.get("scanner_type", "") or "not reported"),
        ):
            ttk.Label(metrics, text=f"{label}: {value}", style="Metric.TLabel").pack(side="left", padx=(0, 6))

        for index, task in enumerate(article["tasks"], start=1):
            self._render_task_card(task, index)
        self.rendering = False
        if self.canvas is not None:
            self.canvas.yview_moveto(0)

    def _render_task_card(self, task: Dict[str, Any], index: int) -> None:
        tk, ttk = self.tk, self.ttk
        card = ttk.Frame(self.scroll_frame, style="Card.TFrame", padding=(14, 12, 14, 12))
        card.pack(fill="x", pady=(0, 12))
        key = task["task_review_key"]
        existing = self.reviews.get(key, {})
        ttk.Label(card, text=f"{index}. {task['display_task_name']}", style="Body.TLabel", font=("Microsoft YaHei", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        provenance = (
            f"Normalized key: {task['normalized_task_name']} | source groups: {', '.join(task['source_groups']) or 'unknown'} | "
            f"task index: {', '.join(task['task_indices']) or 'unknown'} | merged occurrences: {task['occurrence_count']}"
        )
        ttk.Label(card, text=provenance, style="Muted.TLabel", wraplength=1380).grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 6))
        if task["clue_sentences"]:
            ttk.Label(card, text="Task clue sentence", style="Body.TLabel", font=("Microsoft YaHei", 9, "bold")).grid(row=2, column=0, columnspan=2, sticky="w")
            clue_box = tk.Text(card, height=3, wrap="word", font=("Consolas", 9), bg="#f7fafc", relief="solid", borderwidth=1)
            clue_box.insert("1.0", "\n\n".join(task["clue_sentences"]))
            clue_box.configure(state="disabled")
            clue_box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(2, 8))

        match = task["source_match"]
        match_summary = f"Match: {match['match_method']} | section: {match['match_section'] or 'not found'} | score: {match['match_score']}"
        if match["match_query"]:
            match_summary += f" | query: {match['match_query']}"
        ttk.Label(card, text=match_summary, style="Evidence.TLabel", wraplength=1380).grid(row=4, column=0, columnspan=2, sticky="w")
        context_box = tk.Text(card, height=13, wrap="word", font=("Consolas", 9), bg="#0f172a", fg="#e2e8f0", insertbackground="#ffffff", relief="flat", padx=10, pady=8)
        context_box.insert("1.0", match["context"] or "No source span was located by the original task name or clue sentence.")
        if match["context"] and match["context_match_start"] is not None:
            start = int(match["context_match_start"])
            end = int(match["context_match_end"])
            context_box.tag_add("match", f"1.0+{start}c", f"1.0+{end}c")
            context_box.tag_config("match", background="#f59e0b", foreground="#111827")
        context_box.configure(state="disabled")
        context_box.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(3, 10))

        review_panel = ttk.Frame(card, style="Panel.TFrame")
        review_panel.grid(row=6, column=0, columnspan=2, sticky="ew")
        review_panel.columnconfigure(4, weight=1)
        ttk.Label(review_panel, text="Evidence sufficiency", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 7))
        score_var = tk.StringVar(value=str(existing.get("evidence_sufficiency_score", "")))
        score_choice = ttk.Combobox(review_panel, textvariable=score_var, width=50, state="readonly")
        score_choice["values"] = list(self.SCORE_LABELS)
        score_choice.configure(values=[f"{key_value} | {label}" if key_value else label for key_value, label in self.SCORE_LABELS.items()])
        current_score = score_var.get()
        score_choice.set(f"{current_score} | {self.SCORE_LABELS[current_score]}" if current_score else self.SCORE_LABELS[""])
        score_choice.bind("<<ComboboxSelected>>", lambda _event: self._schedule_autosave())
        score_choice.grid(row=0, column=1, sticky="w", padx=(0, 18))
        ttk.Label(review_panel, text="Manual status", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=0, column=2, sticky="w", padx=(0, 7))
        status_var = tk.StringVar(value=str(existing.get("manual_in_scanner_status", "")))
        status_choice = ttk.Combobox(review_panel, textvariable=status_var, width=35, state="readonly")
        status_choice.configure(values=[f"{key_value} | {label}" if key_value else label for key_value, label in self.STATUS_LABELS.items()])
        current_status = status_var.get()
        status_choice.set(f"{current_status} | {self.STATUS_LABELS[current_status]}" if current_status else self.STATUS_LABELS[""])
        status_choice.bind("<<ComboboxSelected>>", lambda _event: self._schedule_autosave())
        status_choice.grid(row=0, column=3, sticky="w")
        ttk.Label(review_panel, text="Notes", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=1, column=0, sticky="nw", pady=(8, 0))
        note_box = tk.Text(review_panel, height=3, wrap="word", font=("Microsoft YaHei", 9), bg="#ffffff", relief="solid", borderwidth=1)
        note_box.insert("1.0", str(existing.get("reviewer_note", "")))
        note_box.bind("<KeyRelease>", lambda _event: self._schedule_autosave())
        note_box.bind("<FocusOut>", lambda _event: self._schedule_autosave())
        note_box.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(8, 0))
        self.score_vars[key] = score_var
        self.status_vars[key] = status_var
        self.note_widgets[key] = note_box

    @staticmethod
    def _choice_key(value: Any, labels: Dict[str, str]) -> str:
        selected = str(value or "")
        if selected in labels:
            return selected
        if selected == labels.get("", ""):
            return ""
        key = selected.split(" | ", 1)[0] if " | " in selected else ""
        return key if key in labels else ""

    def _collect_current_reviews(self) -> None:
        for task in self._current_article()["tasks"]:
            key = task["task_review_key"]
            self.reviews[key] = {
                "task_review_key": key,
                "PMID": task["PMID"],
                "normalized_task_name": task["normalized_task_name"],
                "display_task_name": task["display_task_name"],
                "source_groups": task["source_groups"],
                "task_indices": task["task_indices"],
                "match_method": task["source_match"]["match_method"],
                "match_section": task["source_match"]["match_section"],
                "match_score": task["source_match"]["match_score"],
                "evidence_sufficiency_score": self._choice_key(self.score_vars[key].get(), self.SCORE_LABELS),
                "manual_in_scanner_status": self._choice_key(self.status_vars[key].get(), self.STATUS_LABELS),
                "reviewer_note": self.note_widgets[key].get("1.0", "end-1c").strip(),
                "reviewed_at": utc_now(),
            }

    def _review_payload(self) -> Dict[str, Any]:
        return {
            "generated_at": utc_now(),
            "sample_path": str(self.sample_path),
            "article_count": len(self.articles),
            "task_reviews": sorted(self.reviews.values(), key=lambda item: (item["PMID"], item["normalized_task_name"])),
        }

    def _write_outputs(self, final: bool) -> None:
        self._collect_current_reviews()
        payload = self._review_payload()
        target = self.final_json_path if final else self.autosave_path
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if final:
            rows = payload["task_reviews"]
            fields = list(rows[0]) if rows else ["task_review_key"]
            with self.final_csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=fields, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)

    def _schedule_autosave(self) -> None:
        if self.rendering or self.root is None:
            return
        if self.autosave_after_id is not None:
            self.root.after_cancel(self.autosave_after_id)
        self.autosave_after_id = self.root.after(650, self._autosave)

    def _autosave(self) -> None:
        self.autosave_after_id = None
        self._write_outputs(final=False)
        if self.autosave_var is not None:
            self.autosave_var.set(f"Autosaved {datetime.now().strftime('%H:%M:%S')}")

    def _save(self, show_message: bool = False) -> None:
        self._write_outputs(final=True)
        self._write_outputs(final=False)
        if show_message:
            self.messagebox.showinfo("Saved", f"Review saved to:\n{self.final_csv_path}")

    def _save_and_next(self) -> None:
        self._save(show_message=False)
        self._next_article()

    def _previous_article(self) -> None:
        self._write_outputs(final=False)
        self.article_index = max(0, self.article_index - 1)
        self._render_article()

    def _next_article(self) -> None:
        self._write_outputs(final=False)
        self.article_index = min(len(self.articles) - 1, self.article_index + 1)
        self._render_article()

    def _jump_to_article(self) -> None:
        wanted = self.jump_var.get()
        for index, article in enumerate(self.articles):
            if article["PMID"] == wanted:
                self._write_outputs(final=False)
                self.article_index = index
                self._render_article()
                break

    def _on_close(self) -> None:
        self._write_outputs(final=False)
        self.root.destroy()


def launch_task_inscanner_review(sample_path: Path, output_dir: Path, title: str = "Task In-Scanner Context Review") -> None:
    TaskInScannerContextReviewUI(sample_path=sample_path, output_dir=output_dir, title=title).run()
