"""
Desktop UI for manually reviewing task-cluster membership.

The reviewer displays one task cluster at a time. Each task occurrence can be
marked as:
  1. kept in the current cluster,
  2. reclassified to a new task type, or
  3. excluded from all downstream analyses.

The UI writes one final JSON per cluster, an autosave JSON per cluster, and
aggregate CSV/XLSX/JSONL tables for downstream processing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


DECISION_INCLUDE = "include_in_current_cluster"
DECISION_RECLASSIFY = "reclassify_to_new_task"
DECISION_EXCLUDE_ALL = "exclude_from_all_analysis"

DECISION_LABELS = {
    DECISION_INCLUDE: "Keep in current cluster",
    DECISION_RECLASSIFY: "Reclassify to another task",
    DECISION_EXCLUDE_ALL: "Exclude from all analyses",
}

RECLASSIFY_PREFIX = "should be classified into "
TASKS_PER_PAGE = 10


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def compact_text(value: Any, max_len: int = 220) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    while "  " in text:
        text = text.replace("  ", " ")
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def normalize_task_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .;:，。；：")


def extract_reclassified_task_name(reason: str) -> str:
    text = str(reason or "").strip()
    lower = text.lower()
    prefix = RECLASSIFY_PREFIX.lower()
    if lower.startswith(prefix):
        return normalize_task_name(text[len(RECLASSIFY_PREFIX) :])
    return ""


@dataclass
class TaskClusterReviewConfig:
    cluster_json_dir: Path
    naming_results_csv: Path
    output_dir: Path
    title: str = "ARneuro K258 Task Cluster Membership Review"


class TaskClusterMembershipReviewer:
    """Tkinter UI for checking task occurrence membership in task clusters."""

    def __init__(self, config: TaskClusterReviewConfig) -> None:
        self.config = config
        self.output_dir = config.output_dir
        self.review_json_dir = self.output_dir / "cluster_review_json"
        self.autosave_json_dir = self.output_dir / "cluster_review_autosave_json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.review_json_dir.mkdir(parents=True, exist_ok=True)
        self.autosave_json_dir.mkdir(parents=True, exist_ok=True)

        self.cluster_paths = sorted(config.cluster_json_dir.glob("cluster_*_tasks.json"))
        if not self.cluster_paths:
            raise FileNotFoundError(f"No cluster task JSON files found in: {config.cluster_json_dir}")

        self.cluster_ids = [self._cluster_id_from_path(path) for path in self.cluster_paths]
        self.naming_by_cluster = self._load_naming_results(config.naming_results_csv)
        self.index = self._first_unreviewed_index()

        self.root = None
        self.canvas = None
        self.scroll_frame = None
        self.header_panel = None
        self.task_pagination_panel = None
        self.tk = None
        self.ttk = None
        self.messagebox = None
        self.simpledialog = None

        self.status_var = None
        self.jump_var = None
        self.filter_var = None
        self.reclassify_visible_var = None
        self.cluster_name_var = None
        self.cluster_excluded_var = None
        self.cluster_exclusion_reason_var = None
        self.autosave_status_var = None
        self.current_task_page = 0

        self.decision_vars: Dict[str, Any] = {}
        self.reason_vars: Dict[str, Any] = {}
        self.reclassified_name_vars: Dict[str, Any] = {}
        self.task_name_vars: Dict[str, Any] = {}
        self.visible_keys: List[str] = []
        self.existing_review: Dict[str, Any] = {}
        self.existing_task_reviews: Dict[str, Dict[str, Any]] = {}
        self.cluster_data: Dict[str, Any] = {}
        self.autosave_after_id = None
        self.rendering = False

    @staticmethod
    def _cluster_id_from_path(path: Path) -> int:
        name = path.stem.replace("cluster_", "").replace("_tasks", "")
        return int(name)

    def _review_path(self, cluster_id: int) -> Path:
        return self.review_json_dir / f"cluster_{cluster_id:03d}_membership_review.json"

    def _autosave_path(self, cluster_id: int) -> Path:
        return self.autosave_json_dir / f"cluster_{cluster_id:03d}_membership_autosave.json"

    def _first_unreviewed_index(self) -> int:
        for idx, cluster_id in enumerate(self.cluster_ids):
            if not self._review_path(cluster_id).exists():
                return idx
        return 0

    def _load_naming_results(self, path: Path) -> Dict[int, Dict[str, Any]]:
        if not path.exists():
            return {}
        import pandas as pd

        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, sep=";")
        else:
            df = pd.read_excel(path)
        records: Dict[int, Dict[str, Any]] = {}
        for row in df.to_dict("records"):
            cluster_id = as_int(row.get("cluster_id"), -1)
            if cluster_id >= 0:
                records[cluster_id] = row
        return records

    def _load_cluster(self, cluster_id: int) -> Dict[str, Any]:
        path = self.config.cluster_json_dir / f"cluster_{cluster_id:03d}_tasks.json"
        return read_json(path)

    def _load_existing_review_record(self, cluster_id: int) -> Dict[str, Any]:
        final_path = self._review_path(cluster_id)
        autosave_path = self._autosave_path(cluster_id)
        if final_path.exists() and autosave_path.exists():
            if autosave_path.stat().st_mtime > final_path.stat().st_mtime:
                return read_json(autosave_path)
            return read_json(final_path)
        if final_path.exists():
            return read_json(final_path)
        if autosave_path.exists():
            return read_json(autosave_path)
        return {}

    @staticmethod
    def _membership_key(cluster_id: int, occurrence: Dict[str, Any]) -> str:
        pmid = str(occurrence.get("PMID", "")).strip()
        task_index = str(occurrence.get("task_index", "")).strip()
        task_id = str(occurrence.get("task_id", "")).strip()
        source_group = str(occurrence.get("source_group", "")).strip()
        return f"{cluster_id:03d}|{pmid}|{task_index}|{task_id}|{source_group}"

    def run(self) -> None:
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk

        self.tk = tk
        self.ttk = ttk
        self.messagebox = messagebox
        self.simpledialog = simpledialog

        root = tk.Tk()
        self.root = root
        root.title(self.config.title)
        root.geometry("1480x930")
        root.minsize(1140, 740)
        root.configure(bg="#f3f6fb")
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._configure_styles()
        self._build_shell()
        self._render_cluster()
        root.mainloop()

    def _configure_styles(self) -> None:
        assert self.ttk is not None
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#f3f6fb")
        style.configure("Panel.TFrame", background="#ffffff", relief="flat")
        style.configure("Card.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("Header.TLabel", background="#f3f6fb", foreground="#162033", font=("Microsoft YaHei", 16, "bold"))
        style.configure("SmallHeader.TLabel", background="#f3f6fb", foreground="#344054", font=("Microsoft YaHei", 10, "bold"))
        style.configure("Title.TLabel", background="#ffffff", foreground="#111827", font=("Microsoft YaHei", 13, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#2f3a4a", font=("Microsoft YaHei", 10))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#667085", font=("Microsoft YaHei", 9))
        style.configure("Metric.TLabel", background="#eef4ff", foreground="#1d4ed8", font=("Microsoft YaHei", 9, "bold"), padding=(8, 4))
        style.configure("Danger.TLabel", background="#fff1f2", foreground="#be123c", font=("Microsoft YaHei", 9, "bold"), padding=(8, 4))
        style.configure("Warn.TLabel", background="#fffbeb", foreground="#92400e", font=("Microsoft YaHei", 9, "bold"), padding=(8, 4))
        style.configure("Primary.TButton", font=("Microsoft YaHei", 10, "bold"), padding=(12, 7))
        style.configure("TCheckbutton", background="#ffffff", font=("Microsoft YaHei", 10))
        style.configure("TEntry", padding=(4, 3))

    def _build_shell(self) -> None:
        assert self.root is not None and self.tk is not None and self.ttk is not None
        tk = self.tk
        ttk = self.ttk

        outer = ttk.Frame(self.root, style="App.TFrame", padding=(18, 14, 18, 14))
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer, style="App.TFrame")
        top.pack(fill="x", pady=(0, 12))
        ttk.Label(top, text="K258 task cluster manual review", style="Header.TLabel").pack(side="left")
        right = ttk.Frame(top, style="App.TFrame")
        right.pack(side="right")
        ttk.Button(right, text="Previous", command=self._previous_cluster).pack(side="left", padx=4)
        ttk.Button(right, text="Temp save", command=self._save_temp_current_cluster).pack(side="left", padx=4)
        ttk.Button(right, text="Final save", command=lambda: self._save_current_cluster(show_message=True), style="Primary.TButton").pack(side="left", padx=4)
        ttk.Button(right, text="Save + next", command=self._save_and_next, style="Primary.TButton").pack(side="left", padx=4)
        ttk.Button(right, text="Next", command=self._next_cluster).pack(side="left", padx=4)

        nav = ttk.Frame(outer, style="App.TFrame")
        nav.pack(fill="x", pady=(0, 10))
        self.status_var = tk.StringVar(value="")
        self.autosave_status_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self.status_var, style="SmallHeader.TLabel").pack(side="left")
        ttk.Label(nav, textvariable=self.autosave_status_var, style="SmallHeader.TLabel").pack(side="left", padx=(18, 0))

        ttk.Label(nav, text="Jump:", background="#f3f6fb", font=("Microsoft YaHei", 10)).pack(side="left", padx=(20, 6))
        self.jump_var = tk.StringVar()
        jump = ttk.Combobox(nav, textvariable=self.jump_var, width=10, state="readonly")
        jump["values"] = [f"{cid:03d}" for cid in self.cluster_ids]
        jump.bind("<<ComboboxSelected>>", lambda _event: self._jump_to_cluster())
        jump.pack(side="left")

        ttk.Label(nav, text="Filter:", background="#f3f6fb", font=("Microsoft YaHei", 10)).pack(side="left", padx=(20, 6))
        self.filter_var = tk.StringVar(value="")
        filter_entry = ttk.Entry(nav, textvariable=self.filter_var, width=38)
        filter_entry.pack(side="left")
        filter_entry.bind("<KeyRelease>", self._on_filter_changed)

        action = ttk.Frame(outer, style="App.TFrame")
        action.pack(fill="x", pady=(0, 10))
        ttk.Button(action, text="Visible: keep", command=lambda: self._set_all_visible_decision(DECISION_INCLUDE)).pack(side="left", padx=(0, 4))
        ttk.Button(action, text="Visible: exclude from this cluster", command=self._exclude_visible_from_cluster).pack(side="left", padx=4)
        ttk.Button(action, text="Visible: exclude from all analyses", command=lambda: self._set_all_visible_decision(DECISION_EXCLUDE_ALL)).pack(side="left", padx=4)
        ttk.Label(action, text="Visible reclassify to:", background="#f3f6fb", font=("Microsoft YaHei", 10)).pack(side="left", padx=(18, 6))
        self.reclassify_visible_var = tk.StringVar(value="")
        ttk.Entry(action, textvariable=self.reclassify_visible_var, width=36).pack(side="left")
        ttk.Button(action, text="Apply reclassify", command=self._reclassify_visible).pack(side="left", padx=4)
        ttk.Button(action, text="Batch reclassify tasks", command=self._open_batch_reclassify_window).pack(side="left", padx=(12, 4))
        ttk.Button(action, text="Exclude entire cluster", command=self._exclude_entire_cluster).pack(side="right", padx=4)
        ttk.Button(action, text="Restore entire cluster", command=self._restore_entire_cluster).pack(side="right", padx=4)

        self.header_panel = ttk.Frame(outer, style="Panel.TFrame", padding=(14, 12, 14, 12))
        self.header_panel.pack(fill="x", pady=(0, 12))

        # This toolbar deliberately lives outside the scrolling task canvas.
        self.task_pagination_panel = ttk.Frame(outer, style="Panel.TFrame", padding=(12, 8, 12, 8))
        self.task_pagination_panel.pack(fill="x", pady=(0, 10))

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, bg="#f3f6fb", highlightthickness=0)
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

    def _current_cluster_id(self) -> int:
        return self.cluster_ids[self.index]

    def _clear_frame(self, frame: Any) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _render_cluster(self) -> None:
        assert self.jump_var is not None and self.status_var is not None and self.tk is not None
        self.rendering = True
        cluster_id = self._current_cluster_id()
        self.cluster_data = self._load_cluster(cluster_id)
        self.existing_review = self._load_existing_review_record(cluster_id)
        self.existing_task_reviews = {
            str(item.get("membership_key", "")): item
            for item in self.existing_review.get("task_membership_reviews", [])
            if item.get("membership_key")
        }
        self.decision_vars = {}
        self.reason_vars = {}
        self.reclassified_name_vars = {}
        self.task_name_vars = {}
        self.visible_keys = []
        self.current_task_page = 0

        self.cluster_excluded_var = self.tk.BooleanVar(value=bool(self.existing_review.get("cluster_excluded", False)))
        naming = self.naming_by_cluster.get(cluster_id, {})
        default_cluster_name = str(naming.get("recommended_cluster_task_name") or f"Cluster {cluster_id:03d}")
        self.cluster_name_var = self.tk.StringVar(
            value=str(self.existing_review.get("reviewed_cluster_task_name") or default_cluster_name)
        )
        self.cluster_exclusion_reason_var = self.tk.StringVar(
            value=str(self.existing_review.get("cluster_exclusion_reason", "") or "")
        )
        self.cluster_name_var.trace_add("write", lambda *_args: self._schedule_autosave())
        self.cluster_excluded_var.trace_add("write", lambda *_args: self._schedule_autosave())
        self.cluster_exclusion_reason_var.trace_add("write", lambda *_args: self._schedule_autosave())

        self.jump_var.set(f"{cluster_id:03d}")
        reviewed = sum(1 for cid in self.cluster_ids if self._review_path(cid).exists())
        autosaved = sum(1 for cid in self.cluster_ids if self._autosave_path(cid).exists())
        self.status_var.set(f"Cluster {self.index + 1}/{len(self.cluster_ids)} | final saved {reviewed}/{len(self.cluster_ids)}")
        if self.autosave_status_var is not None:
            self.autosave_status_var.set(f"autosaved {autosaved}/{len(self.cluster_ids)}")
        self._render_header()
        self._render_task_cards()
        if self.canvas is not None:
            self.canvas.yview_moveto(0)
        self.rendering = False

    def _render_header(self) -> None:
        assert self.header_panel is not None and self.ttk is not None
        ttk = self.ttk
        self._clear_frame(self.header_panel)
        cluster_id = self._current_cluster_id()
        data = self.cluster_data
        stats = data.get("stats", {})
        naming = self.naming_by_cluster.get(cluster_id, {})

        model_title = str(naming.get("recommended_cluster_task_name") or f"Cluster {cluster_id:03d}")
        title = self.cluster_name_var.get().strip() if self.cluster_name_var is not None else model_title
        row1 = ttk.Frame(self.header_panel, style="Panel.TFrame")
        row1.pack(fill="x")
        ttk.Label(row1, text=f"Cluster {cluster_id:03d}: {title}", style="Title.TLabel").pack(side="left", anchor="w")
        ttk.Button(row1, text="Rename cluster", command=self._rename_current_cluster).pack(side="left", padx=(12, 0))
        review_path = self._review_path(cluster_id)
        autosave_path = self._autosave_path(cluster_id)
        if review_path.exists():
            ttk.Label(row1, text="final saved", style="Metric.TLabel").pack(side="right")
        elif autosave_path.exists():
            ttk.Label(row1, text="autosaved only", style="Warn.TLabel").pack(side="right")
        else:
            ttk.Label(row1, text="not saved", style="Danger.TLabel").pack(side="right")

        metric_row = ttk.Frame(self.header_panel, style="Panel.TFrame")
        metric_row.pack(fill="x", pady=(10, 8))
        metrics = [
            ("articles", stats.get("article_count", naming.get("article_count", ""))),
            ("occurrences", stats.get("task_occurrence_count", naming.get("task_occurrence_count", ""))),
            ("unique tasks", stats.get("unique_task_count", naming.get("unique_task_count", ""))),
            ("patient N", stats.get("patient_participant_total_sum", "")),
            ("typical N", stats.get("typical_human_participant_total_sum", "")),
            ("domain", naming.get("core_linguistic_domain", "")),
            ("operation", naming.get("core_processing_operation", "")),
        ]
        for label, value in metrics:
            ttk.Label(metric_row, text=f"{label}: {value}", style="Metric.TLabel").pack(side="left", padx=(0, 6))

        definition = naming.get("short_definition") or ""
        paradigm = naming.get("paradigm_description") or ""
        evidence = naming.get("evidence_summary") or ""
        ttk.Label(self.header_panel, text=f"Model name: {model_title}", style="Muted.TLabel", wraplength=1320).pack(anchor="w", pady=(8, 2))
        ttk.Label(self.header_panel, text=f"Definition: {compact_text(definition, 520)}", style="Body.TLabel", wraplength=1320).pack(anchor="w", pady=(2, 2))
        ttk.Label(self.header_panel, text=f"Paradigm: {compact_text(paradigm, 760)}", style="Body.TLabel", wraplength=1320).pack(anchor="w", pady=(2, 2))
        ttk.Label(self.header_panel, text=f"Evidence: {compact_text(evidence, 760)}", style="Muted.TLabel", wraplength=1320).pack(anchor="w", pady=(2, 8))

        cluster_action = ttk.Frame(self.header_panel, style="Panel.TFrame")
        cluster_action.pack(fill="x")
        ttk.Checkbutton(cluster_action, text="Exclude this entire cluster from all analyses", variable=self.cluster_excluded_var).pack(side="left")
        ttk.Label(cluster_action, text="Cluster exclusion reason:", style="Body.TLabel").pack(side="left", padx=(16, 6))
        ttk.Entry(cluster_action, textvariable=self.cluster_exclusion_reason_var).pack(side="left", fill="x", expand=True)

    def _rename_current_cluster(self) -> None:
        if self.simpledialog is None or self.cluster_name_var is None:
            return
        cluster_id = self._current_cluster_id()
        current_name = self.cluster_name_var.get().strip()
        new_name = self.simpledialog.askstring(
            "Rename cluster",
            f"Enter reviewed name for Cluster {cluster_id:03d}:",
            initialvalue=current_name,
            parent=self.root,
        )
        if new_name is None:
            return
        new_name = normalize_task_name(new_name)
        if not new_name:
            return
        self.cluster_name_var.set(new_name)
        self._render_header()
        self._save_temp_current_cluster()

    def _iter_occurrences(self) -> List[Dict[str, Any]]:
        occurrences = self.cluster_data.get("occurrences", [])
        return occurrences if isinstance(occurrences, list) else []

    def _task_matches_filter(self, occurrence: Dict[str, Any], query: str) -> bool:
        if not query:
            return True
        haystack = " ".join(
            str(occurrence.get(key, ""))
            for key in [
                "PMID",
                "task_index",
                "source_group",
                "structured_task_name",
                "reviewed_task_name",
                "standardized_task_name",
                "raw_task_name",
                "description",
                "clue_sentences",
                "linguistic_domain",
                "processing_operation",
                "stimulus_modality",
                "stimulus_unit",
                "response_mode",
            ]
        ).lower()
        return query.lower() in haystack

    def _decision_from_existing(self, existing: Dict[str, Any]) -> str:
        decision = str(existing.get("membership_decision", "") or "")
        if decision in DECISION_LABELS:
            return decision
        if bool(existing.get("exclude_from_all_analysis", False)):
            return DECISION_EXCLUDE_ALL
        reason = str(existing.get("exclusion_reason", "") or "")
        if extract_reclassified_task_name(reason):
            return DECISION_RECLASSIFY
        if bool(existing.get("include_in_cluster", True)):
            return DECISION_INCLUDE
        return DECISION_RECLASSIFY if reason.lower().startswith(RECLASSIFY_PREFIX) else DECISION_EXCLUDE_ALL

    def _ensure_task_vars(self, cluster_id: int, occurrence: Dict[str, Any]) -> str:
        assert self.tk is not None
        key = self._membership_key(cluster_id, occurrence)
        if key not in self.decision_vars:
            existing = self.existing_task_reviews.get(key, {})
            decision = self._decision_from_existing(existing)
            reason = str(existing.get("exclusion_reason", "") or "")
            reclassified_name = str(existing.get("reclassified_task_name", "") or "")
            if not reclassified_name:
                reclassified_name = extract_reclassified_task_name(reason)
            default_task_name = (
                occurrence.get("structured_task_name")
                or occurrence.get("standardized_task_name")
                or occurrence.get("raw_task_name")
                or "Unnamed task"
            )
            reviewed_task_name = str(existing.get("reviewed_task_name") or default_task_name)
            self.decision_vars[key] = self.tk.StringVar(value=decision)
            self.reason_vars[key] = self.tk.StringVar(value=reason)
            self.reclassified_name_vars[key] = self.tk.StringVar(value=reclassified_name)
            self.task_name_vars[key] = self.tk.StringVar(value=reviewed_task_name)
            self.decision_vars[key].trace_add("write", lambda *_args: self._schedule_autosave())
            self.reason_vars[key].trace_add("write", lambda *_args: self._schedule_autosave())
            self.reclassified_name_vars[key].trace_add("write", lambda *_args: self._schedule_autosave())
            self.task_name_vars[key].trace_add("write", lambda *_args: self._schedule_autosave())
        return key

    def _on_filter_changed(self, _event: Any = None) -> None:
        """Restart pagination whenever the task filter changes."""
        self.current_task_page = 0
        self._render_task_cards()

    @staticmethod
    def _page_button_numbers(page_count: int, current_page: int) -> List[Any]:
        """Return compact, clickable page numbers with ellipses for long clusters."""
        if page_count <= 11:
            return list(range(page_count))

        nearby = {0, page_count - 1}
        nearby.update(range(max(0, current_page - 2), min(page_count, current_page + 3)))
        numbers = sorted(nearby)
        display: List[Any] = []
        previous = -1
        for number in numbers:
            if number - previous > 1:
                display.append(None)
            display.append(number)
            previous = number
        return display

    def _goto_task_page(self, page_index: int, page_count: int) -> None:
        if page_count <= 0:
            return
        self.current_task_page = max(0, min(page_index, page_count - 1))
        self._render_task_cards()
        if self.canvas is not None:
            self.canvas.yview_moveto(0)

    def _render_task_pagination(self, total_count: int, page_count: int) -> None:
        assert self.task_pagination_panel is not None and self.ttk is not None
        ttk = self.ttk
        start = self.current_task_page * TASKS_PER_PAGE + 1 if total_count else 0
        end = min(total_count, (self.current_task_page + 1) * TASKS_PER_PAGE)

        ttk.Label(
            self.task_pagination_panel,
            text=f"Tasks {start}-{end} of {total_count} | Page {self.current_task_page + 1}/{page_count}",
            style="SmallHeader.TLabel",
        ).pack(side="left")

        buttons = ttk.Frame(self.task_pagination_panel, style="Panel.TFrame")
        buttons.pack(side="right")
        ttk.Button(
            buttons,
            text="Previous page",
            command=lambda: self._goto_task_page(self.current_task_page - 1, page_count),
            state="normal" if self.current_task_page > 0 else "disabled",
        ).pack(side="left", padx=(0, 4))
        for number in self._page_button_numbers(page_count, self.current_task_page):
            if number is None:
                ttk.Label(buttons, text="...", style="Body.TLabel").pack(side="left", padx=3)
                continue
            style = "Primary.TButton" if number == self.current_task_page else "TButton"
            ttk.Button(
                buttons,
                text=str(number + 1),
                width=3,
                style=style,
                command=lambda target=number: self._goto_task_page(target, page_count),
            ).pack(side="left", padx=2)
        ttk.Button(
            buttons,
            text="Next page",
            command=lambda: self._goto_task_page(self.current_task_page + 1, page_count),
            state="normal" if self.current_task_page < page_count - 1 else "disabled",
        ).pack(side="left", padx=(4, 0))

    def _render_task_cards(self) -> None:
        assert self.scroll_frame is not None and self.task_pagination_panel is not None and self.ttk is not None
        ttk = self.ttk
        self._clear_frame(self.scroll_frame)
        self._clear_frame(self.task_pagination_panel)
        cluster_id = self._current_cluster_id()
        query = self.filter_var.get().strip() if self.filter_var is not None else ""
        self.visible_keys = []

        visible_occurrences: List[Any] = []
        for idx, occurrence in enumerate(self._iter_occurrences(), start=1):
            key = self._ensure_task_vars(cluster_id, occurrence)
            reviewed_name_match = query.lower() in self.task_name_vars[key].get().lower() if query else True
            if not self._task_matches_filter(occurrence, query) and not reviewed_name_match:
                continue
            self.visible_keys.append(key)
            visible_occurrences.append((idx, occurrence, key))

        visible_count = len(visible_occurrences)
        if visible_count == 0:
            empty = ttk.Frame(self.scroll_frame, style="Card.TFrame", padding=18)
            empty.pack(fill="x", pady=(0, 10), padx=(2, 10))
            ttk.Label(empty, text="No tasks match the current filter.", style="Title.TLabel").pack(anchor="w")
            return

        page_count = (visible_count + TASKS_PER_PAGE - 1) // TASKS_PER_PAGE
        if self.current_task_page >= page_count:
            self.current_task_page = page_count - 1
        if page_count > 1:
            self._render_task_pagination(visible_count, page_count)

        start_index = self.current_task_page * TASKS_PER_PAGE
        end_index = start_index + TASKS_PER_PAGE
        for idx, occurrence, key in visible_occurrences[start_index:end_index]:
            self._render_task_card(idx, occurrence, key)

    def _render_task_card(self, idx: int, occurrence: Dict[str, Any], key: str) -> None:
        assert self.scroll_frame is not None and self.ttk is not None
        ttk = self.ttk
        card = ttk.Frame(self.scroll_frame, style="Card.TFrame", padding=(14, 12, 14, 12))
        card.pack(fill="x", pady=(0, 10), padx=(2, 10))

        top = ttk.Frame(card, style="Panel.TFrame")
        top.pack(fill="x")
        task_name = occurrence.get("structured_task_name") or occurrence.get("standardized_task_name") or occurrence.get("raw_task_name") or "Unnamed task"
        reviewed_task_name = self.task_name_vars[key].get().strip()
        title = reviewed_task_name or task_name
        ttk.Label(top, text=f"{idx}. {title}", style="Title.TLabel", wraplength=820).pack(side="left", anchor="w")
        ttk.Button(top, text="Rename task", command=lambda k=key, original=task_name: self._rename_single_task(k, original)).pack(side="left", padx=(10, 0))
        decision_box = ttk.Combobox(top, textvariable=self.decision_vars[key], state="readonly", width=30)
        decision_box["values"] = list(DECISION_LABELS.keys())
        decision_box.pack(side="right", padx=(8, 0))

        name_row = ttk.Frame(card, style="Panel.TFrame")
        name_row.pack(fill="x", pady=(7, 2))
        ttk.Label(name_row, text="Reviewed task name:", style="Body.TLabel").pack(side="left")
        ttk.Entry(name_row, textvariable=self.task_name_vars[key]).pack(side="left", fill="x", expand=True, padx=(8, 0))

        button_row = ttk.Frame(card, style="Panel.TFrame")
        button_row.pack(fill="x", pady=(6, 2))
        ttk.Button(button_row, text="Keep", command=lambda k=key: self._mark_task_keep(k)).pack(side="left", padx=(0, 4))
        ttk.Button(button_row, text="Reclassify", command=lambda k=key: self._mark_task_reclassify(k)).pack(side="left", padx=4)
        ttk.Button(button_row, text="Exclude all", command=lambda k=key: self._mark_task_exclude_all(k)).pack(side="left", padx=4)
        ttk.Label(button_row, text="New task type:", style="Body.TLabel").pack(side="left", padx=(18, 6))
        ttk.Entry(button_row, textvariable=self.reclassified_name_vars[key], width=38).pack(side="left")

        meta = (
            f"PMID {occurrence.get('PMID', '')} | task_index {occurrence.get('task_index', '')} | "
            f"{occurrence.get('source_group', '')} | participants {occurrence.get('participant_total_parsed_for_source_group', '')}"
        )
        ttk.Label(card, text=meta, style="Muted.TLabel").pack(anchor="w", pady=(5, 3))
        names = (
            f"raw: {occurrence.get('raw_task_name', '')}    "
            f"extracted: {occurrence.get('extracted_task_name', '')}    "
            f"original: {occurrence.get('original_standardized_task_name', '')}"
        )
        ttk.Label(card, text=compact_text(names, 940), style="Muted.TLabel", wraplength=1300).pack(anchor="w", pady=(0, 6))
        feature_line = (
            f"domain={occurrence.get('linguistic_domain', '')}; "
            f"operation={occurrence.get('processing_operation', '')}; "
            f"modality={occurrence.get('stimulus_modality', '')}; "
            f"unit={occurrence.get('stimulus_unit', '')}; "
            f"response={occurrence.get('response_mode', '')}"
        )
        ttk.Label(card, text=feature_line, style="Body.TLabel", wraplength=1300).pack(anchor="w", pady=(0, 6))
        ttk.Label(card, text=f"Description: {occurrence.get('description', '')}", style="Body.TLabel", wraplength=1300).pack(anchor="w", pady=(0, 5))
        ttk.Label(card, text=f"Clue sentence: {occurrence.get('clue_sentences', '')}", style="Muted.TLabel", wraplength=1300).pack(anchor="w", pady=(0, 8))

        reason_row = ttk.Frame(card, style="Panel.TFrame")
        reason_row.pack(fill="x")
        ttk.Label(reason_row, text="Reason:", style="Body.TLabel").pack(side="left")
        ttk.Entry(reason_row, textvariable=self.reason_vars[key]).pack(side="left", fill="x", expand=True, padx=(8, 0))

    def _rename_single_task(self, key: str, original_name: str) -> None:
        if self.simpledialog is None:
            return
        current_name = self.task_name_vars[key].get().strip() if key in self.task_name_vars else original_name
        new_name = self.simpledialog.askstring(
            "Rename task",
            "Enter reviewed task name:",
            initialvalue=current_name,
            parent=self.root,
        )
        if new_name is None:
            return
        new_name = normalize_task_name(new_name)
        if not new_name:
            return
        self.task_name_vars[key].set(new_name)
        self._save_temp_current_cluster()

    @staticmethod
    def _name_tokens(value: Any) -> set[str]:
        """Split task names into words so matching is case-insensitive and order-free."""
        return set(re.findall(r"[^\W_]+", str(value or "").casefold(), flags=re.UNICODE))

    def _batch_reclassify_matches(self, source_name_query: str) -> List[str]:
        target_tokens = self._name_tokens(source_name_query)
        if not target_tokens:
            return []

        cluster_id = self._current_cluster_id()
        matched_keys: List[str] = []
        for occurrence in self._iter_occurrences():
            key = self._ensure_task_vars(cluster_id, occurrence)
            names = [
                self.task_name_vars[key].get(),
                occurrence.get("structured_task_name", ""),
                occurrence.get("standardized_task_name", ""),
                occurrence.get("raw_task_name", ""),
                occurrence.get("extracted_task_name", ""),
                occurrence.get("original_standardized_task_name", ""),
            ]
            if any(target_tokens.issubset(self._name_tokens(name)) for name in names):
                matched_keys.append(key)
        return matched_keys

    def _open_batch_reclassify_window(self) -> None:
        """Open a dialog for assigning matching task occurrences to one category."""
        if self.root is None or self.tk is None or self.ttk is None:
            return
        ttk = self.ttk
        window = self.tk.Toplevel(self.root)
        window.title("Batch reclassify tasks")
        window.geometry("620x190")
        window.minsize(520, 170)
        window.transient(self.root)

        panel = ttk.Frame(window, padding=(18, 16, 18, 14))
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="Batch reclassify tasks in the current cluster", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            panel,
            text="Enter the task category once. It is used both to find matching task names and as the new category. Word order and capitalization are ignored.",
            style="Body.TLabel",
            wraplength=560,
        ).pack(anchor="w", pady=(8, 10))

        ttk.Label(panel, text="Task category:", style="Body.TLabel").pack(anchor="w")
        category_var = self.tk.StringVar(value="")
        entry = ttk.Entry(panel, textvariable=category_var)
        entry.pack(fill="x")
        entry.focus_set()
        preview_var = self.tk.StringVar(value="0 matching task occurrences")
        ttk.Label(panel, textvariable=preview_var, style="Muted.TLabel").pack(anchor="w", pady=(6, 8))

        def update_preview(*_args: Any) -> None:
            category = normalize_task_name(category_var.get())
            count = len(self._batch_reclassify_matches(category)) if category else 0
            preview_var.set(f"{count} matching task occurrence(s) in this cluster")

        category_var.trace_add("write", update_preview)

        buttons = ttk.Frame(panel, style="Panel.TFrame")
        buttons.pack(fill="x", pady=(2, 0))

        def apply_batch_reclassify() -> None:
            category = normalize_task_name(category_var.get())
            if not category:
                if self.messagebox is not None:
                    self.messagebox.showwarning("Missing category", "Enter the new task category first.", parent=window)
                return
            matched_keys = self._batch_reclassify_matches(category)
            if not matched_keys:
                if self.messagebox is not None:
                    self.messagebox.showinfo("No matches", "No task name contains all of the entered words.", parent=window)
                return
            for key in matched_keys:
                # Keep the original/reviewed task name unchanged. Only update
                # the downstream task-category assignment.
                self.reclassified_name_vars[key].set(category)
                self.decision_vars[key].set(DECISION_RECLASSIFY)
                self.reason_vars[key].set(f"{RECLASSIFY_PREFIX}{category}.")
            self._save_temp_current_cluster()
            self._render_task_cards()
            window.destroy()
            if self.messagebox is not None:
                self.messagebox.showinfo(
                    "Batch reclassification complete",
                    f"Assigned {len(matched_keys)} task occurrence(s) to category:\n{category}",
                    parent=self.root,
                )

        ttk.Button(buttons, text="Apply batch reclassification", command=apply_batch_reclassify, style="Primary.TButton").pack(side="right")
        ttk.Button(buttons, text="Cancel", command=window.destroy).pack(side="right", padx=(0, 8))
        window.bind("<Return>", lambda _event: apply_batch_reclassify())
        window.bind("<Escape>", lambda _event: window.destroy())

    def _mark_task_keep(self, key: str) -> None:
        self.decision_vars[key].set(DECISION_INCLUDE)
        self.reason_vars[key].set("")

    def _mark_task_reclassify(self, key: str) -> None:
        name = normalize_task_name(self.reclassified_name_vars[key].get())
        self.decision_vars[key].set(DECISION_RECLASSIFY)
        if name:
            self.reason_vars[key].set(f"{RECLASSIFY_PREFIX}{name}.")

    def _mark_task_exclude_all(self, key: str) -> None:
        self.decision_vars[key].set(DECISION_EXCLUDE_ALL)
        if not str(self.reason_vars[key].get()).strip():
            self.reason_vars[key].set("exclude from all analyses: not a valid target language task.")

    def _set_all_visible_decision(self, decision: str) -> None:
        for key in self.visible_keys:
            self.decision_vars[key].set(decision)
            if decision == DECISION_INCLUDE:
                self.reason_vars[key].set("")
            elif decision == DECISION_EXCLUDE_ALL and not str(self.reason_vars[key].get()).strip():
                self.reason_vars[key].set("exclude from all analyses: not a valid target language task.")

    def _exclude_visible_from_cluster(self) -> None:
        for key in self.visible_keys:
            self.decision_vars[key].set(DECISION_RECLASSIFY)
            if not str(self.reason_vars[key].get()).strip():
                self.reason_vars[key].set("excluded from this cluster; reclassification target not specified.")

    def _reclassify_visible(self) -> None:
        assert self.reclassify_visible_var is not None
        name = normalize_task_name(self.reclassify_visible_var.get())
        if not name:
            if self.messagebox is not None:
                self.messagebox.showwarning("Missing task type", "Please enter the new task type before applying.")
            return
        for key in self.visible_keys:
            self.reclassified_name_vars[key].set(name)
            self.decision_vars[key].set(DECISION_RECLASSIFY)
            self.reason_vars[key].set(f"{RECLASSIFY_PREFIX}{name}.")

    def _exclude_entire_cluster(self) -> None:
        assert self.cluster_excluded_var is not None and self.cluster_exclusion_reason_var is not None
        self.cluster_excluded_var.set(True)
        if not self.cluster_exclusion_reason_var.get().strip():
            self.cluster_exclusion_reason_var.set("whole cluster excluded: tasks are outside the target analysis scope.")
        for key in list(self.decision_vars):
            self.decision_vars[key].set(DECISION_EXCLUDE_ALL)
            if not str(self.reason_vars[key].get()).strip():
                self.reason_vars[key].set(self.cluster_exclusion_reason_var.get())

    def _restore_entire_cluster(self) -> None:
        assert self.cluster_excluded_var is not None
        self.cluster_excluded_var.set(False)
        for key in list(self.decision_vars):
            self.decision_vars[key].set(DECISION_INCLUDE)
            self.reason_vars[key].set("")
            self.reclassified_name_vars[key].set("")

    def _collect_current_review(self, temporary: bool = False) -> Dict[str, Any]:
        cluster_id = self._current_cluster_id()
        data = self.cluster_data
        naming = self.naming_by_cluster.get(cluster_id, {})
        reviews: List[Dict[str, Any]] = []
        included_count = 0
        reclassified_count = 0
        excluded_all_count = 0
        missing_reason = 0

        cluster_excluded = bool(self.cluster_excluded_var.get()) if self.cluster_excluded_var is not None else False
        cluster_reason = str(self.cluster_exclusion_reason_var.get()).strip() if self.cluster_exclusion_reason_var is not None else ""
        reviewed_cluster_name = (
            normalize_task_name(self.cluster_name_var.get())
            if self.cluster_name_var is not None
            else str(naming.get("recommended_cluster_task_name", ""))
        )

        for occurrence in self._iter_occurrences():
            key = self._ensure_task_vars(cluster_id, occurrence)
            original_task_name = (
                occurrence.get("structured_task_name")
                or occurrence.get("standardized_task_name")
                or occurrence.get("raw_task_name")
                or ""
            )
            reviewed_task_name = normalize_task_name(self.task_name_vars[key].get()) or normalize_task_name(original_task_name)
            decision = str(self.decision_vars[key].get())
            if decision not in DECISION_LABELS:
                decision = DECISION_INCLUDE
            reason = str(self.reason_vars[key].get()).strip()
            reclassified_name = normalize_task_name(self.reclassified_name_vars[key].get())
            manual_reclassified_name = extract_reclassified_task_name(reason)
            if manual_reclassified_name:
                decision = DECISION_RECLASSIFY
                reclassified_name = manual_reclassified_name
            if decision == DECISION_RECLASSIFY and not reclassified_name:
                reclassified_name = extract_reclassified_task_name(reason)
            if decision == DECISION_RECLASSIFY and reclassified_name and not reason:
                reason = f"{RECLASSIFY_PREFIX}{reclassified_name}."

            include_in_cluster = decision == DECISION_INCLUDE and not cluster_excluded
            exclude_from_all = decision == DECISION_EXCLUDE_ALL or cluster_excluded
            if include_in_cluster:
                included_count += 1
            elif decision == DECISION_RECLASSIFY:
                reclassified_count += 1
            else:
                excluded_all_count += 1
            if not include_in_cluster and not reason and not cluster_reason:
                missing_reason += 1

            final_assignment = ""
            if include_in_cluster:
                final_assignment = reviewed_cluster_name
            elif decision == DECISION_RECLASSIFY:
                final_assignment = reclassified_name

            reviews.append(
                {
                    "membership_key": key,
                    "cluster_id": cluster_id,
                    "recommended_cluster_task_name": naming.get("recommended_cluster_task_name", ""),
                    "reviewed_cluster_task_name": reviewed_cluster_name,
                    "membership_decision": decision,
                    "include_in_cluster": include_in_cluster,
                    "exclude_from_all_analysis": exclude_from_all,
                    "reclassified_task_name": reclassified_name,
                    "final_task_assignment_after_review": final_assignment,
                    "exclusion_reason": reason or cluster_reason,
                    "PMID": occurrence.get("PMID", ""),
                    "task_index": occurrence.get("task_index", ""),
                    "task_id": occurrence.get("task_id", ""),
                    "source_group": occurrence.get("source_group", ""),
                    "structured_task_name": occurrence.get("structured_task_name", ""),
                    "reviewed_task_name": reviewed_task_name,
                    "task_name_was_edited": reviewed_task_name != normalize_task_name(original_task_name),
                    "standardized_task_name": occurrence.get("standardized_task_name", ""),
                    "raw_task_name": occurrence.get("raw_task_name", ""),
                    "extracted_task_name": occurrence.get("extracted_task_name", ""),
                    "original_standardized_task_name": occurrence.get("original_standardized_task_name", ""),
                    "description": occurrence.get("description", ""),
                    "clue_sentences": occurrence.get("clue_sentences", ""),
                    "linguistic_domain": occurrence.get("linguistic_domain", ""),
                    "processing_operation": occurrence.get("processing_operation", ""),
                    "stimulus_modality": occurrence.get("stimulus_modality", ""),
                    "stimulus_unit": occurrence.get("stimulus_unit", ""),
                    "response_mode": occurrence.get("response_mode", ""),
                    "participant_total_raw_for_source_group": occurrence.get("participant_total_raw_for_source_group", ""),
                    "participant_total_parsed_for_source_group": occurrence.get("participant_total_parsed_for_source_group", ""),
                }
            )

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return {
            "cluster_id": cluster_id,
            "reviewed_at": now if not temporary else self.existing_review.get("reviewed_at", ""),
            "temporary_saved_at": now if temporary else "",
            "cluster_excluded": cluster_excluded,
            "cluster_exclusion_reason": cluster_reason,
            "recommended_cluster_task_name": naming.get("recommended_cluster_task_name", ""),
            "reviewed_cluster_task_name": reviewed_cluster_name,
            "short_definition": naming.get("short_definition", ""),
            "paradigm_description": naming.get("paradigm_description", ""),
            "stats": data.get("stats", {}),
            "included_occurrence_count": included_count,
            "reclassified_occurrence_count": reclassified_count,
            "excluded_from_all_occurrence_count": excluded_all_count,
            "excluded_occurrence_count": reclassified_count + excluded_all_count,
            "excluded_without_reason_count": missing_reason,
            "task_membership_reviews": reviews,
        }

    def _save_temp_current_cluster(self) -> bool:
        record = self._collect_current_review(temporary=True)
        write_json(self._autosave_path(record["cluster_id"]), record)
        if self.autosave_status_var is not None:
            self.autosave_status_var.set(f"autosaved at {datetime.now().strftime('%H:%M:%S')}")
        return True

    def _schedule_autosave(self) -> None:
        if self.rendering or self.root is None:
            return
        if self.autosave_after_id is not None:
            try:
                self.root.after_cancel(self.autosave_after_id)
            except Exception:
                pass
        self.autosave_after_id = self.root.after(900, self._save_temp_current_cluster)

    def _save_current_cluster(self, show_message: bool = True) -> bool:
        assert self.messagebox is not None
        record = self._collect_current_review(temporary=False)
        missing_reason = int(record.get("excluded_without_reason_count", 0))
        if missing_reason:
            ok = self.messagebox.askyesno(
                "Missing exclusion reasons",
                f"{missing_reason} excluded/reclassified tasks do not have a reason. Save anyway?",
            )
            if not ok:
                return False
        write_json(self._review_path(record["cluster_id"]), record)
        self.existing_review = record
        self._save_temp_current_cluster()
        self._write_aggregate_outputs()
        reviewed = sum(1 for cid in self.cluster_ids if self._review_path(cid).exists())
        if self.status_var is not None:
            self.status_var.set(f"Cluster {self.index + 1}/{len(self.cluster_ids)} | final saved {reviewed}/{len(self.cluster_ids)}")
        self._render_header()
        if show_message:
            self.messagebox.showinfo("Saved", f"Cluster {record['cluster_id']:03d} review has been saved.")
        return True

    def _write_aggregate_outputs(self) -> None:
        import pandas as pd

        cluster_records: List[Dict[str, Any]] = []
        task_records: List[Dict[str, Any]] = []
        for path in sorted(self.review_json_dir.glob("cluster_*_membership_review.json")):
            record = read_json(path)
            cluster_records.append(
                {
                    "cluster_id": record.get("cluster_id", ""),
                    "reviewed_at": record.get("reviewed_at", ""),
                    "cluster_excluded": record.get("cluster_excluded", False),
                    "cluster_exclusion_reason": record.get("cluster_exclusion_reason", ""),
                    "recommended_cluster_task_name": record.get("recommended_cluster_task_name", ""),
                    "reviewed_cluster_task_name": record.get("reviewed_cluster_task_name", record.get("recommended_cluster_task_name", "")),
                    "included_occurrence_count": record.get("included_occurrence_count", 0),
                    "reclassified_occurrence_count": record.get("reclassified_occurrence_count", 0),
                    "excluded_from_all_occurrence_count": record.get("excluded_from_all_occurrence_count", 0),
                    "excluded_without_reason_count": record.get("excluded_without_reason_count", 0),
                    "article_count": record.get("stats", {}).get("article_count", ""),
                    "task_occurrence_count": record.get("stats", {}).get("task_occurrence_count", ""),
                    "unique_task_count": record.get("stats", {}).get("unique_task_count", ""),
                    "patient_participant_total_sum": record.get("stats", {}).get("patient_participant_total_sum", ""),
                    "typical_human_participant_total_sum": record.get("stats", {}).get("typical_human_participant_total_sum", ""),
                }
            )
            for item in record.get("task_membership_reviews", []):
                task_records.append(item)

        cluster_df = pd.DataFrame(cluster_records).sort_values("cluster_id") if cluster_records else pd.DataFrame()
        task_df = pd.DataFrame(task_records).sort_values(["cluster_id", "PMID", "task_index"]) if task_records else pd.DataFrame()
        reclass_df = self._build_reclassification_groups(task_df)
        exclude_all_df = task_df[task_df["exclude_from_all_analysis"].astype(str).str.lower().isin(["true", "1"])] if not task_df.empty else pd.DataFrame()

        cluster_df.to_csv(self.output_dir / "k258_cluster_manual_review_summary.csv", index=False, encoding="utf-8-sig", sep=";")
        task_df.to_csv(self.output_dir / "k258_task_occurrence_manual_membership_review.csv", index=False, encoding="utf-8-sig", sep=";")
        reclass_df.to_csv(self.output_dir / "k258_reclassified_task_groups.csv", index=False, encoding="utf-8-sig", sep=";")
        exclude_all_df.to_csv(self.output_dir / "k258_tasks_excluded_from_all_analysis.csv", index=False, encoding="utf-8-sig", sep=";")

        with pd.ExcelWriter(self.output_dir / "k258_task_cluster_manual_review.xlsx", engine="openpyxl") as writer:
            cluster_df.to_excel(writer, sheet_name="cluster_summary", index=False)
            task_df.to_excel(writer, sheet_name="task_membership", index=False)
            reclass_df.to_excel(writer, sheet_name="reclassification_groups", index=False)
            exclude_all_df.to_excel(writer, sheet_name="excluded_from_all", index=False)

        with (self.output_dir / "k258_task_occurrence_manual_membership_review.jsonl").open("w", encoding="utf-8") as file_obj:
            for record in task_records:
                file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _build_reclassification_groups(self, task_df: Any) -> Any:
        import pandas as pd

        if task_df.empty or "membership_decision" not in task_df.columns:
            return pd.DataFrame()
        subset = task_df[task_df["membership_decision"] == DECISION_RECLASSIFY].copy()
        if subset.empty:
            return pd.DataFrame()
        subset["reclassified_task_name"] = subset["reclassified_task_name"].fillna("").map(normalize_task_name)
        subset = subset[subset["reclassified_task_name"] != ""].copy()
        rows: List[Dict[str, Any]] = []
        for task_name, group in subset.groupby("reclassified_task_name", dropna=False):
            rows.append(
                {
                    "reclassified_task_name": task_name,
                    "occurrence_count": int(len(group)),
                    "article_count": int(group["PMID"].astype(str).nunique()) if "PMID" in group else "",
                    "source_groups": ",".join(sorted(set(group.get("source_group", pd.Series(dtype=str)).astype(str)))),
                    "PMIDs": ",".join(group.get("PMID", pd.Series(dtype=str)).astype(str).drop_duplicates().tolist()),
                    "task_indices": ",".join(group.get("task_index", pd.Series(dtype=str)).astype(str).tolist()),
                    "source_cluster_ids": ",".join(sorted(set(group.get("cluster_id", pd.Series(dtype=str)).astype(str)))),
                    "representative_task_names": " | ".join(group.get("structured_task_name", pd.Series(dtype=str)).astype(str).drop_duplicates().head(12).tolist()),
                    "representative_descriptions": " | ".join(group.get("description", pd.Series(dtype=str)).astype(str).drop_duplicates().head(8).tolist()),
                    "representative_clue_sentences": " | ".join(group.get("clue_sentences", pd.Series(dtype=str)).astype(str).drop_duplicates().head(8).tolist()),
                }
            )
        return pd.DataFrame(rows).sort_values(["occurrence_count", "reclassified_task_name"], ascending=[False, True])

    def _save_and_next(self) -> None:
        if self._save_current_cluster(show_message=False):
            self._next_cluster()

    def _previous_cluster(self) -> None:
        self._save_temp_current_cluster()
        if self.index > 0:
            self.index -= 1
            self._render_cluster()

    def _next_cluster(self) -> None:
        self._save_temp_current_cluster()
        if self.index < len(self.cluster_ids) - 1:
            self.index += 1
            self._render_cluster()

    def _jump_to_cluster(self) -> None:
        if self.jump_var is None:
            return
        self._save_temp_current_cluster()
        try:
            cluster_id = int(self.jump_var.get())
        except Exception:
            return
        if cluster_id in self.cluster_ids:
            self.index = self.cluster_ids.index(cluster_id)
            self._render_cluster()

    def _on_close(self) -> None:
        self._save_temp_current_cluster()
        if self.root is not None:
            self.root.destroy()


def review_task_cluster_membership(config: TaskClusterReviewConfig) -> None:
    reviewer = TaskClusterMembershipReviewer(config)
    reviewer.run()
