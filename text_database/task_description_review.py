"""Prepare a fixed random sample and UI for reviewing task-description quality."""

from __future__ import annotations

import csv
import json
import random
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def compact(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def unique_values(values: List[Any]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        text = compact(value)
        key = text.casefold()
        if text and key not in seen:
            seen.add(key)
            result.append(text)
    return result


def preferred_display_name(rows: List[sqlite3.Row]) -> str:
    names = Counter(compact(row["final_task_name"]) for row in rows)
    names.pop("", None)
    return sorted(names.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[0][0] if names else ""


def prepare_task_description_sample(
    database_path: Path,
    output_dir: Path,
    sample_size: int = 100,
    description_samples_per_task: int = 10,
    random_seed: int = 20260721,
    force: bool = False,
) -> Dict[str, Any]:
    """Sample distinct task names, retaining provenance for every displayed description."""

    database_path = Path(database_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = output_dir / "task_description_quality_sample.json"
    if sample_path.exists() and not force:
        return json.loads(sample_path.read_text(encoding="utf-8"))
    if not database_path.exists():
        raise FileNotFoundError(f"SQLite database not found: {database_path}")

    with sqlite3.connect(str(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT occurrence_key, pmid, task_index, source_group, final_task_name,
                   normalized_task_name, description, normalized_task_description,
                   clue_sentences, exclude_from_all_analysis
            FROM task_occurrences
            WHERE normalized_task_name<>'' AND exclude_from_all_analysis=0
            ORDER BY normalized_task_name, pmid, task_index, occurrence_key
            """
        ).fetchall()
    groups: Dict[str, List[sqlite3.Row]] = defaultdict(list)
    for row in rows:
        groups[str(row["normalized_task_name"])].append(row)

    task_keys = sorted(groups)
    selected_keys = sorted(random.Random(random_seed).sample(task_keys, min(max(1, int(sample_size)), len(task_keys))))
    tasks: List[Dict[str, Any]] = []
    for task_key in selected_keys:
        group = groups[task_key]
        descriptions: List[Dict[str, Any]] = []
        seen_descriptions = set()
        for row in group:
            text = compact(row["description"]) or compact(row["normalized_task_description"])
            text_key = text.casefold()
            if not text or text_key in seen_descriptions:
                continue
            seen_descriptions.add(text_key)
            descriptions.append(
                {
                    "description": text,
                    "PMID": str(row["pmid"]),
                    "task_index": str(row["task_index"]),
                    "source_group": str(row["source_group"] or ""),
                    "occurrence_key": str(row["occurrence_key"]),
                    "clue_sentence": compact(row["clue_sentences"]),
                }
            )
        task_seed = random_seed + sum(ord(char) for char in task_key)
        samples = random.Random(task_seed).sample(descriptions, min(description_samples_per_task, len(descriptions))) if descriptions else []
        tasks.append(
            {
                "task_review_key": task_key,
                "normalized_task_name": task_key,
                "display_task_name": preferred_display_name(group),
                "task_occurrence_count": len(group),
                "article_count": len({str(row["pmid"]) for row in group}),
                "description_count": len(descriptions),
                "original_name_variants": unique_values([row["final_task_name"] for row in group]),
                "description_samples": samples,
            }
        )
    payload = {
        "generated_at": now_iso(),
        "database_path": str(database_path),
        "random_seed": random_seed,
        "unique_task_count_in_database": len(task_keys),
        "sample_size": len(tasks),
        "description_samples_per_task": description_samples_per_task,
        "tasks": tasks,
    }
    sample_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_rows = []
    for task in tasks:
        primary = task["description_samples"][0] if task["description_samples"] else {}
        audit_rows.append(
            {
                "normalized_task_name": task["normalized_task_name"],
                "display_task_name": task["display_task_name"],
                "task_occurrence_count": task["task_occurrence_count"],
                "article_count": task["article_count"],
                "unique_description_count": task["description_count"],
                "primary_description": primary.get("description", ""),
                "primary_description_PMID": primary.get("PMID", ""),
                "primary_description_task_index": primary.get("task_index", ""),
            }
        )
    audit_path = output_dir / "task_description_quality_sample_audit.csv"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(audit_rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(audit_rows)
    return payload


class TaskDescriptionReviewUI:
    """One-task-at-a-time review UI, with autosave and per-task provenance."""

    QUALITY_LABELS = {
        "": "Not reviewed",
        "sufficient_accurate": "Sufficient and accurate for semantic embedding",
        "partly_accurate": "Partly accurate but missing important task detail",
        "insufficient_or_inaccurate": "Insufficient or inaccurate; do not embed as-is",
        "no_usable_description": "No usable task description",
    }

    def __init__(self, sample_path: Path, output_dir: Path) -> None:
        self.sample_path = Path(sample_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.payload = json.loads(self.sample_path.read_text(encoding="utf-8"))
        self.tasks: List[Dict[str, Any]] = self.payload.get("tasks", [])
        if not self.tasks:
            raise RuntimeError(f"No task records in sample: {self.sample_path}")
        self.autosave_path = self.output_dir / "task_description_quality_autosave.json"
        self.final_json_path = self.output_dir / "task_description_quality_review_results.json"
        self.final_csv_path = self.output_dir / "task_description_quality_review_results.csv"
        self.reviews = self._load_reviews()
        self.index = self._first_unreviewed_index()
        self.root = None
        self.tk = None
        self.ttk = None
        self.messagebox = None
        self.status_var = None
        self.autosave_var = None
        self.jump_var = None
        self.sample_var = None
        self.quality_var = None
        self.note_box = None
        self.description_box = None
        self.clue_box = None
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
        return {str(row.get("task_review_key", "")): row for row in payload.get("task_reviews", []) if row.get("task_review_key")}

    def _first_unreviewed_index(self) -> int:
        for index, task in enumerate(self.tasks):
            if not self.reviews.get(task["task_review_key"], {}).get("description_quality", ""):
                return index
        return 0

    def run(self) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.root = tk.Tk()
        self.root.title("ARneuro Task Description Quality Review")
        self.root.geometry("1440x880")
        self.root.minsize(1080, 720)
        self.root.configure(bg="#f4f7fb")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._configure_style()
        self._build_shell()
        self._render_task()
        self.root.mainloop()

    def _configure_style(self) -> None:
        style = self.ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("App.TFrame", background="#f4f7fb")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Header.TLabel", background="#f4f7fb", foreground="#16243b", font=("Microsoft YaHei", 16, "bold"))
        style.configure("Title.TLabel", background="#ffffff", foreground="#152238", font=("Microsoft YaHei", 14, "bold"))
        style.configure("Body.TLabel", background="#ffffff", foreground="#334155", font=("Microsoft YaHei", 10))
        style.configure("Muted.TLabel", background="#ffffff", foreground="#64748b", font=("Microsoft YaHei", 9))
        style.configure("Metric.TLabel", background="#eaf2fb", foreground="#1e4d75", font=("Microsoft YaHei", 9, "bold"), padding=(8, 4))
        style.configure("Primary.TButton", font=("Microsoft YaHei", 10, "bold"), padding=(10, 6))

    def _build_shell(self) -> None:
        tk, ttk = self.tk, self.ttk
        outer = ttk.Frame(self.root, style="App.TFrame", padding=(18, 14, 18, 14))
        outer.pack(fill="both", expand=True)
        top = ttk.Frame(outer, style="App.TFrame")
        top.pack(fill="x", pady=(0, 10))
        ttk.Label(top, text="Task description quality review", style="Header.TLabel").pack(side="left")
        controls = ttk.Frame(top, style="App.TFrame")
        controls.pack(side="right")
        ttk.Button(controls, text="Previous", command=self._previous).pack(side="left", padx=3)
        ttk.Button(controls, text="Save", command=lambda: self._save(True), style="Primary.TButton").pack(side="left", padx=3)
        ttk.Button(controls, text="Save + next", command=self._save_and_next, style="Primary.TButton").pack(side="left", padx=3)
        ttk.Button(controls, text="Next", command=self._next).pack(side="left", padx=3)
        nav = ttk.Frame(outer, style="App.TFrame")
        nav.pack(fill="x", pady=(0, 10))
        self.status_var = tk.StringVar(value="")
        self.autosave_var = tk.StringVar(value="")
        ttk.Label(nav, textvariable=self.status_var, style="Header.TLabel", font=("Microsoft YaHei", 10, "bold")).pack(side="left")
        ttk.Label(nav, textvariable=self.autosave_var, style="Header.TLabel", font=("Microsoft YaHei", 9)).pack(side="left", padx=(16, 0))
        ttk.Label(nav, text="Jump:", background="#f4f7fb", font=("Microsoft YaHei", 10)).pack(side="left", padx=(22, 6))
        self.jump_var = tk.StringVar()
        jump = ttk.Combobox(nav, textvariable=self.jump_var, width=52, state="readonly")
        jump["values"] = [task["normalized_task_name"] for task in self.tasks]
        jump.bind("<<ComboboxSelected>>", lambda _event: self._jump())
        jump.pack(side="left")
        self.content = ttk.Frame(outer, style="Panel.TFrame", padding=(18, 16, 18, 16))
        self.content.pack(fill="both", expand=True)

    def _current_task(self) -> Dict[str, Any]:
        return self.tasks[self.index]

    @staticmethod
    def _clear(frame: Any) -> None:
        for child in frame.winfo_children():
            child.destroy()

    def _render_task(self) -> None:
        tk, ttk = self.tk, self.ttk
        self.rendering = True
        self._clear(self.content)
        task = self._current_task()
        existing = self.reviews.get(task["task_review_key"], {})
        self.jump_var.set(task["normalized_task_name"])
        reviewed_count = sum(bool(self.reviews.get(item["task_review_key"], {}).get("description_quality", "")) for item in self.tasks)
        self.status_var.set(f"Task {self.index + 1}/{len(self.tasks)} | reviewed {reviewed_count}/{len(self.tasks)}")
        self.autosave_var.set(f"Autosave: {self.autosave_path.name}")
        ttk.Label(self.content, text=task["display_task_name"] or task["normalized_task_name"], style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(self.content, text=f"Normalized lookup key: {task['normalized_task_name']}", style="Muted.TLabel").grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 10))
        metrics = ttk.Frame(self.content, style="Panel.TFrame")
        metrics.grid(row=2, column=0, columnspan=2, sticky="w", pady=(0, 12))
        for label, value in (
            ("Task occurrences", task["task_occurrence_count"]),
            ("Articles", task["article_count"]),
            ("Unique descriptions", task["description_count"]),
            ("Shown samples", len(task["description_samples"])),
        ):
            ttk.Label(metrics, text=f"{label}: {value}", style="Metric.TLabel").pack(side="left", padx=(0, 6))
        ttk.Label(self.content, text="Original display-name variants", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Label(self.content, text=" | ".join(task["original_name_variants"]), style="Muted.TLabel", wraplength=1320).grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 10))
        ttk.Label(self.content, text="Description sample", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=5, column=0, sticky="w")
        self.sample_var = tk.StringVar()
        sample_selector = ttk.Combobox(self.content, textvariable=self.sample_var, width=64, state="readonly")
        sample_selector["values"] = [
            f"{index + 1} | PMID {sample['PMID']} | task {sample['task_index']} | {sample['source_group'] or 'source unknown'}"
            for index, sample in enumerate(task["description_samples"])
        ] or ["No extracted description"]
        sample_selector.current(0)
        sample_selector.bind("<<ComboboxSelected>>", lambda _event: self._render_description_sample())
        sample_selector.grid(row=5, column=1, sticky="w", padx=(10, 0))
        self.description_box = tk.Text(self.content, height=16, wrap="word", font=("Consolas", 10), bg="#0f172a", fg="#e2e8f0", insertbackground="#ffffff", relief="flat", padx=12, pady=10)
        self.description_box.grid(row=6, column=0, columnspan=2, sticky="nsew", pady=(5, 12))
        ttk.Label(self.content, text="Associated clue sentence", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=7, column=0, columnspan=2, sticky="w")
        self.clue_box = tk.Text(self.content, height=5, wrap="word", font=("Consolas", 9), bg="#f7fafc", fg="#1e293b", relief="solid", borderwidth=1, padx=8, pady=7)
        self.clue_box.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(3, 12))
        review = ttk.Frame(self.content, style="Panel.TFrame")
        review.grid(row=9, column=0, columnspan=2, sticky="ew")
        review.columnconfigure(3, weight=1)
        ttk.Label(review, text="Description quality", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.quality_var = tk.StringVar(value=str(existing.get("description_quality", "")))
        quality_selector = ttk.Combobox(review, textvariable=self.quality_var, width=62, state="readonly")
        quality_selector["values"] = [f"{key} | {label}" if key else label for key, label in self.QUALITY_LABELS.items()]
        current = self.quality_var.get()
        quality_selector.set(f"{current} | {self.QUALITY_LABELS[current]}" if current else self.QUALITY_LABELS[""])
        quality_selector.bind("<<ComboboxSelected>>", lambda _event: self._schedule_autosave())
        quality_selector.grid(row=0, column=1, sticky="w")
        ttk.Label(review, text="Notes", style="Body.TLabel", font=("Microsoft YaHei", 10, "bold")).grid(row=1, column=0, sticky="nw", pady=(10, 0))
        self.note_box = tk.Text(review, height=4, wrap="word", font=("Microsoft YaHei", 9), bg="#ffffff", relief="solid", borderwidth=1)
        self.note_box.insert("1.0", str(existing.get("reviewer_note", "")))
        self.note_box.bind("<KeyRelease>", lambda _event: self._schedule_autosave())
        self.note_box.bind("<FocusOut>", lambda _event: self._schedule_autosave())
        self.note_box.grid(row=1, column=1, columnspan=3, sticky="ew", pady=(10, 0))
        self.content.columnconfigure(1, weight=1)
        self.content.rowconfigure(6, weight=1)
        self._render_description_sample()
        self.rendering = False

    def _render_description_sample(self) -> None:
        task = self._current_task()
        selected = self.sample_var.get()
        try:
            index = int(selected.split(" | ", 1)[0]) - 1
        except Exception:
            index = 0
        samples = task["description_samples"]
        sample = samples[index] if 0 <= index < len(samples) else {}
        self.description_box.configure(state="normal")
        self.description_box.delete("1.0", "end")
        self.description_box.insert("1.0", sample.get("description", "No extracted description is available for this task."))
        self.description_box.configure(state="disabled")
        self.clue_box.configure(state="normal")
        self.clue_box.delete("1.0", "end")
        self.clue_box.insert("1.0", sample.get("clue_sentence", "No clue sentence recorded."))
        self.clue_box.configure(state="disabled")

    @staticmethod
    def _choice_key(value: Any, labels: Dict[str, str]) -> str:
        text = str(value or "")
        if text in labels:
            return text
        if text == labels.get("", ""):
            return ""
        key = text.split(" | ", 1)[0] if " | " in text else ""
        return key if key in labels else ""

    def _collect_current(self) -> None:
        task = self._current_task()
        self.reviews[task["task_review_key"]] = {
            "task_review_key": task["task_review_key"],
            "normalized_task_name": task["normalized_task_name"],
            "display_task_name": task["display_task_name"],
            "description_quality": self._choice_key(self.quality_var.get(), self.QUALITY_LABELS),
            "reviewed_description_sample": self.sample_var.get(),
            "reviewer_note": self.note_box.get("1.0", "end-1c").strip(),
            "reviewed_at": now_iso(),
        }

    def _payload(self) -> Dict[str, Any]:
        return {
            "generated_at": now_iso(),
            "sample_path": str(self.sample_path),
            "task_reviews": sorted(self.reviews.values(), key=lambda row: row["normalized_task_name"]),
        }

    def _write(self, final: bool) -> None:
        self._collect_current()
        payload = self._payload()
        (self.final_json_path if final else self.autosave_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if final:
            rows = payload["task_reviews"]
            with self.final_csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
                writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]) if rows else ["task_review_key"], delimiter=";")
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
        self._write(False)
        self.autosave_var.set(f"Autosaved {datetime.now().strftime('%H:%M:%S')}")

    def _save(self, show_message: bool) -> None:
        self._write(True)
        self._write(False)
        if show_message:
            self.messagebox.showinfo("Saved", f"Review saved to:\n{self.final_csv_path}")

    def _save_and_next(self) -> None:
        self._save(False)
        self._next()

    def _previous(self) -> None:
        self._write(False)
        self.index = max(0, self.index - 1)
        self._render_task()

    def _next(self) -> None:
        self._write(False)
        self.index = min(len(self.tasks) - 1, self.index + 1)
        self._render_task()

    def _jump(self) -> None:
        key = self.jump_var.get()
        for index, task in enumerate(self.tasks):
            if task["normalized_task_name"] == key:
                self._write(False)
                self.index = index
                self._render_task()
                return

    def _on_close(self) -> None:
        self._write(False)
        self.root.destroy()


def launch_task_description_review(sample_path: Path, output_dir: Path) -> None:
    TaskDescriptionReviewUI(sample_path=sample_path, output_dir=output_dir).run()
