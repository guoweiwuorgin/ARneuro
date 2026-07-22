"""
Interactive Markdown issue reviewer for incomplete segmentation cases.

This module opens a desktop GUI that displays source Markdown for papers whose
Methods and/or Results sections are missing after segmentation. Reviewers can
label why the segmentation failed using predefined categories or a custom note.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ISSUE_TYPES = [
    {
        "code": "pdf_conversion_failed",
        "label": "1. PDF 转换失败",
        "description": "Markdown 明显残缺、乱码严重、正文缺页，或只剩参考文献/页眉页脚。",
    },
    {
        "code": "method_title_exists_but_segmentation_failed",
        "label": "2. Method title 存在但分割失败",
        "description": "能看到 Methods/Subjects/Procedure 等标题，但现有逻辑没正确归入 Methods。",
    },
    {
        "code": "no_explicit_section_markers",
        "label": "3. 缺乏明确标记",
        "description": "正文是连续叙述，没有清晰的 Methods 或 Results 标题。",
    },
    {
        "code": "non_study_review_or_other",
        "label": "4. 非 study 研究",
        "description": "文章是 review、commentary、editorial、protocol、case note 或其他非实证研究。",
    },
    {
        "code": "other",
        "label": "5. 其他原因",
        "description": "需要手动输入原因。",
    },
]


@dataclass
class MarkdownReviewItem:
    pmid: str
    markdown_path: Path
    missing_methods: bool = False
    missing_results: bool = False
    methods_len: int = 0
    results_len: int = 0
    content_path: str = ""
    meta_path: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def read_existing_annotations(path: Path) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}

    annotations: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file_obj:
        for line in file_obj:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid = str(record.get("pmid", "")).strip()
            if pmid:
                annotations[pmid] = record
    return annotations


def append_jsonl(record: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file_obj:
        file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8-sig")
        return

    fieldnames: List[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_missing_methods_results_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def candidate_markdown_paths(
    pmid: str,
    record: Dict[str, str],
    search_roots: Iterable[Path],
) -> List[Path]:
    candidates: List[Path] = []

    content_path = Path(record.get("content_path", ""))
    if content_path.exists():
        meta_path = content_path.with_name(
            content_path.name.replace("_structured_content.json", "_structured_meta.json")
        )
        if meta_path.exists():
            try:
                meta = read_json(meta_path)
                for nested_key in ("source_file",):
                    source = meta.get(nested_key, "")
                    if source:
                        candidates.append(Path(source))
                for container_key in ("supplement_meta", "original_meta"):
                    nested = meta.get(container_key, {})
                    source = nested.get("source_file", "") if isinstance(nested, dict) else ""
                    if source:
                        candidates.append(Path(source))
            except Exception:
                pass

    file_name = f"paper_{pmid}.md"
    for root in search_roots:
        if not root.exists():
            continue
        direct = root / file_name
        candidates.append(direct)
        for found in root.rglob(file_name):
            candidates.append(found)

    unique: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            unique.append(candidate)
    return unique


def build_review_items_from_missing_report(
    missing_report_csv: Path,
    search_roots: Iterable[Path],
) -> List[MarkdownReviewItem]:
    rows = load_missing_methods_results_csv(missing_report_csv)
    items: List[MarkdownReviewItem] = []

    for row in rows:
        pmid = str(row.get("pmid", "")).strip()
        if not pmid:
            continue
        candidates = candidate_markdown_paths(pmid, row, search_roots)
        markdown_path = candidates[0] if candidates else Path(f"paper_{pmid}.md")
        content_path = row.get("content_path", "")
        meta_path = ""
        if content_path:
            meta_path = str(Path(content_path).with_name(
                Path(content_path).name.replace("_structured_content.json", "_structured_meta.json")
            ))

        items.append(
            MarkdownReviewItem(
                pmid=pmid,
                markdown_path=markdown_path,
                missing_methods=truthy(row.get("missing_methods", "")),
                missing_results=truthy(row.get("missing_results", "")),
                methods_len=int(row.get("methods_len") or 0),
                results_len=int(row.get("results_len") or 0),
                content_path=content_path,
                meta_path=meta_path,
                extra={"candidate_count": len(candidates)},
            )
        )

    return items


class MarkdownIssueReviewer:
    """Tkinter GUI for reviewing Markdown segmentation failure causes."""

    def __init__(
        self,
        items: List[MarkdownReviewItem],
        output_jsonl: Path,
        output_csv: Optional[Path] = None,
        title: str = "ARneuro Markdown 分割问题标注",
    ) -> None:
        if not items:
            raise ValueError("No MarkdownReviewItem entries were provided.")

        self.items = items
        self.output_jsonl = output_jsonl
        self.output_csv = output_csv or output_jsonl.with_suffix(".csv")
        self.title = title
        self.annotations = read_existing_annotations(output_jsonl)
        self.index = self._first_unreviewed_index()

        self.root = None
        self.issue_var = None
        self.other_text = None
        self.markdown_text = None
        self.status_label = None
        self.header_label = None
        self.path_label = None
        self.missing_label = None

    def _first_unreviewed_index(self) -> int:
        for idx, item in enumerate(self.items):
            if item.pmid not in self.annotations:
                return idx
        return 0

    def run(self) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk
        from tkinter.scrolledtext import ScrolledText

        self.tk = tk
        self.messagebox = messagebox
        self.ttk = ttk
        self.ScrolledText = ScrolledText

        root = tk.Tk()
        self.root = root
        root.title(self.title)
        root.geometry("1280x820")
        root.minsize(980, 680)
        root.configure(bg="#f4f6fb")

        self._configure_styles()

        outer = ttk.Frame(root, style="App.TFrame", padding=(18, 16, 18, 16))
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer, style="Header.TFrame", padding=(18, 14, 18, 14))
        header.pack(fill="x")

        self.header_label = ttk.Label(
            header,
            text="",
            style="HeaderTitle.TLabel",
        )
        self.header_label.pack(anchor="w")

        self.missing_label = ttk.Label(header, text="", style="HeaderMeta.TLabel")
        self.missing_label.pack(anchor="w", pady=(6, 0))

        self.path_label = ttk.Label(header, text="", style="HeaderPath.TLabel")
        self.path_label.pack(anchor="w", pady=(4, 0))

        body = ttk.Frame(outer, style="App.TFrame")
        body.pack(fill="both", expand=True, pady=(14, 0))

        left = ttk.Frame(body, style="Panel.TFrame", padding=(12, 12, 12, 12))
        left.pack(side="left", fill="both", expand=True)

        right = ttk.Frame(body, style="Panel.TFrame", padding=(14, 12, 14, 12))
        right.pack(side="right", fill="y", padx=(14, 0))

        ttk.Label(left, text="Markdown 原文", style="PanelTitle.TLabel").pack(anchor="w")
        self.markdown_text = ScrolledText(
            left,
            wrap="word",
            font=("Consolas", 12),
            bg="#ffffff",
            fg="#182235",
            insertbackground="#182235",
            relief="flat",
            borderwidth=0,
            padx=14,
            pady=12,
        )
        self.markdown_text.pack(fill="both", expand=True, pady=(10, 0))

        ttk.Label(right, text="请选择导致检测不到的类型", style="PanelTitle.TLabel").pack(anchor="w")
        self.issue_var = tk.StringVar(value=ISSUE_TYPES[0]["code"])

        for issue in ISSUE_TYPES:
            frame = ttk.Frame(right, style="Choice.TFrame", padding=(10, 8, 10, 8))
            frame.pack(fill="x", pady=(10, 0))
            rb = ttk.Radiobutton(
                frame,
                text=issue["label"],
                value=issue["code"],
                variable=self.issue_var,
                style="Choice.TRadiobutton",
            )
            rb.pack(anchor="w")
            ttk.Label(
                frame,
                text=issue["description"],
                style="ChoiceDesc.TLabel",
                wraplength=330,
            ).pack(anchor="w", pady=(4, 0))

        ttk.Label(right, text="其他原因 / 备注", style="PanelTitle.TLabel").pack(anchor="w", pady=(18, 0))
        self.other_text = tk.Text(
            right,
            height=5,
            wrap="word",
            font=("Microsoft YaHei UI", 10),
            bg="#ffffff",
            fg="#182235",
            relief="flat",
            padx=10,
            pady=8,
        )
        self.other_text.pack(fill="x", pady=(8, 0))

        button_row = ttk.Frame(right, style="Panel.TFrame")
        button_row.pack(fill="x", pady=(18, 0))

        ttk.Button(button_row, text="上一篇", command=self.previous_item, style="Soft.TButton").pack(side="left")
        ttk.Button(button_row, text="跳过", command=self.skip_item, style="Soft.TButton").pack(side="left", padx=(8, 0))
        ttk.Button(button_row, text="保存并下一篇", command=self.save_and_next, style="Primary.TButton").pack(side="right")

        footer = ttk.Frame(outer, style="App.TFrame")
        footer.pack(fill="x", pady=(10, 0))
        self.status_label = ttk.Label(footer, text="", style="Status.TLabel")
        self.status_label.pack(side="left")
        ttk.Button(footer, text="导出 CSV", command=self.export_csv, style="Soft.TButton").pack(side="right")

        root.bind("<Control-s>", lambda _event: self.save_and_next())
        root.bind("<Alt-Right>", lambda _event: self.save_and_next())
        root.bind("<Alt-Left>", lambda _event: self.previous_item())

        self.render_current_item()
        root.mainloop()

    def _configure_styles(self) -> None:
        ttk = self.ttk
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background="#f4f6fb")
        style.configure("Header.TFrame", background="#1f3a5f")
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Choice.TFrame", background="#f7f9fd")
        style.configure("HeaderTitle.TLabel", background="#1f3a5f", foreground="#ffffff", font=("Microsoft YaHei UI", 16, "bold"))
        style.configure("HeaderMeta.TLabel", background="#1f3a5f", foreground="#dbe8ff", font=("Microsoft YaHei UI", 10))
        style.configure("HeaderPath.TLabel", background="#1f3a5f", foreground="#b8c8e6", font=("Consolas", 9))
        style.configure("PanelTitle.TLabel", background="#ffffff", foreground="#182235", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Choice.TRadiobutton", background="#f7f9fd", foreground="#182235", font=("Microsoft YaHei UI", 10, "bold"))
        style.configure("ChoiceDesc.TLabel", background="#f7f9fd", foreground="#596579", font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", background="#f4f6fb", foreground="#596579", font=("Microsoft YaHei UI", 9))
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(12, 8))
        style.configure("Soft.TButton", font=("Microsoft YaHei UI", 10), padding=(10, 7))

    def current_item(self) -> MarkdownReviewItem:
        return self.items[self.index]

    def render_current_item(self) -> None:
        item = self.current_item()
        reviewed_count = len(self.annotations)
        missing_parts = []
        if item.missing_methods:
            missing_parts.append("Methods")
        if item.missing_results:
            missing_parts.append("Results")
        missing_text = ", ".join(missing_parts) if missing_parts else "None"

        self.header_label.configure(
            text=f"PMID {item.pmid}   ({self.index + 1}/{len(self.items)})"
        )
        self.missing_label.configure(
            text=f"缺失字段: {missing_text}    Methods长度: {item.methods_len}    Results长度: {item.results_len}"
        )
        self.path_label.configure(text=str(item.markdown_path))
        self.status_label.configure(
            text=f"已标注 {reviewed_count}/{len(self.items)}；Ctrl+S 保存并下一篇，Alt+←/→ 切换"
        )

        annotation = self.annotations.get(item.pmid)
        if annotation:
            self.issue_var.set(annotation.get("issue_code", ISSUE_TYPES[0]["code"]))
            self.other_text.delete("1.0", "end")
            self.other_text.insert("1.0", annotation.get("note", ""))
        else:
            self.issue_var.set(ISSUE_TYPES[0]["code"])
            self.other_text.delete("1.0", "end")

        markdown = self.read_markdown_for_item(item)
        self.markdown_text.configure(state="normal")
        self.markdown_text.delete("1.0", "end")
        self.markdown_text.insert("1.0", markdown)
        self.markdown_text.configure(state="disabled")
        self.markdown_text.yview_moveto(0)

    def read_markdown_for_item(self, item: MarkdownReviewItem) -> str:
        if item.markdown_path.exists():
            try:
                return item.markdown_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                return item.markdown_path.read_text(encoding="utf-8", errors="replace")

        return (
            f"Markdown file not found:\n{item.markdown_path}\n\n"
            "请检查 merged meta 中的 source_file，或在示例脚本里增加 markdown 搜索目录。"
        )

    def build_annotation_record(self, skipped: bool = False) -> Dict[str, Any]:
        item = self.current_item()
        issue_code = self.issue_var.get()
        issue = next((entry for entry in ISSUE_TYPES if entry["code"] == issue_code), ISSUE_TYPES[-1])
        note = self.other_text.get("1.0", "end").strip()

        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pmid": item.pmid,
            "issue_code": "skipped" if skipped else issue["code"],
            "issue_label": "跳过" if skipped else issue["label"],
            "note": note,
            "markdown_path": str(item.markdown_path),
            "missing_methods": item.missing_methods,
            "missing_results": item.missing_results,
            "methods_len": item.methods_len,
            "results_len": item.results_len,
            "content_path": item.content_path,
            "meta_path": item.meta_path,
        }

    def save_current(self, skipped: bool = False) -> None:
        record = self.build_annotation_record(skipped=skipped)
        self.annotations[record["pmid"]] = record
        append_jsonl(record, self.output_jsonl)
        self.export_csv(silent=True)

    def save_and_next(self) -> None:
        if self.issue_var.get() == "other" and not self.other_text.get("1.0", "end").strip():
            self.messagebox.showwarning("需要备注", "选择“其他原因”时，请输入简短说明。")
            return
        self.save_current(skipped=False)
        self.next_item()

    def skip_item(self) -> None:
        self.save_current(skipped=True)
        self.next_item()

    def next_item(self) -> None:
        if self.index < len(self.items) - 1:
            self.index += 1
            self.render_current_item()
            return
        self.messagebox.showinfo("完成", "已经到最后一篇。标注结果已保存。")
        self.render_current_item()

    def previous_item(self) -> None:
        if self.index > 0:
            self.index -= 1
            self.render_current_item()

    def export_csv(self, silent: bool = False) -> None:
        records = [
            self.annotations[pmid]
            for pmid in sorted(self.annotations.keys(), key=lambda value: (not value.isdigit(), value))
        ]
        write_csv(records, self.output_csv)
        if not silent:
            self.messagebox.showinfo("已导出", f"CSV 已保存到:\n{self.output_csv}")


def review_missing_methods_results(
    missing_report_csv: Path,
    output_jsonl: Path,
    search_roots: Iterable[Path],
    output_csv: Optional[Path] = None,
) -> None:
    """Load a missing-section report and launch the GUI reviewer."""
    items = build_review_items_from_missing_report(
        missing_report_csv=missing_report_csv,
        search_roots=search_roots,
    )
    reviewer = MarkdownIssueReviewer(
        items=items,
        output_jsonl=output_jsonl,
        output_csv=output_csv,
    )
    reviewer.run()
