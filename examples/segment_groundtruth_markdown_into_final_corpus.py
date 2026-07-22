"""
Segment ground-truth Markdown papers missing from the final Step3 corpus.

The script:
    - scans Paper_Mete_Ground_truth_markdown/paper_{PMID}.md;
    - reuses valid existing final content/meta pairs;
    - repairs existing pairs missing Methods or Results;
    - segments absent papers with ARneuro FinalSeg rules plus DeepSeek fallback;
    - writes canonical files to final_segmented_for_analysis/full_segmented;
    - mirrors this 376-paper evaluation set in groundtruth_segmented;
    - writes a detailed resumable report.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
ARNEURO_DIR = STEP3_DIR / "ARneuro"
MARKDOWN_DIR = STEP3_DIR / "Paper_Mete_Ground_truth_markdown"
FINAL_ROOT = STEP3_DIR / "final_segmented_for_analysis"
FULL_SEGMENTED_DIR = FINAL_ROOT / "full_segmented"
GROUNDTRUTH_SEGMENTED_DIR = FINAL_ROOT / "groundtruth_segmented"
REPORT_DIR = FINAL_ROOT / "reports" / "groundtruth_markdown_import"
BACKUP_DIR = REPORT_DIR / "replaced_existing_backups"
DEFAULT_API_KEY = "random-placeholder-ecd0c598d8d2d6f57d3ec57c5d256700"

for import_path in [ARNEURO_DIR, ARNEURO_DIR / "text_processing", STEP3_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

MODULE_PATH = ARNEURO_DIR / "text_processing" / "final_segmentation.py"
spec = importlib.util.spec_from_file_location(
    "groundtruth_final_segmentation",
    MODULE_PATH,
)
final_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = final_module
spec.loader.exec_module(final_module)
FinalSeg = final_module.FinalSeg
has_meaningful_section = final_module.has_meaningful_section


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import and segment 376 ground-truth Markdown papers."
    )
    parser.add_argument("--markdown-dir", default=str(MARKDOWN_DIR))
    parser.add_argument("--final-root", default=str(FINAL_ROOT))
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY", DEFAULT_API_KEY),
    )
    parser.add_argument(
        "--overwrite-all",
        action="store_true",
        help="Re-segment even valid existing content/meta pairs.",
    )
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def output_paths(directory: Path, pmid: str) -> tuple[Path, Path]:
    return (
        directory / f"paper_{pmid}_structured_content.json",
        directory / f"paper_{pmid}_structured_meta.json",
    )


def existing_state(content_path: Path, meta_path: Path) -> Dict[str, Any]:
    state = {
        "files_complete": content_path.exists() and meta_path.exists(),
        "json_valid": False,
        "has_methods": False,
        "has_results": False,
        "content": {},
        "meta": {},
        "error": "",
    }
    if not state["files_complete"]:
        return state
    try:
        content = read_json(content_path)
        meta = read_json(meta_path)
        state.update(
            {
                "json_valid": True,
                "has_methods": has_meaningful_section(content.get("Methods")),
                "has_results": has_meaningful_section(content.get("Results")),
                "content": content,
                "meta": meta,
            }
        )
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
    return state


def backup_existing(
    pmid: str,
    content_path: Path,
    meta_path: Path,
    backup_dir: Path,
) -> None:
    if content_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            content_path,
            backup_dir / f"paper_{pmid}_structured_content.json",
        )
    if meta_path.exists():
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            meta_path,
            backup_dir / f"paper_{pmid}_structured_meta.json",
        )


def write_csv(rows: list[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "pmid",
        "status",
        "previous_state",
        "has_methods",
        "has_results",
        "methods_chars",
        "results_chars",
        "llm_client",
        "llm_model",
        "llm_error",
        "source_markdown",
        "canonical_content",
        "canonical_meta",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> None:
    args = parse_args()
    markdown_dir = Path(args.markdown_dir)
    final_root = Path(args.final_root)
    full_dir = final_root / "full_segmented"
    mirror_dir = final_root / "groundtruth_segmented"
    report_dir = final_root / "reports" / "groundtruth_markdown_import"
    backup_dir = report_dir / "replaced_existing_backups"

    if not markdown_dir.exists():
        raise FileNotFoundError(markdown_dir)
    if not args.api_key:
        raise ValueError("DeepSeek API key is empty.")

    markdown_files = sorted(markdown_dir.glob("paper_*.md"))
    pmid_to_markdown = {}
    for path in markdown_files:
        match = re.fullmatch(r"paper_(\d+)\.md", path.name, flags=re.I)
        if not match:
            raise ValueError(f"Unexpected Markdown filename: {path.name}")
        pmid = match.group(1)
        if pmid in pmid_to_markdown:
            raise ValueError(f"Duplicate PMID Markdown: {pmid}")
        pmid_to_markdown[pmid] = path

    segmenter = FinalSeg(
        config={
            "deepseek_api_key": args.api_key,
            "deepseek_model_name": args.model_name,
            "deepseek_thinking_disabled": True,
        },
        preferred_client="deepseek",
        sleep_seconds=0.0,
    )
    details: list[Dict[str, Any]] = []

    print(f"Ground-truth Markdown papers: {len(pmid_to_markdown)}")
    for index, (pmid, markdown_path) in enumerate(
        sorted(pmid_to_markdown.items(), key=lambda item: int(item[0])),
        start=1,
    ):
        canonical_content, canonical_meta = output_paths(full_dir, pmid)
        mirror_content, mirror_meta = output_paths(mirror_dir, pmid)
        state = existing_state(canonical_content, canonical_meta)
        reusable = (
            state["files_complete"]
            and state["json_valid"]
            and state["has_methods"]
            and state["has_results"]
            and not args.overwrite_all
        )
        previous_state = (
            "complete"
            if reusable
            else "incomplete"
            if state["files_complete"]
            else "absent"
        )

        if reusable:
            write_json(state["content"], mirror_content)
            write_json(state["meta"], mirror_meta)
            details.append(
                {
                    "pmid": pmid,
                    "status": "reused",
                    "previous_state": previous_state,
                    "has_methods": True,
                    "has_results": True,
                    "methods_chars": len(str(state["content"].get("Methods", ""))),
                    "results_chars": len(str(state["content"].get("Results", ""))),
                    "source_markdown": str(markdown_path),
                    "canonical_content": str(canonical_content),
                    "canonical_meta": str(canonical_meta),
                    "error": "",
                }
            )
            print(f"[{index}/{len(pmid_to_markdown)}] PMID {pmid}: reused")
            continue

        print(
            f"[{index}/{len(pmid_to_markdown)}] PMID {pmid}: "
            f"segmenting ({previous_state})"
        )
        try:
            if state["files_complete"]:
                backup_existing(
                    pmid,
                    canonical_content,
                    canonical_meta,
                    backup_dir,
                )
            result = segmenter.segment_markdown(
                markdown_path=markdown_path,
                review_record={
                    "pmid": pmid,
                    "issue_code": "groundtruth_markdown_import",
                    "note": "Complete Markdown supplied for ground-truth evaluation.",
                },
                existing_content=state.get("content") or None,
            )
            content = result.structured
            meta = result.metadata
            meta.update(
                {
                    "pmid": pmid,
                    "groundtruth_markdown_import": True,
                    "groundtruth_import_timestamp": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "previous_final_state": previous_state,
                    "canonical_output_directory": str(full_dir),
                    "mirror_output_directory": str(mirror_dir),
                }
            )
            if not has_meaningful_section(content.get("Methods")):
                raise RuntimeError("Final segmentation produced no meaningful Methods.")
            if not has_meaningful_section(content.get("Results")):
                raise RuntimeError("Final segmentation produced no meaningful Results.")

            write_json(content, canonical_content)
            write_json(meta, canonical_meta)
            write_json(content, mirror_content)
            write_json(meta, mirror_meta)
            llm_meta = meta.get("llm_meta", {})
            details.append(
                {
                    "pmid": pmid,
                    "status": "processed",
                    "previous_state": previous_state,
                    "has_methods": has_meaningful_section(content.get("Methods")),
                    "has_results": has_meaningful_section(content.get("Results")),
                    "methods_chars": len(str(content.get("Methods", "") or "")),
                    "results_chars": len(str(content.get("Results", "") or "")),
                    "llm_client": llm_meta.get("llm_client", ""),
                    "llm_model": llm_meta.get("llm_model", ""),
                    "llm_error": llm_meta.get("llm_error", ""),
                    "source_markdown": str(markdown_path),
                    "canonical_content": str(canonical_content),
                    "canonical_meta": str(canonical_meta),
                    "error": "",
                }
            )
        except Exception as exc:
            details.append(
                {
                    "pmid": pmid,
                    "status": "error",
                    "previous_state": previous_state,
                    "has_methods": False,
                    "has_results": False,
                    "source_markdown": str(markdown_path),
                    "canonical_content": str(canonical_content),
                    "canonical_meta": str(canonical_meta),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}")

        write_csv(details, report_dir / "groundtruth_segmentation_details.csv")

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "markdown_files": len(pmid_to_markdown),
        "reused_complete": sum(row["status"] == "reused" for row in details),
        "processed": sum(row["status"] == "processed" for row in details),
        "errors": sum(row["status"] == "error" for row in details),
        "complete_with_methods": sum(
            bool(row.get("has_methods")) for row in details
        ),
        "complete_with_results": sum(
            bool(row.get("has_results")) for row in details
        ),
        "llm_processed": sum(bool(row.get("llm_client")) for row in details),
        "final_full_segmented_dir": str(full_dir),
        "groundtruth_mirror_dir": str(mirror_dir),
    }
    write_json(summary, report_dir / "groundtruth_segmentation_summary.json")
    write_csv(details, report_dir / "groundtruth_segmentation_details.csv")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
