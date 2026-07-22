"""
Final supplementary segmentation for recoverable missing Methods/Results papers.

Run:
    conda run -n reviewer python ARneuro/examples/step_supp_seg_final_recover_methods_results.py

Outputs:
    final_segmented_for_analysis/segmented
    final_segmented_for_analysis/reports
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
ARNEURO_DIR = STEP3_DIR / "ARneuro"

if str(ARNEURO_DIR) not in sys.path:
    sys.path.insert(0, str(ARNEURO_DIR))

from text_processing.final_segmentation import FinalSeg


REVIEW_CSV = (
    STEP3_DIR
    / "merged_segmented_for_analysis"
    / "reports"
    / "missing_methods_results_review.csv"
)
MERGED_SEGMENTED_DIR = STEP3_DIR / "merged_segmented_for_analysis" / "segmented"
OUTPUT_DIR = STEP3_DIR / "final_segmented_for_analysis"
OUTPUT_SEGMENTED_DIR = OUTPUT_DIR / "segmented"
OUTPUT_FULL_SEGMENTED_DIR = OUTPUT_DIR / "full_segmented"
OUTPUT_REPORTS_DIR = OUTPUT_DIR / "reports"

EXCLUDED_ISSUE_CODES = {
    "pdf_conversion_failed",
    "non_study_review_or_other",
}
EXCLUDED_NOTE_PATTERN = re.compile(r"ERP|排除文章", flags=re.I)

CONFIG = {
    "mimo_api_key": "random-placeholder-9e72ea0aa8cb64ac236a89d146af6c60",
    "mimo_model_name": "mimo-v2.5-pro",
    "deepseek_api_key": "random-placeholder-ecd0c598d8d2d6f57d3ec57c5d256700",
    "deepseek_model_name": "deepseek-chat",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


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


def build_full_final_corpus() -> Dict[str, Any]:
    OUTPUT_FULL_SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    overlaid = 0

    for source_path in MERGED_SEGMENTED_DIR.glob("paper_*_structured_*.json"):
        target_path = OUTPUT_FULL_SEGMENTED_DIR / source_path.name
        shutil.copy2(source_path, target_path)
        copied += 1

    for source_path in OUTPUT_SEGMENTED_DIR.glob("paper_*_structured_*.json"):
        target_path = OUTPUT_FULL_SEGMENTED_DIR / source_path.name
        shutil.copy2(source_path, target_path)
        overlaid += 1

    content_files = list(OUTPUT_FULL_SEGMENTED_DIR.glob("paper_*_structured_content.json"))
    meta_files = list(OUTPUT_FULL_SEGMENTED_DIR.glob("paper_*_structured_meta.json"))

    return {
        "full_segmented_dir": str(OUTPUT_FULL_SEGMENTED_DIR),
        "copied_json_files_from_merged": copied,
        "overlaid_json_files_from_final": overlaid,
        "full_content_count": len(content_files),
        "full_meta_count": len(meta_files),
    }


def pmid_sort_key(record: Dict[str, Any]) -> tuple:
    pmid = str(record.get("pmid", ""))
    return (not pmid.isdigit(), pmid)


def load_review_records() -> List[Dict[str, str]]:
    with REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as file_obj:
        records = list(csv.DictReader(file_obj))
    return sorted(records, key=pmid_sort_key)


def is_recoverable(record: Dict[str, str]) -> bool:
    if record.get("issue_code") in EXCLUDED_ISSUE_CODES:
        return False
    note = record.get("note", "")
    if EXCLUDED_NOTE_PATTERN.search(note):
        return False
    return True


def merged_content_path(pmid: str) -> Path:
    return MERGED_SEGMENTED_DIR / f"paper_{pmid}_structured_content.json"


def merged_meta_path(pmid: str) -> Path:
    return MERGED_SEGMENTED_DIR / f"paper_{pmid}_structured_meta.json"


def output_content_path(pmid: str) -> Path:
    return OUTPUT_SEGMENTED_DIR / f"paper_{pmid}_structured_content.json"


def output_meta_path(pmid: str) -> Path:
    return OUTPUT_SEGMENTED_DIR / f"paper_{pmid}_structured_meta.json"


def copy_excluded(record: Dict[str, str]) -> Dict[str, Any]:
    pmid = record["pmid"]
    content_path = merged_content_path(pmid)
    meta_path = merged_meta_path(pmid)
    if content_path.exists():
        write_json(read_json(content_path), output_content_path(pmid))
    if meta_path.exists():
        meta = read_json(meta_path)
    else:
        meta = {}
    meta["final_seg_excluded"] = True
    meta["final_seg_exclusion_reason"] = record.get("issue_label", "")
    meta["final_seg_review_note"] = record.get("note", "")
    write_json(meta, output_meta_path(pmid))
    return {
        "pmid": pmid,
        "status": "excluded",
        "issue_code": record.get("issue_code", ""),
        "note": record.get("note", ""),
        "has_methods": False,
        "has_results": False,
        "content_path": str(output_content_path(pmid)),
        "meta_path": str(output_meta_path(pmid)),
        "error": "",
    }


def detail_from_existing_meta(record: Dict[str, str]) -> Dict[str, Any]:
    pmid = record["pmid"]
    meta_path = output_meta_path(pmid)
    content_path = output_content_path(pmid)
    meta = read_json(meta_path) if meta_path.exists() else {}
    content = read_json(content_path) if content_path.exists() else {}
    status = "excluded" if meta.get("final_seg_excluded") else "processed"
    return {
        "pmid": pmid,
        "status": status,
        "issue_code": record.get("issue_code", ""),
        "note": record.get("note", ""),
        "has_methods": bool(content.get("Methods")),
        "has_results": bool(content.get("Results")),
        "methods_len": len(str(content.get("Methods", "") or "").strip()),
        "results_len": len(str(content.get("Results", "") or "").strip()),
        "llm_calls_total": meta.get("llm_calls_total", ""),
        "llm_client": meta.get("llm_meta", {}).get("llm_client", ""),
        "llm_error": meta.get("llm_meta", {}).get("llm_error", ""),
        "content_path": str(content_path),
        "meta_path": str(meta_path),
        "error": "",
    }


def process_recoverable(segmenter: FinalSeg, record: Dict[str, str]) -> Dict[str, Any]:
    pmid = record["pmid"]
    markdown_path = Path(record["markdown_path"])
    if not markdown_path.exists():
        raise FileNotFoundError(markdown_path)

    existing_content = {}
    if merged_content_path(pmid).exists():
        existing_content = read_json(merged_content_path(pmid))

    result = segmenter.segment_markdown(
        markdown_path=markdown_path,
        review_record=record,
        existing_content=existing_content,
    )
    result.metadata["pmid"] = pmid
    result.metadata["final_seg_recovered"] = True
    result.metadata["previous_merged_content_path"] = str(merged_content_path(pmid))
    result.metadata["previous_merged_meta_path"] = str(merged_meta_path(pmid))

    write_json(result.structured, output_content_path(pmid))
    write_json(result.metadata, output_meta_path(pmid))

    return {
        "pmid": pmid,
        "status": "processed",
        "issue_code": record.get("issue_code", ""),
        "note": record.get("note", ""),
        "has_methods": bool(result.structured.get("Methods")),
        "has_results": bool(result.structured.get("Results")),
        "methods_len": len(str(result.structured.get("Methods", "") or "").strip()),
        "results_len": len(str(result.structured.get("Results", "") or "").strip()),
        "llm_calls_total": result.metadata.get("llm_calls_total", ""),
        "llm_client": result.metadata.get("llm_meta", {}).get("llm_client", ""),
        "llm_error": result.metadata.get("llm_meta", {}).get("llm_error", ""),
        "content_path": str(output_content_path(pmid)),
        "meta_path": str(output_meta_path(pmid)),
        "error": "",
    }


def main() -> None:
    OUTPUT_SEGMENTED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    records = load_review_records()
    recoverable = [record for record in records if is_recoverable(record)]
    excluded = [record for record in records if not is_recoverable(record)]

    segmenter = FinalSeg(config=CONFIG, preferred_client="deepseek")
    details: List[Dict[str, Any]] = []

    def write_reports(full_corpus_stats: Optional[Dict[str, Any]] = None) -> None:
        full_corpus_stats = full_corpus_stats or {}
        summary = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_reviewed": len(records),
            "recoverable": len(recoverable),
            "excluded": len(excluded),
            "processed": sum(1 for row in details if row.get("status") == "processed"),
            "errors": sum(1 for row in details if row.get("status") == "error"),
            "processed_with_methods": sum(1 for row in details if row.get("status") == "processed" and row.get("has_methods") is True),
            "processed_with_results": sum(1 for row in details if row.get("status") == "processed" and row.get("has_results") is True),
            "output_segmented_dir": str(OUTPUT_SEGMENTED_DIR),
            **full_corpus_stats,
        }
        write_json(summary, OUTPUT_REPORTS_DIR / "final_seg_summary.json")
        write_csv(details, OUTPUT_REPORTS_DIR / "final_seg_details.csv")

    print("Final supplementary segmentation")
    print(f"  Total reviewed: {len(records)}")
    print(f"  Recoverable: {len(recoverable)}")
    print(f"  Excluded: {len(excluded)}")

    for record in excluded:
        if output_content_path(record["pmid"]).exists() and output_meta_path(record["pmid"]).exists():
            details.append(detail_from_existing_meta(record))
        else:
            details.append(copy_excluded(record))
        write_reports()

    for idx, record in enumerate(recoverable, start=1):
        pmid = record["pmid"]
        print(f"[{idx}/{len(recoverable)}] PMID {pmid} ({record.get('issue_code', '')})")
        if output_content_path(pmid).exists() and output_meta_path(pmid).exists():
            print("  existing final result found, skipping")
            details.append(detail_from_existing_meta(record))
            write_reports()
            continue
        try:
            details.append(process_recoverable(segmenter, record))
        except Exception as exc:
            details.append(
                {
                    "pmid": pmid,
                    "status": "error",
                    "issue_code": record.get("issue_code", ""),
                    "note": record.get("note", ""),
                    "has_methods": False,
                    "has_results": False,
                    "content_path": str(output_content_path(pmid)),
                    "meta_path": str(output_meta_path(pmid)),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"  ERROR: {type(exc).__name__}: {exc}")
        write_reports()

    full_corpus_stats = build_full_final_corpus()
    write_reports(full_corpus_stats=full_corpus_stats)

    print("Done")
    print((OUTPUT_REPORTS_DIR / "final_seg_summary.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
