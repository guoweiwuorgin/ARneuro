"""
Full resumable Methods extraction with BigModel GLM-4.5-Air.

Behavior:
    - processes the complete segmented corpus;
    - skips manually excluded PMIDs by default;
    - sends exactly one API request at a time;
    - adds Introduction and Results when Methods is shorter than 2000 chars;
    - saves one JSON immediately after each paper;
    - reuses only successful JSON files when resumed;
    - retries failed or invalid JSON outputs on the next run;
    - rebuilds the final CSV and summary from all per-paper JSON files.

Run:
    conda run --no-capture-output -n reviewer python -B \
      ARneuro/examples/step4_extract_method_info_with_glm4_5_air_full.py

Resume after interruption by running the same command again.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
ARNEURO_DIR = STEP3_DIR / "ARneuro"

for import_path in [ARNEURO_DIR / "feature_extraction", ARNEURO_DIR, STEP3_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

METHOD_INFO_PATH = (
    ARNEURO_DIR / "feature_extraction" / "method_info_extractor.py"
)
method_spec = importlib.util.spec_from_file_location(
    "method_info_extractor",
    METHOD_INFO_PATH,
)
method_module = importlib.util.module_from_spec(method_spec)
assert method_spec and method_spec.loader
sys.modules[method_spec.name] = method_module
method_spec.loader.exec_module(method_module)
METHOD_INFO_COLUMNS = method_module.METHOD_INFO_COLUMNS

GLM_EXTRACTOR_PATH = (
    ARNEURO_DIR / "feature_extraction" / "glm_method_info_extractor.py"
)
glm_spec = importlib.util.spec_from_file_location(
    "glm_method_info_extractor",
    GLM_EXTRACTOR_PATH,
)
glm_module = importlib.util.module_from_spec(glm_spec)
assert glm_spec and glm_spec.loader
sys.modules[glm_spec.name] = glm_module
glm_spec.loader.exec_module(glm_module)
GLMMethodSectionInfoExtractor = glm_module.GLMMethodSectionInfoExtractor


DEFAULT_SEGMENTED_DIR = (
    STEP3_DIR / "final_segmented_for_analysis" / "full_segmented"
)
DEFAULT_OUTPUT_DIR = STEP3_DIR / "method_info_extraction_glm45_air_full"
DEFAULT_EXCLUDED_DETAILS_CSV = (
    STEP3_DIR
    / "final_segmented_for_analysis"
    / "reports"
    / "final_seg_details.csv"
)
DEFAULT_API_KEY = "random-placeholder-7fb7ca41ea0e861ae48cac68bcd5c19f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full resumable serial GLM-4.5-Air Methods extraction."
    )
    parser.add_argument("--segmented-dir", default=str(DEFAULT_SEGMENTED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debugging limit. Omit for the complete corpus.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess successful per-paper JSON files.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Include the manually excluded PMID set.",
    )
    parser.add_argument(
        "--excluded-details-csv",
        default=str(DEFAULT_EXCLUDED_DETAILS_CSV),
    )
    parser.add_argument("--max-method-chars", type=int, default=50000)
    parser.add_argument("--short-methods-threshold", type=int, default=2000)
    parser.add_argument("--supplementary-section-chars", type=int, default=12000)
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=0.0,
        help="Delay after successful requests. Concurrency always remains one.",
    )
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BIGMODEL_API_KEY", DEFAULT_API_KEY),
        help="BIGMODEL_API_KEY overrides the embedded key.",
    )
    return parser.parse_args()


def load_excluded_pmids(details_csv: Path) -> set[str]:
    if not details_csv.exists():
        return set()
    with details_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return {
            str(row.get("pmid", "")).strip()
            for row in csv.DictReader(file_obj)
            if row.get("status") == "excluded"
            and str(row.get("pmid", "")).strip()
        }



def write_skip_file(pmids: set[str], output_dir: Path) -> Path:
    skip_path = output_dir / "skipped_excluded_pmids.txt"
    skip_path.parent.mkdir(parents=True, exist_ok=True)
    skip_path.write_text(
        "\n".join(sorted(pmids, key=lambda value: (not value.isdigit(), value)))
        + ("\n" if pmids else ""),
        encoding="utf-8",
    )
    return skip_path


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file_obj:
            data = json.load(file_obj)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def rebuild_outputs(
    json_dir: Path,
    csv_path: Path,
    summary_path: Path,
    expected_pmids: set[str],
    excluded_pmids: set[str],
) -> Dict[str, Any]:
    successful_records: List[Dict[str, Any]] = []
    successful_pmids: set[str] = set()
    failed_pmids: List[str] = []
    empty_methods_pmids: List[str] = []

    for json_path in sorted(json_dir.glob("paper_*_method_info.json")):
        data = load_json(json_path)
        pmid = str(data.get("pmid", "")).strip()
        if not pmid or pmid not in expected_pmids:
            continue

        metadata = data.get("metadata", {})
        flat_record = data.get("flat_record", {})
        status = metadata.get("status") if isinstance(metadata, dict) else ""

        if status == "success" and isinstance(flat_record, dict):
            successful_records.append(flat_record)
            successful_pmids.add(pmid)
        elif status == "empty_methods":
            empty_methods_pmids.append(pmid)
        else:
            failed_pmids.append(pmid)

    successful_records.sort(
        key=lambda row: (
            not str(row.get("PMID", "")).isdigit(),
            str(row.get("PMID", "")),
        )
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=METHOD_INFO_COLUMNS)
        writer.writeheader()
        for record in successful_records:
            writer.writerow(
                {
                    column: record.get(column, "")
                    for column in METHOD_INFO_COLUMNS
                }
            )

    missing_pmids = sorted(
        expected_pmids - successful_pmids - set(failed_pmids) - set(empty_methods_pmids),
        key=lambda value: (not value.isdigit(), value),
    )
    summary = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expected_valid_papers": len(expected_pmids),
        "successful_papers": len(successful_pmids),
        "failed_papers": len(failed_pmids),
        "empty_methods_papers": len(empty_methods_pmids),
        "not_yet_processed_papers": len(missing_pmids),
        "excluded_papers": len(excluded_pmids),
        "failed_pmids": sorted(
            set(failed_pmids), key=lambda value: (not value.isdigit(), value)
        ),
        "empty_methods_pmids": sorted(
            set(empty_methods_pmids),
            key=lambda value: (not value.isdigit(), value),
        ),
        "not_yet_processed_pmids": missing_pmids,
        "csv_path": str(csv_path),
        "json_dir": str(json_dir),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, ensure_ascii=False, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    segmented_dir = Path(args.segmented_dir)
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    csv_path = output_dir / "method_info_table.csv"
    summary_path = output_dir / "extraction_summary.json"

    if not segmented_dir.exists():
        raise FileNotFoundError(f"Segmented directory not found: {segmented_dir}")
    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.request_interval_seconds < 0:
        raise ValueError("--request-interval-seconds must be >= 0")
    if args.short_methods_threshold < 0:
        raise ValueError("--short-methods-threshold must be >= 0")
    if not args.api_key:
        raise ValueError("BigModel API key is empty.")

    excluded_pmids = (
        set()
        if args.include_excluded
        else load_excluded_pmids(Path(args.excluded_details_csv))
    )
    skip_path = write_skip_file(excluded_pmids, output_dir)

    content_files = sorted(
        segmented_dir.glob("paper_*_structured_content.json")
    )
    expected_pmids = {
        method_module.pmid_from_path(path)
        for path in content_files
        if method_module.pmid_from_path(path) not in excluded_pmids
    }
    if args.limit is not None:
        limited_files = [
            path
            for path in content_files
            if method_module.pmid_from_path(path) not in excluded_pmids
        ][: args.limit]
        expected_pmids = {
            method_module.pmid_from_path(path) for path in limited_files
        }

    extractor = GLMMethodSectionInfoExtractor(
        api_key=args.api_key,
        model_name="GLM-4.5-Air",
        max_method_chars=args.max_method_chars,
        request_interval_seconds=args.request_interval_seconds,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        short_methods_threshold=args.short_methods_threshold,
        supplementary_section_chars=args.supplementary_section_chars,
    )

    print("Starting full GLM-4.5-Air extraction")
    print(f"  Expected papers: {len(expected_pmids)}")
    print(f"  Excluded PMIDs: {len(excluded_pmids)}")
    print("  Concurrency: 1")
    print(f"  Delay after successful requests: {args.request_interval_seconds}s")
    print(f"  Short Methods threshold: {args.short_methods_threshold} chars")
    print(f"  Output directory: {output_dir}")

    try:
        extractor.extract_directory_serial(
            segmented_dir=segmented_dir,
            output_json_dir=json_dir,
            output_csv_path=csv_path,
            limit=args.limit,
            overwrite=args.overwrite,
            skip_pmids=excluded_pmids,
        )
    finally:
        summary = rebuild_outputs(
            json_dir=json_dir,
            csv_path=csv_path,
            summary_path=summary_path,
            expected_pmids=expected_pmids,
            excluded_pmids=excluded_pmids,
        )
        print("Current extraction summary")
        print(f"  Successful: {summary['successful_papers']}")
        print(f"  Failed: {summary['failed_papers']}")
        print(f"  Empty Methods: {summary['empty_methods_papers']}")
        print(f"  Not yet processed: {summary['not_yet_processed_papers']}")
        print(f"  JSON directory: {json_dir}")
        print(f"  Final CSV: {csv_path}")
        print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
