"""
Serial 10-paper Methods extraction test with BigModel GLM-4.5-Air.

This script intentionally sends only one request at a time. It skips the
manually excluded PMID set and stores each paper immediately for inspection.

Run:
    conda run -n reviewer python -B \
      ARneuro/examples/step4_extract_method_info_with_glm4.5.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
ARNEURO_DIR = STEP3_DIR / "ARneuro"

for import_path in [ARNEURO_DIR / "feature_extraction", ARNEURO_DIR, STEP3_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

EXTRACTOR_PATH = (
    ARNEURO_DIR / "feature_extraction" / "glm_method_info_extractor.py"
)
spec = importlib.util.spec_from_file_location(
    "glm_method_info_extractor",
    EXTRACTOR_PATH,
)
extractor_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = extractor_module
spec.loader.exec_module(extractor_module)
GLMMethodSectionInfoExtractor = extractor_module.GLMMethodSectionInfoExtractor


DEFAULT_SEGMENTED_DIR = (
    STEP3_DIR / "final_segmented_for_analysis" / "full_segmented"
)
DEFAULT_OUTPUT_DIR = STEP3_DIR / "method_info_extraction_glm45_air_typical_human_test"
DEFAULT_EXCLUDED_DETAILS_CSV = (
    STEP3_DIR
    / "final_segmented_for_analysis"
    / "reports"
    / "final_seg_details.csv"
)
DEFAULT_API_KEY = "ca8b0e2279df377e368038fcd4fa602a.Eu3E34ORqIWmt64R"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serial 10-paper GLM-4.5-Air Methods extraction test."
    )
    parser.add_argument("--segmented-dir", default=str(DEFAULT_SEGMENTED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument(
        "--excluded-details-csv",
        default=str(DEFAULT_EXCLUDED_DETAILS_CSV),
    )
    parser.add_argument(
        "--max-method-chars",
        type=int,
        default=50000,
    )
    parser.add_argument(
        "--short-methods-threshold",
        type=int,
        default=2000,
        help="Add Introduction and Results when Methods is shorter than this.",
    )
    parser.add_argument(
        "--supplementary-section-chars",
        type=int,
        default=12000,
        help="Maximum characters retained from each supplementary section.",
    )
    parser.add_argument(
        "--request-interval-seconds",
        type=float,
        default=0.0,
        help="Delay between successful serial requests. Default: no delay.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Serial retries with long exponential backoff for HTTP 429.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BIGMODEL_API_KEY", DEFAULT_API_KEY),
        help="BigModel key. BIGMODEL_API_KEY overrides the embedded test key.",
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


def main() -> None:
    args = parse_args()
    segmented_dir = Path(args.segmented_dir)
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    csv_path = output_dir / "method_info_table.csv"

    if not segmented_dir.exists():
        raise FileNotFoundError(f"Segmented directory not found: {segmented_dir}")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")
    if args.request_interval_seconds < 0:
        raise ValueError("--request-interval-seconds must be >= 0")
    if not args.api_key:
        raise ValueError("BigModel API key is empty.")

    excluded_pmids = (
        set()
        if args.include_excluded
        else load_excluded_pmids(Path(args.excluded_details_csv))
    )
    skip_path = write_skip_file(excluded_pmids, output_dir)

    extractor = GLMMethodSectionInfoExtractor(
        api_key=args.api_key,
        model_name="GLM-4.5-Air",
        max_method_chars=args.max_method_chars,
        request_interval_seconds=args.request_interval_seconds,
        max_retries=args.max_retries,
        max_tokens=8192,
        short_methods_threshold=args.short_methods_threshold,
        supplementary_section_chars=args.supplementary_section_chars,
    )

    print("Starting strictly serial GLM-4.5-Air extraction")
    print(f"  Papers requested: {args.limit}")
    print(f"  Delay after successful requests: {args.request_interval_seconds}s")
    print(f"  Short Methods threshold: {args.short_methods_threshold} chars")
    print("  Concurrency: 1")

    records = extractor.extract_directory_serial(
        segmented_dir=segmented_dir,
        output_json_dir=json_dir,
        output_csv_path=csv_path,
        limit=args.limit,
        overwrite=args.overwrite,
        skip_pmids=excluded_pmids,
    )

    print("GLM-4.5-Air serial test complete")
    print(f"  Records available: {len(records)}")
    print(f"  Excluded PMIDs skipped: {len(excluded_pmids)}")
    print(f"  Skip list: {skip_path}")
    print(f"  Per-paper JSON: {json_dir}")
    print(f"  CSV: {csv_path}")


if __name__ == "__main__":
    main()
