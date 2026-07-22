"""
Extract structured Method-section information with MiMo-v2.5 Pro.

Default input:
    D:/language_template/reviewer/current_data/step3_library/final_segmented_for_analysis/full_segmented

Default outputs:
    D:/language_template/reviewer/current_data/step3_library/method_info_extraction/json
    D:/language_template/reviewer/current_data/step3_library/method_info_extraction/method_info_table.csv

Examples:
    conda run -n reviewer python ARneuro/examples/step4_extract_method_info_with_mimo.py --limit 5
    conda run -n reviewer python ARneuro/examples/step4_extract_method_info_with_mimo.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
ARNEURO_DIR = STEP3_DIR / "ARneuro"

if str(ARNEURO_DIR) not in sys.path:
    sys.path.insert(0, str(ARNEURO_DIR))
if str(STEP3_DIR) not in sys.path:
    sys.path.insert(0, str(STEP3_DIR))

METHOD_INFO_MODULE_PATH = ARNEURO_DIR / "feature_extraction" / "method_info_extractor.py"
spec = importlib.util.spec_from_file_location("arneuro_method_info_extractor", METHOD_INFO_MODULE_PATH)
method_info_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = method_info_module
spec.loader.exec_module(method_info_module)
MethodSectionInfoExtractor = method_info_module.MethodSectionInfoExtractor


DEFAULT_SEGMENTED_DIR = STEP3_DIR / "final_segmented_for_analysis" / "full_segmented"
DEFAULT_OUTPUT_DIR = STEP3_DIR / "method_info_extraction"
DEFAULT_EXCLUDED_DETAILS_CSV = (
    STEP3_DIR
    / "final_segmented_for_analysis"
    / "reports"
    / "final_seg_details.csv"
)

DEFAULT_CONFIG = {
    "mimo_api_key": "random-placeholder-9e72ea0aa8cb64ac236a89d146af6c60",
    "mimo_model_name": "mimo-v2.5-pro",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract method information from segmented papers using MiMo."
    )
    parser.add_argument(
        "--segmented-dir",
        default=str(DEFAULT_SEGMENTED_DIR),
        help="Directory containing *_structured_content.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for per-paper JSON records and CSV table.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of papers to process for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract papers even if per-paper JSON already exists.",
    )
    parser.add_argument(
        "--excluded-details-csv",
        default=str(DEFAULT_EXCLUDED_DETAILS_CSV),
        help=(
            "CSV containing final segmentation details. Rows with status=excluded "
            "are skipped by default."
        ),
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Process excluded PMIDs instead of skipping them.",
    )
    parser.add_argument(
        "--max-method-chars",
        type=int,
        default=60000,
        help="Maximum Methods-section characters sent to the model.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=6,
        help="Number of concurrent extraction workers.",
    )
    return parser.parse_args()


def load_excluded_pmids(details_csv: Path) -> set:
    if not details_csv.exists():
        return set()
    with details_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return {
            str(row.get("pmid", "")).strip()
            for row in csv.DictReader(file_obj)
            if row.get("status") == "excluded" and str(row.get("pmid", "")).strip()
        }


def write_skip_file(pmids: set, output_dir: Path) -> Path:
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
    excluded_pmids = set() if args.include_excluded else load_excluded_pmids(Path(args.excluded_details_csv))
    skip_path = write_skip_file(excluded_pmids, output_dir)

    if not segmented_dir.exists():
        raise FileNotFoundError(segmented_dir)

    extractor = MethodSectionInfoExtractor(
        config=DEFAULT_CONFIG,
        client_type="mimo",
        model_name=DEFAULT_CONFIG["mimo_model_name"],
        max_method_chars=args.max_method_chars,
    )

    records = extractor.extract_directory(
        segmented_dir=segmented_dir,
        output_json_dir=json_dir,
        output_csv_path=csv_path,
        limit=args.limit,
        overwrite=args.overwrite,
        skip_pmids=excluded_pmids,
        workers=args.workers,
    )

    print("Method information extraction complete")
    print(f"  Papers in this run: {len(records)}")
    print(f"  Excluded PMIDs skipped: {len(excluded_pmids)}")
    print(f"  Workers: {args.workers}")
    print(f"  Skip list: {skip_path}")
    print(f"  Per-paper JSON dir: {json_dir}")
    print(f"  Table CSV: {csv_path}")


if __name__ == "__main__":
    main()
