"""Serial 10-paper Methods extraction test with DeepSeek-V4-Flash."""

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

EXTRACTOR_PATH = ARNEURO_DIR / "feature_extraction" / "glm_method_info_extractor.py"
spec = importlib.util.spec_from_file_location("deepseek_method_info_extractor", EXTRACTOR_PATH)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = module
spec.loader.exec_module(module)
DeepSeekMethodSectionInfoExtractor = module.DeepSeekMethodSectionInfoExtractor

DEFAULT_SEGMENTED_DIR = STEP3_DIR / "final_segmented_for_analysis" / "full_segmented"
DEFAULT_OUTPUT_DIR = STEP3_DIR / "method_info_extraction_deepseek_v4_flash_test"
DEFAULT_EXCLUDED_DETAILS_CSV = (
    STEP3_DIR / "final_segmented_for_analysis" / "reports" / "final_seg_details.csv"
)
DEFAULT_API_KEY = "sk-XXXXXXXX"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serial 10-paper DeepSeek-V4-Flash Methods extraction test."
    )
    parser.add_argument("--segmented-dir", default=str(DEFAULT_SEGMENTED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-excluded", action="store_true")
    parser.add_argument("--excluded-details-csv", default=str(DEFAULT_EXCLUDED_DETAILS_CSV))
    parser.add_argument("--model-name", default="deepseek-v4-flash")
    parser.add_argument("--max-method-chars", type=int, default=50000)
    parser.add_argument("--short-methods-threshold", type=int, default=2000)
    parser.add_argument("--supplementary-section-chars", type=int, default=12000)
    parser.add_argument("--request-interval-seconds", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY", DEFAULT_API_KEY),
    )
    return parser.parse_args()


def load_excluded_pmids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return {
            str(row.get("pmid", "")).strip()
            for row in csv.DictReader(file_obj)
            if row.get("status") == "excluded" and str(row.get("pmid", "")).strip()
        }


def main() -> None:
    args = parse_args()
    segmented_dir = Path(args.segmented_dir)
    output_dir = Path(args.output_dir)
    if not segmented_dir.exists():
        raise FileNotFoundError(segmented_dir)
    if not args.api_key:
        raise ValueError("DeepSeek API key is empty.")

    excluded_pmids = (
        set() if args.include_excluded
        else load_excluded_pmids(Path(args.excluded_details_csv))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "skipped_excluded_pmids.txt").write_text(
        "\n".join(sorted(excluded_pmids, key=lambda value: (not value.isdigit(), value)))
        + ("\n" if excluded_pmids else ""),
        encoding="utf-8",
    )

    extractor = DeepSeekMethodSectionInfoExtractor(
        api_key=args.api_key,
        model_name=args.model_name,
        max_method_chars=args.max_method_chars,
        request_interval_seconds=args.request_interval_seconds,
        max_retries=args.max_retries,
        max_tokens=args.max_tokens,
        short_methods_threshold=args.short_methods_threshold,
        supplementary_section_chars=args.supplementary_section_chars,
    )
    print("Starting DeepSeek-V4-Flash serial test")
    print(f"  Model: {args.model_name}")
    print("  Thinking: disabled")
    print("  Concurrency: 1")
    records = extractor.extract_directory_serial(
        segmented_dir=segmented_dir,
        output_json_dir=output_dir / "json",
        output_csv_path=output_dir / "method_info_table.csv",
        limit=args.limit,
        overwrite=args.overwrite,
        skip_pmids=excluded_pmids,
    )
    print(f"Completed records: {len(records)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
