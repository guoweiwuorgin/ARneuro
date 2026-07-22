"""
Review the current 109 merged papers still missing Methods and/or Results.

Run from the step3_library workspace:
    conda run -n reviewer python ARneuro/examples/review_current_missing_methods_results.py

The GUI writes:
    merged_segmented_for_analysis/reports/missing_methods_results_review.jsonl
    merged_segmented_for_analysis/reports/missing_methods_results_review.csv
"""

from __future__ import annotations

import sys
from pathlib import Path


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
ARNEURO_DIR = STEP3_DIR / "ARneuro"

if str(ARNEURO_DIR) not in sys.path:
    sys.path.insert(0, str(ARNEURO_DIR))

from utils.markdown_issue_reviewer import review_missing_methods_results


MISSING_REPORT_CSV = (
    STEP3_DIR
    / "merged_segmented_for_analysis"
    / "reports"
    / "missing_methods_results.csv"
)

OUTPUT_JSONL = (
    STEP3_DIR
    / "merged_segmented_for_analysis"
    / "reports"
    / "missing_methods_results_review.jsonl"
)

OUTPUT_CSV = (
    STEP3_DIR
    / "merged_segmented_for_analysis"
    / "reports"
    / "missing_methods_results_review.csv"
)

MARKDOWN_SEARCH_ROOTS = [
    STEP3_DIR / "segmentation_failed_pmids",
    STEP3_DIR / "incomplete_pmids",
    STEP3_DIR,
]


def main() -> None:
    if not MISSING_REPORT_CSV.exists():
        raise FileNotFoundError(
            f"Missing report not found: {MISSING_REPORT_CSV}. "
            "Run check_merged_missing_methods_results.py first."
        )

    review_missing_methods_results(
        missing_report_csv=MISSING_REPORT_CSV,
        output_jsonl=OUTPUT_JSONL,
        output_csv=OUTPUT_CSV,
        search_roots=MARKDOWN_SEARCH_ROOTS,
    )


if __name__ == "__main__":
    main()
