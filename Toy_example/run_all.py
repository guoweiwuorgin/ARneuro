"""Run the complete Toy workflow only when --execute is explicitly supplied."""

from __future__ import annotations

import argparse
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STEPS = [
    "00_create_input_excel.py", "01_validate_pubmed_excel.py", "02_screen_records.py",
    "03_acquire_or_import_pdfs.py", "04_ocr_to_markdown.py", "05_segment_and_validate.py",
    "06_extract_generic_study_info.py", "07_extract_tables_and_coordinates.py",
    "08_build_citation_network.py", "09_build_sqlite_database.py",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run the local two-paper fixture workflow.")
    args = parser.parse_args()
    if not args.execute:
        print("Dry run only. The ordered offline stages are:")
        print("\n".join(f"  {index + 1}. {name}" for index, name in enumerate(STEPS)))
        print("Use --execute to create only local Toy outputs. No model or network call is included.")
        return
    for name in STEPS:
        print(f"\n=== {name} ===")
        runpy.run_path(str(ROOT / name), run_name="__main__")


if __name__ == "__main__":
    main()
