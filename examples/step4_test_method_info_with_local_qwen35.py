"""
Test Method-section information extraction with local Qwen3.5-9B.

Designed for the server environment:
    workdir: /storage/work/wuguowei/reviewer/allrights_pdf_to_markdown_deepseek_ocr/step3_library/
    model:   /storage/work/wuguowei/Bigmodel/Qwen3.5-9B

Run on server:
    cd /storage/work/wuguowei/reviewer/allrights_pdf_to_markdown_deepseek_ocr/step3_library/
    python ARneuro/examples/step4_test_method_info_with_local_qwen35.py

This script processes 10 non-excluded papers by default and writes outputs to:
    method_info_extraction_qwen35_test/
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path


SERVER_STEP3_DIR = Path(
    "/storage/work/wuguowei/reviewer/allrights_pdf_to_markdown_deepseek_ocr/step3_library"
)
SERVER_MODEL_PATH = Path("/storage/work/wuguowei/Bigmodel/Qwen3.5-9B")

STEP3_DIR = SERVER_STEP3_DIR
ARNEURO_DIR = STEP3_DIR / "ARneuro"

if str(ARNEURO_DIR / "feature_extraction") not in sys.path:
    sys.path.insert(0, str(ARNEURO_DIR / "feature_extraction"))
if str(ARNEURO_DIR) not in sys.path:
    sys.path.insert(0, str(ARNEURO_DIR))
if str(STEP3_DIR) not in sys.path:
    sys.path.insert(0, str(STEP3_DIR))

LOCAL_EXTRACTOR_PATH = ARNEURO_DIR / "feature_extraction" / "local_method_info_extractor.py"
spec = importlib.util.spec_from_file_location("local_method_info_extractor", LOCAL_EXTRACTOR_PATH)
local_module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = local_module
spec.loader.exec_module(local_module)
LocalModelMethodSectionInfoExtractor = local_module.LocalModelMethodSectionInfoExtractor


DEFAULT_SEGMENTED_DIR = STEP3_DIR / "final_segmented_for_analysis" / "full_segmented"
DEFAULT_OUTPUT_DIR = STEP3_DIR / "method_info_extraction_qwen35_test"
DEFAULT_EXCLUDED_DETAILS_CSV = (
    STEP3_DIR
    / "final_segmented_for_analysis"
    / "reports"
    / "final_seg_details.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test method information extraction with local Qwen3.5-9B."
    )
    parser.add_argument(
        "--model-path",
        default=str(SERVER_MODEL_PATH),
        help="Local HuggingFace model directory.",
    )
    parser.add_argument(
        "--segmented-dir",
        default=str(DEFAULT_SEGMENTED_DIR),
        help="Directory containing paper_*_structured_content.json files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for local-model test results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of non-excluded papers to test.",
    )
    parser.add_argument(
        "--max-method-chars",
        type=int,
        default=45000,
        help="Maximum Methods-section characters sent to the local model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum generated tokens per paper.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        help="auto, bfloat16, float16, or float32.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map value.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even if per-paper JSON exists.",
    )
    parser.add_argument(
        "--include-excluded",
        action="store_true",
        help="Do not skip manually excluded PMIDs.",
    )
    parser.add_argument(
        "--excluded-details-csv",
        default=str(DEFAULT_EXCLUDED_DETAILS_CSV),
        help="CSV where status=excluded PMIDs are listed.",
    )
    return parser.parse_args()


def load_excluded_pmids(details_csv: Path) -> set[str]:
    if not details_csv.exists():
        return set()
    with details_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return {
            str(row.get("pmid", "")).strip()
            for row in csv.DictReader(file_obj)
            if row.get("status") == "excluded" and str(row.get("pmid", "")).strip()
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
    model_path = Path(args.model_path)
    segmented_dir = Path(args.segmented_dir)
    output_dir = Path(args.output_dir)
    json_dir = output_dir / "json"
    csv_path = output_dir / "method_info_table.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Local model path not found: {model_path}")
    if not segmented_dir.exists():
        raise FileNotFoundError(f"Segmented dir not found: {segmented_dir}")

    excluded_pmids = set() if args.include_excluded else load_excluded_pmids(Path(args.excluded_details_csv))
    skip_path = write_skip_file(excluded_pmids, output_dir)

    extractor = LocalModelMethodSectionInfoExtractor(
        model_path=model_path,
        max_method_chars=args.max_method_chars,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
    )

    records = extractor.extract_directory(
        segmented_dir=segmented_dir,
        output_json_dir=json_dir,
        output_csv_path=csv_path,
        limit=args.limit,
        overwrite=args.overwrite,
        skip_pmids=excluded_pmids,
        workers=1,
    )

    print("Local Qwen3.5 method information extraction test complete")
    print(f"  Model path: {model_path}")
    print(f"  Papers in this run: {len(records)}")
    print(f"  Excluded PMIDs skipped: {len(excluded_pmids)}")
    print(f"  Skip list: {skip_path}")
    print(f"  Per-paper JSON dir: {json_dir}")
    print(f"  Table CSV: {csv_path}")


if __name__ == "__main__":
    main()
