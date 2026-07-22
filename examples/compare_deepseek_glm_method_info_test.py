"""Compare the 10-paper DeepSeek and GLM Methods extraction results."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
DEFAULT_DEEPSEEK_DIR = STEP3_DIR / "method_info_extraction_deepseek_v4_flash_test" / "json"
DEFAULT_GLM_DIR = STEP3_DIR / "method_info_extraction_glm45_air_full" / "json"
DEFAULT_OUTPUT_DIR = STEP3_DIR / "method_info_model_comparison_test"

IGNORED_FIELDS = {"evidence", "missing_or_uncertain_fields"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare DeepSeek and GLM method-info JSON.")
    parser.add_argument("--deepseek-json-dir", default=str(DEFAULT_DEEPSEEK_DIR))
    parser.add_argument("--glm-json-dir", default=str(DEFAULT_GLM_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def load_results(directory: Path) -> dict[str, dict[str, Any]]:
    results = {}
    for path in directory.glob("paper_*_method_info.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("metadata", {}).get("status") != "success":
            continue
        pmid = str(data.get("pmid", "")).strip()
        extracted = data.get("extracted_json", {})
        if pmid and isinstance(extracted, dict):
            results[pmid] = extracted
    return results


def normalize_scalar(value: Any) -> str:
    text = str(value if value is not None else "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(normalize_scalar(item) for item in value if normalize_scalar(item))
    if isinstance(value, dict):
        return {
            str(key): normalize_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, str) and "," in value:
        parts = [normalize_scalar(item) for item in value.split(",")]
        return sorted(item for item in parts if item)
    return normalize_scalar(value)


def main() -> None:
    args = parse_args()
    deepseek_dir = Path(args.deepseek_json_dir)
    glm_dir = Path(args.glm_json_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not deepseek_dir.exists() or not glm_dir.exists():
        raise FileNotFoundError("DeepSeek or GLM test JSON directory is missing.")

    deepseek = load_results(deepseek_dir)
    glm = load_results(glm_dir)
    common_pmids = sorted(set(deepseek) & set(glm), key=lambda value: (not value.isdigit(), value))
    all_fields = sorted(
        (
            set().union(*(deepseek[pmid].keys() for pmid in common_pmids))
            | set().union(*(glm[pmid].keys() for pmid in common_pmids))
        )
        - IGNORED_FIELDS
    ) if common_pmids else []

    field_stats = defaultdict(lambda: {"matched": 0, "total": 0})
    paper_rows = []
    difference_rows = []
    for pmid in common_pmids:
        matched = 0
        for field in all_fields:
            deepseek_value = deepseek[pmid].get(field, "")
            glm_value = glm[pmid].get(field, "")
            same = normalize_value(deepseek_value) == normalize_value(glm_value)
            field_stats[field]["total"] += 1
            if same:
                field_stats[field]["matched"] += 1
                matched += 1
            else:
                difference_rows.append(
                    {
                        "PMID": pmid,
                        "field": field,
                        "deepseek_value": json.dumps(deepseek_value, ensure_ascii=False),
                        "glm_value": json.dumps(glm_value, ensure_ascii=False),
                    }
                )
        paper_rows.append(
            {
                "PMID": pmid,
                "matched_fields": matched,
                "total_fields": len(all_fields),
                "agreement_rate": round(matched / len(all_fields), 4) if all_fields else 0,
            }
        )

    field_rows = []
    for field in all_fields:
        stats = field_stats[field]
        field_rows.append(
            {
                "field": field,
                "matched": stats["matched"],
                "total": stats["total"],
                "agreement_rate": round(stats["matched"] / stats["total"], 4)
                if stats["total"] else 0,
            }
        )
    field_rows.sort(key=lambda row: (row["agreement_rate"], row["field"]))
    total_cells = len(common_pmids) * len(all_fields)
    matched_cells = sum(row["matched"] for row in field_rows)
    summary = {
        "deepseek_successful_pmids": len(deepseek),
        "glm_successful_pmids": len(glm),
        "common_pmids": len(common_pmids),
        "compared_fields": len(all_fields),
        "matched_cells": matched_cells,
        "total_cells": total_cells,
        "overall_agreement_rate": round(matched_cells / total_cells, 4)
        if total_cells else 0,
        "ignored_fields": sorted(IGNORED_FIELDS),
    }

    def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(
        output_dir / "paper_agreement.csv",
        paper_rows,
        ["PMID", "matched_fields", "total_fields", "agreement_rate"],
    )
    write_csv(
        output_dir / "field_agreement.csv",
        field_rows,
        ["field", "matched", "total", "agreement_rate"],
    )
    write_csv(
        output_dir / "differences.csv",
        difference_rows,
        ["PMID", "field", "deepseek_value", "glm_value"],
    )
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Comparison output: {output_dir}")


if __name__ == "__main__":
    main()
