"""
Parse downloaded Batch JSONL into per-paper JSON, CSV, and retry JSONL.

The parser accepts unordered Batch results and matches each response by its
unique custom_id. Invalid JSON and API errors are placed in retry_requests.jsonl.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from common import (
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    GLMMethodSectionInfoExtractor,
    extract_json_object,
    read_json,
    read_jsonl,
    write_csv,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse the 100-paper GLM Batch test results."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    return parser.parse_args()


def get_response_content(record: Dict[str, Any]) -> str:
    response = record.get("response", {})
    if not isinstance(response, dict):
        raise ValueError("Missing response object.")
    status_code = response.get("status_code")
    if status_code != 200:
        raise ValueError(f"Batch response status_code={status_code}")
    body = response.get("body", {})
    choices = body.get("choices", []) if isinstance(body, dict) else []
    if not choices:
        raise ValueError("Response contains no choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "") if isinstance(message, dict) else ""
    if not content:
        raise ValueError("Response message content is empty.")
    return str(content)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_path = output_dir / "batch_results.jsonl"
    error_path = output_dir / "batch_errors.jsonl"
    input_path = output_dir / "batch_requests_100.jsonl"
    manifest_path = output_dir / "batch_manifest_100.json"
    json_dir = output_dir / "parsed_json"
    csv_path = output_dir / "method_info_table.csv"
    retry_path = output_dir / "retry_requests.jsonl"
    report_path = output_dir / "parse_report.json"

    if not result_path.exists():
        raise FileNotFoundError(
            f"Download completed Batch results first: {result_path}"
        )
    if not input_path.exists() or not manifest_path.exists():
        raise FileNotFoundError("Batch input or manifest is missing.")

    manifest = read_json(manifest_path)
    manifest_by_id = {
        item["custom_id"]: item
        for item in manifest.get("items", [])
        if isinstance(item, dict) and item.get("custom_id")
    }
    input_requests = {
        item["custom_id"]: item
        for item in read_jsonl(input_path)
        if item.get("custom_id")
    }

    extractor = GLMMethodSectionInfoExtractor(
        api_key="batch-result-parsing-does-not-call-api",
        model_name=args.model,
    )
    result_records = read_jsonl(result_path)
    error_records = read_jsonl(error_path)
    batch_by_id = {
        record.get("custom_id"): record
        for record in result_records
        if record.get("custom_id")
    }
    batch_errors_by_id = {
        record.get("custom_id"): record
        for record in error_records
        if record.get("custom_id")
    }

    flat_records = []
    successful_ids = []
    failures: Dict[str, str] = {}
    json_dir.mkdir(parents=True, exist_ok=True)

    for custom_id, item in manifest_by_id.items():
        pmid = str(item.get("pmid", "")).strip()
        batch_record = batch_by_id.get(custom_id)
        if batch_record is None:
            if custom_id in batch_errors_by_id:
                failures[custom_id] = json.dumps(
                    batch_errors_by_id[custom_id],
                    ensure_ascii=False,
                )[:2000]
            else:
                failures[custom_id] = "No output record for custom_id."
            continue

        try:
            raw_text = get_response_content(batch_record)
            extracted = extract_json_object(raw_text)
            normalized = extractor.coerce_raw_json(pmid, extracted)
            flat_record = extractor.flatten_record(pmid, normalized)
            metadata = {
                "pmid": pmid,
                "source_file": item.get("source_file", ""),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "method_chars": item.get("method_chars", 0),
                "short_methods_threshold": manifest.get(
                    "short_methods_threshold",
                    2000,
                ),
                "source_sections_used": item.get(
                    "source_sections_used",
                    [],
                ),
                "client_type": "glm_batch",
                "model": args.model,
                "status": "success",
                "error": "",
                "custom_id": custom_id,
                "batch_response_id": batch_record.get("id", ""),
                "raw_generation_preview": raw_text[:1000],
            }
            write_json(
                {
                    "pmid": pmid,
                    "extracted_json": normalized,
                    "flat_record": flat_record,
                    "metadata": metadata,
                },
                json_dir / f"paper_{pmid}_method_info.json",
            )
            flat_records.append(flat_record)
            successful_ids.append(custom_id)
        except Exception as exc:
            failures[custom_id] = f"{type(exc).__name__}: {exc}"

    flat_records.sort(
        key=lambda row: (
            not str(row.get("PMID", "")).isdigit(),
            str(row.get("PMID", "")),
        )
    )
    write_csv(flat_records, csv_path)
    retry_requests = [
        input_requests[custom_id]
        for custom_id in manifest_by_id
        if custom_id in failures and custom_id in input_requests
    ]
    write_jsonl(retry_requests, retry_path)

    report = {
        "expected_requests": len(manifest_by_id),
        "downloaded_success_records": len(result_records),
        "downloaded_error_records": len(error_records),
        "parsed_successfully": len(successful_ids),
        "retry_request_count": len(retry_requests),
        "failures": failures,
        "parsed_json_dir": str(json_dir),
        "csv_path": str(csv_path),
        "retry_jsonl": str(retry_path),
    }
    write_json(report, report_path)

    print("Batch results parsed")
    print(f"  Expected: {report['expected_requests']}")
    print(f"  Parsed successfully: {report['parsed_successfully']}")
    print(f"  Retry requests: {report['retry_request_count']}")
    print(f"  Per-paper JSON: {json_dir}")
    print(f"  CSV: {csv_path}")
    print(f"  Report: {report_path}")
    print(f"  Retry JSONL: {retry_path}")


if __name__ == "__main__":
    main()
