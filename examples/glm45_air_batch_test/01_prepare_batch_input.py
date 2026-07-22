"""
Prepare a 100-paper GLM-4.5-Air Batch API JSONL input file.

This step is fully offline and does not call the BigModel API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    DEFAULT_EXCLUDED_DETAILS_CSV,
    DEFAULT_MODEL_NAME,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEGMENTED_DIR,
    MethodSectionInfoExtractor,
    load_excluded_pmids,
    pmid_from_path,
    read_json,
    write_json,
    write_jsonl,
)
DEFAULT_MODEL_NAME = "glm-4-0520"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare 100 GLM-4.5-Air Batch extraction requests."
    )
    parser.add_argument("--segmented-dir", default=str(DEFAULT_SEGMENTED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--excluded-details-csv", default=str(DEFAULT_EXCLUDED_DETAILS_CSV))
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-method-chars", type=int, default=50000)
    parser.add_argument("--short-methods-threshold", type=int, default=2000)
    parser.add_argument("--supplementary-section-chars", type=int, default=12000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    segmented_dir = Path(args.segmented_dir)
    output_dir = Path(args.output_dir)
    input_path = output_dir / "batch_requests_100.jsonl"
    manifest_path = output_dir / "batch_manifest_100.json"

    if not segmented_dir.exists():
        raise FileNotFoundError(segmented_dir)
    if args.limit < 1 or args.limit > 50000:
        raise ValueError("--limit must be between 1 and 50000")

    excluded_pmids = load_excluded_pmids(Path(args.excluded_details_csv))
    files = [
        path
        for path in sorted(
            segmented_dir.glob("paper_*_structured_content.json")
        )
        if pmid_from_path(path) not in excluded_pmids
    ][: args.limit]
    if len(files) < args.limit:
        raise RuntimeError(
            f"Only {len(files)} eligible papers found; requested {args.limit}."
        )

    extractor = MethodSectionInfoExtractor(
        config={},
        client_type="glm",
        model_name=args.model,
        max_method_chars=args.max_method_chars,
        short_methods_threshold=args.short_methods_threshold,
        supplementary_section_chars=args.supplementary_section_chars,
        max_tokens=args.max_tokens,
    )

    requests = []
    manifest_items = []
    seen_custom_ids = set()
    for index, content_path in enumerate(files, start=1):
        pmid = pmid_from_path(content_path)
        custom_id = f"paper-{pmid}"
        if custom_id in seen_custom_ids:
            raise RuntimeError(f"Duplicate custom_id: {custom_id}")
        seen_custom_ids.add(custom_id)

        content = read_json(content_path)
        methods = str(content.get("Methods", "") or "")
        messages = extractor.build_prompt(pmid, content)
        _payload, sections_used = extractor.build_source_payload(pmid, content)
        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v4/chat/completions",
            "body": {
                "model": args.model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "response_format": {"type": "json_object"},
                "stream": False,
            },
        }
        requests.append(request)
        manifest_items.append(
            {
                "index": index,
                "custom_id": custom_id,
                "pmid": pmid,
                "source_file": str(content_path),
                "method_chars": len(methods),
                "source_sections_used": sections_used,
            }
        )

    write_jsonl(requests, input_path)
    size_bytes = input_path.stat().st_size
    manifest = {
        "model": args.model,
        "request_count": len(requests),
        "input_file": str(input_path),
        "input_size_bytes": size_bytes,
        "input_size_mb": round(size_bytes / 1024 / 1024, 3),
        "excluded_pmids_count": len(excluded_pmids),
        "short_methods_threshold": args.short_methods_threshold,
        "items": manifest_items,
    }
    write_json(manifest, manifest_path)

    if size_bytes > 100 * 1024 * 1024:
        raise RuntimeError("Generated Batch input exceeds the 100 MB limit.")

    print("Batch input prepared")
    print(f"  Requests: {len(requests)}")
    print(f"  Model: {args.model}")
    print(f"  Size: {manifest['input_size_mb']} MB")
    print(f"  JSONL: {input_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
