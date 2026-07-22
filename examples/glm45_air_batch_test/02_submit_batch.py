"""Upload the prepared JSONL file and create a BigModel Batch task."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_DIR,
    create_zai_client,
    get_api_key,
    object_value,
    sdk_object_to_dict,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload and submit the 100-paper GLM Batch test."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--keep-input-file",
        action="store_true",
        help="Keep the uploaded input file on BigModel after completion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    input_path = output_dir / "batch_requests_100.jsonl"
    state_path = output_dir / "batch_job.json"
    if not input_path.exists():
        raise FileNotFoundError(
            f"Prepare the Batch input first: {input_path}"
        )
    if state_path.exists():
        raise FileExistsError(
            f"Batch state already exists: {state_path}. "
            "Remove or rename it before creating another task."
        )

    client = create_zai_client(get_api_key(args.api_key))
    with input_path.open("rb") as input_file:
        uploaded = client.files.create(file=input_file, purpose="batch")
    input_file_id = object_value(uploaded, "id")
    if not input_file_id:
        raise RuntimeError(f"Upload returned no file id: {uploaded}")

    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v4/chat/completions",
        auto_delete_input_file=not args.keep_input_file,
        metadata={
            "description": "100-paper Methods information extraction test",
            "project": "reviewer_method_info",
            "model": "glm-4-0520",
        },
    )
    batch_id = object_value(batch, "id")
    if not batch_id:
        raise RuntimeError(f"Batch creation returned no batch id: {batch}")

    state = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_path": str(input_path),
        "input_file_id": input_file_id,
        "batch_id": batch_id,
        "batch": sdk_object_to_dict(batch),
    }
    write_json(state, state_path)
    print("Batch submitted")
    print(f"  Input file ID: {input_file_id}")
    print(f"  Batch ID: {batch_id}")
    print(f"  State: {state_path}")


if __name__ == "__main__":
    main()
