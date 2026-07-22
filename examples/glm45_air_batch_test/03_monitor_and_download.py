"""Retrieve Batch status and download successful/error result JSONL files."""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from common import (
    DEFAULT_OUTPUT_DIR,
    create_zai_client,
    get_api_key,
    object_value,
    read_json,
    sdk_object_to_dict,
    write_json,
)


TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor and download the 100-paper GLM Batch test."
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Poll until the Batch reaches a terminal status.",
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def download_file(client, file_id: str, destination: Path) -> None:
    content = client.files.content(file_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_to_file = getattr(content, "write_to_file", None)
    if callable(write_to_file):
        write_to_file(str(destination))
        return
    raw_content = getattr(content, "content", content)
    if isinstance(raw_content, str):
        destination.write_text(raw_content, encoding="utf-8")
    elif isinstance(raw_content, bytes):
        destination.write_bytes(raw_content)
    else:
        raise RuntimeError(f"Unsupported downloaded content: {type(content)}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    state_path = output_dir / "batch_job.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Submit the Batch first: {state_path}")
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be >= 1")

    state = read_json(state_path)
    batch_id = str(state.get("batch_id", "")).strip()
    if not batch_id:
        raise ValueError(f"No batch_id in {state_path}")

    client = create_zai_client(get_api_key(args.api_key))
    while True:
        batch = client.batches.retrieve(batch_id)
        status = str(object_value(batch, "status", "unknown"))
        state["last_checked_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        state["batch"] = sdk_object_to_dict(batch)
        write_json(state, state_path)
        print(f"Batch {batch_id}: {status}")

        if status in TERMINAL_STATUSES or not args.wait:
            break
        time.sleep(args.poll_seconds)

    if status != "completed":
        print("No result download: Batch is not completed.")
        return

    output_file_id = object_value(batch, "output_file_id")
    error_file_id = object_value(batch, "error_file_id")
    if output_file_id:
        output_path = output_dir / "batch_results.jsonl"
        download_file(client, str(output_file_id), output_path)
        print(f"Successful results: {output_path}")
    if error_file_id:
        error_path = output_dir / "batch_errors.jsonl"
        download_file(client, str(error_file_id), error_path)
        print(f"Request errors: {error_path}")


if __name__ == "__main__":
    main()
