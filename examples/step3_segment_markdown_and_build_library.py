"""步骤3：对DeepSeek OCR生成的Markdown做文档分割，构建文档库并统计各part分割情况。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List
from text_processing.document_segmentation import DocumentSegmenter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from text_processing.document_segmentation import DocumentSegmenter


CANONICAL_PARTS = [
    "Title",
    "Author",
    "Keywords",
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "References",
    "Acknowledgements",
    "Other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="步骤3：文档分割与文档库构建")
    parser.add_argument("--markdown-dir", default="./data/workflow/step2_markdown", help="步骤2产出的markdown目录")
    parser.add_argument("--output-dir", default="./data/workflow/step3_library", help="文档库输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_dir = Path(args.markdown_dir)
    output_dir = Path(args.output_dir)
    segmented_dir = output_dir / "segmented"
    segmented_dir.mkdir(parents=True, exist_ok=True)

    segmenter = DocumentSegmenter(config={})
    md_files = sorted(markdown_dir.glob("*.md"))

    global_part_counter: Counter = Counter()
    document_index: List[Dict] = []

    for md_file in md_files:
        structured, meta = segmenter.segment_document(str(md_file))
        part_counter: Counter = Counter(meta.get("sections_detected", []))
        global_part_counter.update(part_counter)

        content_json, meta_json = segmenter.save_structured_outputs(
            structured=structured,
            metadata=meta,
            output_dir=str(segmented_dir),
            base_name=md_file.stem,
        )

        document_index.append(
            {
                "markdown": str(md_file),
                "structured_content_json": str(content_json),
                "structured_meta_json": str(meta_json),
                "required_sections_complete": meta.get("required_sections_complete", False),
                "strategy": meta.get("strategy", "rule_based"),
                "num_blocks": meta.get("block_count", 0),
                "num_tables": meta.get("table_count", 0),
                "part_counter": dict(part_counter),
            }
        )

    index_path = output_dir / "document_library_index.json"
    stats_path = output_dir / "segmentation_part_stats.json"
    index_path.write_text(json.dumps(document_index, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "documents": len(document_index),
                "global_part_counter": {part: global_part_counter.get(part, 0) for part in CANONICAL_PARTS},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "documents": len(document_index),
                "index": str(index_path),
                "part_stats": str(stats_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
