"""步骤3：对DeepSeek OCR生成的Markdown做文档分割，构建文档库并统计各part分割情况。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List
import sys

# 保持依赖路径的添加
sys.path.append("/storage/work/wuguowei/reviewer/Code_tools/")
import os
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)
from ARneuro.text_processing.document_segmentation import DocumentSegmenter


CANONICAL_PARTS = {
    "abstract": ["abstract", "summary"],
    "introduction": ["introduction", "background"],
    "methods": ["methods", "materials and methods", "methodology"],
    "results": ["results", "findings"],
    "discussion": ["discussion", "conclusion"],
    "references": ["references", "bibliography"],
}


def map_to_part(section_title: str) -> str:
    lower = section_title.lower()
    for part, keys in CANONICAL_PARTS.items():
        if any(k in lower for k in keys):
            return part
    return "other"


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
        doc_structure, tables, tables_info, tables_annotation = segmenter.parse_markdown_file(str(md_file))
        validation = segmenter.validate_sections(doc_structure)

        part_counter: Counter = Counter()
        for title in doc_structure.keys():
            part_counter[map_to_part(title)] += 1
        global_part_counter.update(part_counter)

        output_json = segmented_dir / f"{md_file.stem}_segmentation.json"
        output_json.write_text(
            json.dumps(
                {
                    "source_markdown": str(md_file),
                    "document_structure": doc_structure,
                    "validation": validation,
                    "tables": tables,
                    "tables_info": tables_info,
                    "tables_annotation": tables_annotation,
                    "part_counter": dict(part_counter),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        document_index.append(
            {
                "markdown": str(md_file),
                "segmentation_json": str(output_json),
                "num_sections": len(doc_structure),
                "num_tables": len(tables),
                "part_counter": dict(part_counter),
                "validation": validation,
            }
        )

    index_path = output_dir / "document_library_index.json"
    stats_path = output_dir / "segmentation_part_stats.json"
    index_path.write_text(json.dumps(document_index, ensure_ascii=False, indent=2), encoding="utf-8")
    stats_path.write_text(
        json.dumps(
            {
                "documents": len(document_index),
                "global_part_counter": dict(global_part_counter),
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
