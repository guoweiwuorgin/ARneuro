"""步骤4：基于Markdown文件系统提取激活表格（brain activation tables）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from table_processing.table_extractor import TableExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="步骤4：提取激活表格")
    parser.add_argument("--markdown-dir", default=f"{PROJECT_ROOT}/examples/data/workflow/step2_markdown", help="markdown目录")
    parser.add_argument("--output-dir", default=f"{PROJECT_ROOT}/examples/data/workflow/step4_activation_tables", help="输出目录")
    parser.add_argument("--context-lines", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    markdown_dir = Path(args.markdown_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = TableExtractor(config={})
    md_files = sorted(markdown_dir.glob("*.md"))

    all_results: List[Dict] = []
    total_activation_tables = 0

    for md in md_files:
        tables_info = extractor.extract_tables_from_markdown(str(md), context_lines=args.context_lines)
        categorized = extractor.categorize_tables(tables_info)
        activation_tables = categorized.get("brain_activation", [])
        total_activation_tables += len(activation_tables)

        out_file = output_dir / f"{md.stem}_activation_tables.json"
        out_file.write_text(
            json.dumps(
                {
                    "source_markdown": str(md),
                    "total_tables": len(tables_info),
                    "activation_tables": activation_tables,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        all_results.append(
            {
                "markdown": str(md),
                "total_tables": len(tables_info),
                "activation_table_count": len(activation_tables),
                "activation_tables_json": str(out_file),
            }
        )

    summary = {
        "documents": len(all_results),
        "total_activation_tables": total_activation_tables,
        "results": all_results,
    }
    summary_path = output_dir / "activation_tables_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
