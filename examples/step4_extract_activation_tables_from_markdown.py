"""步骤4：基于新结构化输出批量提取激活表格（brain activation tables）。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from table_processing.table_pipeline import TableProcessingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="步骤4：从structured_content提取激活表格")
    parser.add_argument(
        "--structured-dir",
        default=f"{PROJECT_ROOT}/examples/data/workflow/step3_library/segmented",
        help="step3 结构化输出目录（包含 *_structured_content.json）",
    )
    parser.add_argument(
        "--markdown-dir",
        default=f"{PROJECT_ROOT}/examples/data/workflow/step2_markdown",
        help="原始 markdown 目录（用于补充上下文，可选）",
    )
    parser.add_argument(
        "--output-dir",
        default=f"{PROJECT_ROOT}/examples/data/workflow/step4_activation_tables",
        help="输出目录",
    )
    parser.add_argument("--model-name", default="deepseek-chat")
    parser.add_argument("--no-llm", action="store_true", help="不启用LLM，仅做规则分类")
    return parser.parse_args()


def _guess_source_markdown(structured_file: Path, markdown_dir: Path) -> Path | None:
    stem = structured_file.name.replace("_structured_content.json", "")
    candidate = markdown_dir / f"{stem}.md"
    return candidate if candidate.exists() else None


def main() -> None:
    args = parse_args()
    structured_dir = Path(args.structured_dir)
    markdown_dir = Path(args.markdown_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = TableProcessingPipeline(config={"model_name": args.model_name})

    structured_files = sorted(structured_dir.glob("*_structured_content.json"))
    all_results: List[Dict] = []
    total_activation_tables = 0

    for sf in structured_files:
        source_md = _guess_source_markdown(sf, markdown_dir)
        doc_out = output_dir / sf.stem.replace("_structured_content", "")

        results = pipeline.process_structured_content_file(
            structured_content_file=str(sf),
            source_markdown=str(source_md) if source_md else None,
            output_dir=str(doc_out),
            process_brain_tables=not args.no_llm,
            model_name=args.model_name,
        )

        brain_tables = results.get("brain_activation_tables", {}).get("brain_tables", [])
        total_activation_tables += len(brain_tables)

        all_results.append(
            {
                "structured_content": str(sf),
                "source_markdown": str(source_md) if source_md else None,
                "total_tables": results.get("tables_extracted", 0),
                "activation_table_count": len(brain_tables),
                "results_json": str(Path(results.get("output_files", {}).get("full_results", ""))),
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
