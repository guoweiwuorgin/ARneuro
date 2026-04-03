"""步骤2：读取步骤1下载的PDF，使用DeepSeek OCR转换为Markdown。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

from ARneuro.ocr_processing.deepseek_ocr import DeepSeekOCRProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="步骤2：DeepSeek OCR批量转换")
    parser.add_argument("--step1-results", default="./data/workflow/step1/screening_results.csv", help="步骤1输出CSV")
    parser.add_argument("--pdf-dir", default=None, help="可选：直接指定PDF目录（覆盖step1结果）")
    parser.add_argument("--output-dir", default="./data/workflow/step2_markdown", help="Markdown输出目录")
    parser.add_argument("--model-path", default="/storage/work/wuguowei/Bigmodel/DeepSeek-OCR-2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dpi", type=int, default=200)
    return parser.parse_args()


def collect_pdf_paths(step1_csv: Path, pdf_dir: Path | None) -> List[Path]:
    if pdf_dir is not None:
        return sorted(pdf_dir.glob("*.pdf"))

    paths: List[Path] = []
    with step1_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdf_path = str(row.get("PDFPath", "")).strip()
            if pdf_path:
                p = Path(pdf_path)
                if p.exists():
                    paths.append(p)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_paths = collect_pdf_paths(Path(args.step1_results), Path(args.pdf_dir) if args.pdf_dir else None)
    processor = DeepSeekOCRProcessor(
        model_path=args.model_path,
        output_dir=str(output_dir),
        device=args.device,
        dpi=args.dpi,
    )

    rows: List[Dict] = []
    for pdf_path in pdf_paths:
        result = processor.process_pdf(str(pdf_path))
        rows.append(
            {
                "pdf": str(pdf_path),
                "status": result.status,
                "markdown": str(result.markdown_path) if result.markdown_path else "",
                "pages_processed": result.pages_processed,
                "processing_time": round(result.processing_time, 2),
                "error": result.error_message or "",
            }
        )

    report_json = output_dir / "ocr_report.json"
    report_csv = output_dir / "ocr_report.csv"
    report_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    with report_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["pdf", "status", "markdown", "pages_processed", "processing_time", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "total_pdf": len(rows),
                "success": len([r for r in rows if r["status"] == "success"]),
                "failed": len([r for r in rows if r["status"] != "success"]),
                "report_json": str(report_json),
                "report_csv": str(report_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
