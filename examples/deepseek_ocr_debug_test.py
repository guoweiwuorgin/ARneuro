"""
DeepSeek-OCR 专用调试测试脚本。

用途：
1. 批量抽样 PDF 调用 DeepSeekOCRProcessor。
2. 验证调用链（PDF->图片->模型->Markdown）是否可用。
3. 生成 JSON/CSV 报告，便于核查成功率、耗时与失败原因。

示例：
python examples/deepseek_ocr_debug_test.py \
  --pdf-dir /path/to/pdfs \
  --output-dir ./debug_output \
  --model-path /storage/work/wuguowei/Bigmodel/deepseek-ocr \
  --sample-size 5
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from ARneuro.ocr_processing.deepseek_ocr import DeepSeekOCRProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR 专用测试脚本")
    parser.add_argument("--pdf-dir", required=True, help="待测试PDF目录")
    parser.add_argument("--output-dir", default="./deepseek_ocr_debug_output", help="调试输出目录")
    parser.add_argument(
        "--model-path",
        default="/storage/work/wuguowei/Bigmodel/deepseek-ocr",
        help="DeepSeek-OCR 本地模型路径",
    )
    parser.add_argument("--sample-size", type=int, default=5, help="抽样PDF数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", default="cuda", help="推理设备: cuda/cpu")
    parser.add_argument("--dpi", type=int, default=200, help="PDF渲染DPI")
    parser.add_argument(
        "--prompt",
        default="请对该页做高保真OCR，输出结构化Markdown，保留表格。",
        help="DeepSeek OCR提示词",
    )
    return parser.parse_args()


def choose_pdfs(pdf_dir: Path, sample_size: int, seed: int) -> List[Path]:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"目录中没有PDF: {pdf_dir}")

    random.seed(seed)
    if len(pdfs) <= sample_size:
        return pdfs
    return random.sample(pdfs, sample_size)


def export_reports(report_rows: List[Dict[str, Any]], summary: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_report = output_dir / "deepseek_ocr_debug_report.json"
    csv_report = output_dir / "deepseek_ocr_debug_report.csv"

    json_report.write_text(
        json.dumps({"summary": summary, "details": report_rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    headers = [
        "pdf_name",
        "pdf_path",
        "status",
        "markdown_path",
        "pages_processed",
        "processing_time",
        "error_message",
    ]
    with csv_report.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in report_rows:
            writer.writerow({k: row.get(k, "") for k in headers})

    print(f"[DONE] JSON报告: {json_report}")
    print(f"[DONE] CSV报告: {csv_report}")


def main() -> None:
    args = parse_args()

    pdf_dir = Path(args.pdf_dir)
    output_dir = Path(args.output_dir)
    markdown_dir = output_dir / "markdown"

    selected = choose_pdfs(pdf_dir, args.sample_size, args.seed)

    processor = DeepSeekOCRProcessor(
        model_path=args.model_path,
        output_dir=str(markdown_dir),
        device=args.device,
        prompt=args.prompt,
        dpi=args.dpi,
    )

    rows: List[Dict[str, Any]] = []
    print(f"[INFO] 待测PDF目录: {pdf_dir}")
    print(f"[INFO] 抽样数量: {len(selected)}")
    print(f"[INFO] Markdown输出目录: {markdown_dir}")

    for idx, pdf_path in enumerate(selected, start=1):
        print(f"[{idx}/{len(selected)}] 测试: {pdf_path.name}")
        result = processor.process_pdf(str(pdf_path))
        rows.append(
            {
                "pdf_name": pdf_path.name,
                "pdf_path": str(pdf_path),
                "status": result.status,
                "markdown_path": str(result.markdown_path) if result.markdown_path else "",
                "pages_processed": result.pages_processed,
                "processing_time": round(result.processing_time, 2),
                "error_message": result.error_message or "",
            }
        )

    success_count = sum(1 for r in rows if r["status"] == "success")
    failed_count = len(rows) - success_count
    summary = {
        "pdf_dir": str(pdf_dir),
        "output_dir": str(output_dir),
        "model_path": args.model_path,
        "sample_size": len(rows),
        "success_count": success_count,
        "failed_count": failed_count,
        "avg_processing_time": round(sum(r["processing_time"] for r in rows) / len(rows), 2) if rows else 0.0,
    }

    export_reports(rows, summary, output_dir)
    print(
        "[SUMMARY] "
        f"success={summary['success_count']} "
        f"failed={summary['failed_count']} "
        f"avg_time={summary['avg_processing_time']}s"
    )


if __name__ == "__main__":
    main()
