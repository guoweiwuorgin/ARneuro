"""
本地调试脚本：随机抽取10个PDF，使用GLM-OCR API模式进行OCR并评估Markdown质量。

注意：
1. 此脚本仅用于本地调试，不会自动推送到任何远程仓库。
2. 默认使用用户提供的Windows路径与API Key，可按需修改。
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, List

from ARneuro.ocr_processing.glm_ocr import GLMOCRProcessor, OCRStatus
from ARneuro.config.config_manager import ConfigManager


# ===== 用户本地调试参数（可直接修改）=====
GLM_API_KEY = "a6d6b245776a4041bb5e7492ba0a179d.O8uspuG7JTLrydE8"
PDF_DIR = Path(r"D:\language_template\reviewer\allrights_pdf")
OUTPUT_DIR = Path(r"D:\language_template\reviewer\ARneuro_test")
SAMPLE_SIZE = 10


def _build_debug_config(output_dir: Path, api_key: str) -> Dict[str, Any]:
    """构造最小可运行配置（API模式）。"""
    return {
        "ocr_processing": {
            "backend": "api",
            "output_dir": str(output_dir),
            "api_key": api_key,
            "api_base_url": "https://open.bigmodel.cn/api/paas/v4/layout_parsing",
            "api_model": "glm-ocr",
            "api_timeout": 300,
            "api_use_base64": True,
            "api_return_crop_images": False,
            "api_need_layout_visualization": False,
        }
    }


def _choose_pdfs(pdf_dir: Path, sample_size: int) -> List[Path]:
    """随机选择PDF文件。"""
    all_pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not all_pdfs:
        raise FileNotFoundError(f"目录中未找到PDF文件: {pdf_dir}")

    if len(all_pdfs) <= sample_size:
        return all_pdfs

    return random.sample(all_pdfs, sample_size)


def _quality_tag(quality_score: float, text_length: int) -> str:
    """简化质量标签，用于人工快速审阅。"""
    if text_length < 500:
        return "too_short"
    if quality_score >= 80:
        return "good"
    if quality_score >= 60:
        return "medium"
    return "poor"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 使用临时配置文件，避免影响主配置
    config_path = OUTPUT_DIR / "glmocr_api_debug_config.json"
    config_data = _build_debug_config(OUTPUT_DIR, GLM_API_KEY)

    # ConfigManager默认读取yaml，这里通过set注入配置后保存为yaml
    cfg = ConfigManager(str(OUTPUT_DIR / "glmocr_api_debug_config.yaml"))
    cfg.config = config_data
    cfg.save()
    config_path = Path(cfg.config_path)

    selected_pdfs = _choose_pdfs(PDF_DIR, SAMPLE_SIZE)
    print(f"[INFO] 总PDF数: {len(list(PDF_DIR.glob('*.pdf')))}")
    print(f"[INFO] 本次抽样数量: {len(selected_pdfs)}")
    print(f"[INFO] 输出目录: {OUTPUT_DIR}")

    processor = GLMOCRProcessor(config_path=str(config_path), backend="api")

    report_rows = []
    for i, pdf_path in enumerate(selected_pdfs, 1):
        print(f"[{i}/{len(selected_pdfs)}] OCR处理中: {pdf_path.name}")
        result = processor.process_pdf(pdf_path)

        row = {
            "pdf_name": pdf_path.name,
            "pdf_path": str(pdf_path),
            "status": result.status.value if hasattr(result.status, "value") else str(result.status),
            "markdown_path": str(result.markdown_path) if result.markdown_path else "",
            "text_length": result.text_length,
            "pages_processed": result.pages_processed,
            "quality_score": round(result.quality_score, 2),
            "quality_level": result.quality_level.value if hasattr(result.quality_level, "value") else str(result.quality_level),
            "language": result.language,
            "processing_time": round(result.processing_time, 2),
            "backend": result.metadata.get("backend"),
            "api_model": result.metadata.get("model"),
            "manual_quality_tag": _quality_tag(result.quality_score, result.text_length),
            "error_message": result.error_message or "",
        }
        report_rows.append(row)

    success_rows = [r for r in report_rows if r["status"] == OCRStatus.SUCCESS.value]
    fail_rows = [r for r in report_rows if r["status"] != OCRStatus.SUCCESS.value]

    summary = {
        "pdf_dir": str(PDF_DIR),
        "output_dir": str(OUTPUT_DIR),
        "sample_size": len(report_rows),
        "success_count": len(success_rows),
        "failed_count": len(fail_rows),
        "avg_quality_score_success": round(
            sum(r["quality_score"] for r in success_rows) / len(success_rows), 2
        ) if success_rows else 0.0,
        "avg_text_length_success": round(
            sum(r["text_length"] for r in success_rows) / len(success_rows), 2
        ) if success_rows else 0.0,
    }

    json_report = OUTPUT_DIR / "ocr_quality_report.json"
    csv_report = OUTPUT_DIR / "ocr_quality_report.csv"

    with open(json_report, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": report_rows}, f, ensure_ascii=False, indent=2)

    # 简单CSV输出
    headers = list(report_rows[0].keys()) if report_rows else [
        "pdf_name", "status", "quality_score", "quality_level", "text_length", "error_message"
    ]
    with open(csv_report, "w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")
        for row in report_rows:
            values = [str(row.get(h, "")).replace(",", "，") for h in headers]
            f.write(",".join(values) + "\n")

    print("[DONE] OCR调试完成")
    print(f"[DONE] JSON报告: {json_report}")
    print(f"[DONE] CSV报告: {csv_report}")
    print(f"[DONE] 成功/失败: {summary['success_count']}/{summary['failed_count']}")


if __name__ == "__main__":
    main()
