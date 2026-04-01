"""
DeepSeek-OCR 本地PDF识别脚本（独立于 GLM-OCR）。

说明：
- 使用本地模型目录（默认 `/storage/work/wuguowei/Bigmodel/DeepSeek-OCR-2`）。
- 输入 PDF，逐页渲染为图片后送入模型识别，输出 Markdown。
- 不依赖 glm_ocr.py 的任何实现。
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

from ..core.logger import get_module_logger
from ..utils.file_utils import ensure_dir, is_valid_pdf


@dataclass
class DeepSeekOCRResult:
    """DeepSeek-OCR 单文档处理结果。"""

    pdf_path: Path
    markdown_path: Optional[Path]
    status: str
    pages_processed: int
    processing_time: float
    error_message: Optional[str] = None


@dataclass
class _PageResult:
    """单页OCR结果（内部结构，不暴露到公共接口）。"""

    page_num: int
    success: bool
    markdown: str = ""
    error: Optional[str] = None


class DeepSeekOCRProcessor:
    """使用本地 DeepSeek-OCR 模型执行 PDF -> Markdown。"""

    def __init__(
        self,
        model_path: str = "/storage/work/wuguowei/Bigmodel/DeepSeek-OCR-2",
        output_dir: str = "./data/deepseek_markdown",
        device: str = "cuda",
        prompt: str = "<tr>\n<|grounding|>Convert the document to markdown. ",
        dpi: int = 200,
    ):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.device = device
        self.prompt = prompt
        self.dpi = dpi

        self.logger = get_module_logger("deepseek_ocr")
        ensure_dir(self.output_dir)

        self._model = None
        self._tokenizer = None
        self.max_retry = 2

    def _load_model(self) -> None:
        """加载 DeepSeek-OCR-2 模型和分词器（官方推荐方式）。"""
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DeepSeek-OCR 模型路径不存在: {self.model_path}. "
                "请确认本地模型目录是否正确。"
            )

        self.logger.info("加载 DeepSeek-OCR 模型: %s", self.model_path)

        # 加载分词器
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path), trust_remote_code=True
        )

        # 设置模型加载参数
        model_kwargs = {
            "trust_remote_code": True,
            "use_safetensors": True,
        }

        # 启用 Flash Attention 2 以提升性能
        if torch.cuda.is_available():
            model_kwargs["_attn_implementation"] = "flash_attention_2"
            model_kwargs["torch_dtype"] = torch.bfloat16

        # 使用 AutoModel 加载（官方推荐，兼容性最佳）
        self._model = AutoModel.from_pretrained(
            str(self.model_path), **model_kwargs
        )

        # 将模型设置为评估模式并移动到 GPU（如果可用）
        if torch.cuda.is_available():
            self._model = self._model.eval().cuda().to(torch.bfloat16)
        else:
            self._model = self._model.eval().to(self.device)

        self.logger.info("DeepSeek-OCR 模型加载成功（AutoModel + AutoTokenizer）")

    def _render_pdf_to_images(self, pdf_path: Path):
        """将 PDF 转为 PIL 图片，优先使用 pypdfium2。"""
        backends = ["pypdfium2", "fitz", "pdf2image"]
        last_error: Optional[Exception] = None

        for backend in backends:
            if importlib.util.find_spec(backend) is None:
                continue

            try:
                if backend == "pypdfium2":
                    pdfium = importlib.import_module("pypdfium2")
                    doc = pdfium.PdfDocument(str(pdf_path))
                    pages = []
                    scale = self.dpi / 72
                    for i in range(len(doc)):
                        page = doc[i]
                        bitmap = page.render(scale=scale)
                        pil_image = bitmap.to_pil()
                        pages.append(pil_image.convert("RGB"))
                    return pages

                if backend == "fitz":
                    fitz = importlib.import_module("fitz")
                    pil_image_mod = importlib.import_module("PIL.Image")
                    doc = fitz.open(str(pdf_path))
                    pages = []
                    zoom = self.dpi / 72
                    matrix = fitz.Matrix(zoom, zoom)
                    for i in range(len(doc)):
                        pix = doc[i].get_pixmap(matrix=matrix, alpha=False)
                        pil_image = pil_image_mod.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        pages.append(pil_image)
                    return pages

                if backend == "pdf2image":
                    module = importlib.import_module("pdf2image")
                    return module.convert_from_path(str(pdf_path), dpi=self.dpi)
            except Exception as exc:
                last_error = exc
                self.logger.warning("%s 渲染失败，尝试下一个后端: %s", backend, exc)

        raise RuntimeError(
            "无法将 PDF 渲染为图像。请安装以下任意依赖："
            "pypdfium2 或 pymupdf 或 pdf2image(poppler)。"
        ) from last_error

    def _ocr_single_image(self, image, image_path: Path, output_dir: Path) -> str:
        """
        使用官方推荐的方式进行单张图片的 OCR。

        model.infer() 不返回文本，而是将识别结果写入
        output_dir/result.mmd 文件，推理完成后读取该文件获取内容。

        注意：每页必须使用独立的 output_dir（调用方已保证），
        且推理前会清除旧的 result.mmd，防止读到上次残留结果。
        """
        if self._model is None or self._tokenizer is None:
            self._load_model()

        # 确保当前页面的输出目录存在
        output_dir.mkdir(parents=True, exist_ok=True)

        # 临时保存图像文件，因为模型的 infer 方法需要文件路径
        temp_img_path = output_dir / f"{image_path.stem}.png"
        image.save(temp_img_path, "PNG")

        # 推理前清除可能残留的旧 result.mmd，防止读到脏数据
        result_mmd_path = output_dir / "result.mmd"
        if result_mmd_path.exists():
            result_mmd_path.unlink()

        # 调用模型推理，结果写入 output_dir/result.mmd
        res = self._model.infer(
            self._tokenizer,
            prompt=self.prompt,
            image_file=str(temp_img_path),
            output_path=str(output_dir),
            base_size=1024,
            image_size=768,
            crop_mode=True,
            save_results=True,
        )

        # 读取模型写入的 result.mmd 作为本页 OCR 文本
        if not result_mmd_path.exists():
            raise FileNotFoundError(
                f"model.infer() 执行完毕但未生成预期的结果文件: {result_mmd_path}"
            )

        markdown = result_mmd_path.read_text(encoding="utf-8").strip()

        if not markdown:
            raise ValueError(
                f"result.mmd 文件存在但内容为空: {result_mmd_path}"
            )

        return markdown

    def _run_page_ocr_with_retry(
        self, image, image_path: Path, infer_output_dir: Path
    ) -> Tuple[bool, str, Optional[str]]:
        """单页OCR，带重试。"""
        last_error: Optional[str] = None
        for attempt in range(self.max_retry + 1):
            try:
                markdown = self._ocr_single_image(
                    image=image,
                    image_path=image_path,
                    output_dir=infer_output_dir,
                )
                if markdown.strip():
                    return True, markdown, None
                last_error = "空OCR结果"
            except Exception as exc:
                last_error = str(exc)
                self.logger.warning(
                    "第%s次OCR失败（%s）: %s", attempt + 1, image_path.name, exc
                )

            if attempt < self.max_retry:
                time.sleep(1)

        return False, "", last_error or "未知错误"

    def _build_markdown_with_page_markers(self, pdf_name: str, page_results: List[_PageResult]) -> str:
        """生成完整Markdown，包含失败页标记，便于后续人工复核。"""
        lines: List[str] = []
        lines.append(f"# {pdf_name}")
        lines.append(f"\n**Total Pages:** {len(page_results)}")
        failed_pages = [str(r.page_num) for r in page_results if not r.success]
        if failed_pages:
            lines.append(f"**Failed Pages:** {', '.join(failed_pages)}")
        else:
            lines.append("**Status:** All pages converted successfully")
        lines.append("\n" + "=" * 80 + "\n")

        for result in page_results:
            lines.append(f"\n")
            if result.success:
                lines.append(f"\n## Page {result.page_num}\n")
                lines.append(result.markdown.strip())
            else:
                lines.append(f"\n## ⚠️ Page {result.page_num} - OCR FAILED\n")
                lines.append("```")
                lines.append(f"ERROR: {result.error or 'Unknown error'}")
                lines.append("This page could not be converted to markdown.")
                lines.append(f"Please check the original PDF for page {result.page_num}.")
                lines.append("```")
            lines.append("\n" + "-" * 80 + "\n")
        return "\n".join(lines)

    def process_pdf(self, pdf_path: str, output_markdown: Optional[str] = None) -> DeepSeekOCRResult:
        start = time.time()
        pdf_file = Path(pdf_path)

        if not pdf_file.exists():
            return DeepSeekOCRResult(
                pdf_path=pdf_file,
                markdown_path=None,
                status="failed",
                pages_processed=0,
                processing_time=0.0,
                error_message=f"PDF不存在: {pdf_file}",
            )

        if not is_valid_pdf(pdf_file):
            return DeepSeekOCRResult(
                pdf_path=pdf_file,
                markdown_path=None,
                status="failed",
                pages_processed=0,
                processing_time=0.0,
                error_message=f"不是有效PDF: {pdf_file}",
            )

        markdown_path = Path(output_markdown) if output_markdown else self.output_dir / f"{pdf_file.stem}.md"
        ensure_dir(markdown_path.parent)

        try:
            images = self._render_pdf_to_images(pdf_file)
            page_results: List[_PageResult] = []
            tmp_root = self.output_dir / "tmp" / pdf_file.stem
            ensure_dir(tmp_root)
            try:
                for idx, image in enumerate(images, start=1):
                    self.logger.info("OCR处理中: %s - 第 %s 页", pdf_file.name, idx)
                    image_path = tmp_root / f"{pdf_file.stem}_page{idx:04d}.png"
                    image.save(image_path, "PNG")
                    # 每页使用独立子目录，确保各页的 result.mmd 不互相覆盖
                    infer_output_dir = tmp_root / f"page_{idx:04d}"

                    success, markdown, error = self._run_page_ocr_with_retry(
                        image=image,
                        image_path=image_path,
                        infer_output_dir=infer_output_dir,
                    )
                    page_results.append(
                        _PageResult(page_num=idx, success=success, markdown=markdown, error=error)
                    )
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

            complete_markdown = self._build_markdown_with_page_markers(pdf_file.stem, page_results)
            markdown_path.write_text(complete_markdown, encoding="utf-8")

            failed_pages = [str(p.page_num) for p in page_results if not p.success]
            elapsed = time.time() - start
            err = None
            if failed_pages:
                err = f"部分页面OCR失败: {', '.join(failed_pages)}"
            return DeepSeekOCRResult(
                pdf_path=pdf_file,
                markdown_path=markdown_path,
                status="success",
                pages_processed=len(images),
                processing_time=elapsed,
                error_message=err,
            )
        except Exception as exc:
            elapsed = time.time() - start
            self.logger.exception("DeepSeek-OCR处理失败: %s", exc)
            return DeepSeekOCRResult(
                pdf_path=pdf_file,
                markdown_path=None,
                status="failed",
                pages_processed=0,
                processing_time=elapsed,
                error_message=str(exc),
            )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepSeek-OCR 本地PDF识别脚本")
    parser.add_argument("pdf", help="输入PDF路径")
    parser.add_argument("--output", help="输出Markdown路径，默认写入 output_dir/<pdf名>.md", default=None)
    parser.add_argument(
        "--model-path",
        default="/storage/work/wuguowei/Bigmodel/DeepSeek-OCR-2",
        help="DeepSeek-OCR 本地模型目录",
    )
    parser.add_argument("--output-dir", default="./data/deepseek_markdown", help="默认输出目录")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda / cpu")
    parser.add_argument("--dpi", type=int, default=200, help="PDF渲染分辨率")
    parser.add_argument(
        "--prompt",
        default="<tr>\n<|grounding|>Convert the document to markdown. ",
        help="发送给模型的提示词",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    processor = DeepSeekOCRProcessor(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device,
        prompt=args.prompt,
        dpi=args.dpi,
    )
    result = processor.process_pdf(args.pdf, output_markdown=args.output)

    if result.status == "success":
        print(f"✅ OCR完成: {result.pdf_path}")
        print(f"📄 输出文件: {result.markdown_path}")
        print(f"📚 页数: {result.pages_processed}")
        print(f"⏱️ 用时: {result.processing_time:.2f}s")
    else:
        print(f"❌ OCR失败: {result.pdf_path}")
        print(f"原因: {result.error_message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()