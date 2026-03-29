"""
DeepSeek-OCR 本地PDF识别脚本（独立于 GLM-OCR）。

说明：
- 使用本地模型目录（默认 `/storage/work/wuguowei/Bigmodel/deepseek-ocr`）。
- 输入 PDF，逐页渲染为图片后送入模型识别，输出 Markdown。
- 不依赖 glm_ocr.py 的任何实现。
"""

from __future__ import annotations

import argparse
import inspect
import importlib
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
        model_path: str = "/storage/work/wuguowei/Bigmodel/deepseek-ocr",
        output_dir: str = "./data/deepseek_markdown",
        device: str = "cuda",
        prompt: str = "请对该页做高保真OCR，输出结构化Markdown，保留表格。",
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
        self._processor = None
        self.max_retry = 2

    def _load_model(self) -> None:
        if self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"DeepSeek-OCR 模型路径不存在: {self.model_path}. "
                "请确认本地模型目录是否正确。"
            )

        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        self.logger.info("加载 DeepSeek-OCR 模型: %s", self.model_path)
        self._processor = AutoProcessor.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None,
        )
        if not self.device.startswith("cuda"):
            self._model.to(self.device)

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

    def _ocr_via_infer(self, image_path: Path, output_dir: Path) -> Optional[str]:
        """优先复用 DeepSeek 官方常见 infer 接口（若模型提供）。"""
        if not hasattr(self._model, "infer"):
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        infer_sig = inspect.signature(self._model.infer)
        kwargs = {}

        candidate_args = {
            "tokenizer": getattr(self._processor, "tokenizer", None),
            "prompt": self.prompt,
            "image_file": str(image_path),
            "output_path": str(output_dir),
            "base_size": 1024,
            "image_size": 640,
            "crop_mode": True,
            "save_results": True,
            "test_compress": True,
        }
        for key, value in candidate_args.items():
            if key in infer_sig.parameters and value is not None:
                kwargs[key] = value

        old_stdout = os.dup(1)
        old_stderr = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 1)
            os.dup2(devnull_fd, 2)
            self._model.infer(**kwargs)
        finally:
            os.close(devnull_fd)
            os.dup2(old_stdout, 1)
            os.dup2(old_stderr, 2)
            os.close(old_stdout)
            os.close(old_stderr)

        # 优先读取官方产物 result.mmd；若不存在，再读取 infer 返回值
        result_mmd = output_dir / "result.mmd"
        if result_mmd.exists():
            return result_mmd.read_text(encoding="utf-8")
        return None

    def _ocr_single_image(self, image, image_path: Optional[Path] = None, infer_output_dir: Optional[Path] = None) -> str:
        if self._model is None or self._processor is None:
            self._load_model()

        if image_path is not None and infer_output_dir is not None:
            infer_markdown = self._ocr_via_infer(image_path=image_path, output_dir=infer_output_dir)
            if infer_markdown is not None:
                return infer_markdown

        # 兼容不同模型暴露接口：ocr/chat/generate
        if hasattr(self._model, "ocr"):
            try:
                result = self._model.ocr(image=image, prompt=self.prompt)
            except TypeError:
                result = self._model.ocr(image, self.prompt)
            return str(result)

        if hasattr(self._model, "chat"):
            chat_sig = inspect.signature(self._model.chat)
            kwargs = {}
            if "image" in chat_sig.parameters:
                kwargs["image"] = image
            if "query" in chat_sig.parameters:
                kwargs["query"] = self.prompt
            elif "prompt" in chat_sig.parameters:
                kwargs["prompt"] = self.prompt

            if "processor" in chat_sig.parameters:
                kwargs["processor"] = self._processor
            if "tokenizer" in chat_sig.parameters and hasattr(self._processor, "tokenizer"):
                kwargs["tokenizer"] = self._processor.tokenizer

            if kwargs:
                result = self._model.chat(**kwargs)
            else:
                result = self._model.chat(image, self.prompt)
            return str(result)

        # 通用 Transformers 生成路径
        model_inputs = self._processor(images=image, text=self.prompt, return_tensors="pt")
        if hasattr(self._model, "device"):
            model_inputs = {k: v.to(self._model.device) for k, v in model_inputs.items()}

        output_ids = self._model.generate(**model_inputs, max_new_tokens=4096)
        decoded = self._processor.batch_decode(output_ids, skip_special_tokens=True)
        return decoded[0] if decoded else ""

    def _run_page_ocr_with_retry(self, image, image_path: Path, infer_output_dir: Path) -> Tuple[bool, str, Optional[str]]:
        """单页OCR，带重试。"""
        last_error: Optional[str] = None
        for attempt in range(self.max_retry + 1):
            try:
                markdown = self._ocr_single_image(
                    image=image,
                    image_path=image_path,
                    infer_output_dir=infer_output_dir,
                )
                if markdown.strip():
                    return True, markdown, None
                last_error = "空OCR结果"
            except Exception as exc:
                last_error = str(exc)
                self.logger.warning("第%s次OCR失败（%s）: %s", attempt + 1, image_path.name, exc)

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
            lines.append(f"\n<!-- Page {result.page_num} -->")
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
                    infer_output_dir = tmp_root / f"page_{idx:04d}"

                    success, markdown, error = self._run_page_ocr_with_retry(
                        image=image,
                        image_path=image_path,
                        infer_output_dir=infer_output_dir,
                    )
                    page_results.append(
                        _PageResult(page_num=idx, success=success, markdown=markdown, error=error)
                    )

                complete_markdown = self._build_markdown_with_page_markers(pdf_file.stem, page_results)
                markdown_path.write_text(complete_markdown, encoding="utf-8")
            finally:
                shutil.rmtree(tmp_root, ignore_errors=True)

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
        default="/storage/work/wuguowei/Bigmodel/deepseek-ocr",
        help="DeepSeek-OCR 本地模型目录",
    )
    parser.add_argument("--output-dir", default="./data/deepseek_markdown", help="默认输出目录")
    parser.add_argument("--device", default="cuda", help="推理设备，例如 cuda / cpu")
    parser.add_argument("--dpi", type=int, default=200, help="PDF渲染分辨率")
    parser.add_argument(
        "--prompt",
        default="请对该页做高保真OCR，输出结构化Markdown，保留表格。",
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
