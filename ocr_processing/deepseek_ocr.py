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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

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

    def _ocr_single_image(self, image) -> str:
        if self._model is None or self._processor is None:
            self._load_model()

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

    def process_pdf(self, pdf_path: str, output_markdown: Optional[str] = None) -> DeepSeekOCRResult:
        import time

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
            page_texts: List[str] = []
            for idx, image in enumerate(images, start=1):
                self.logger.info("OCR处理中: %s - 第 %s 页", pdf_file.name, idx)
                page_md = self._ocr_single_image(image)
                page_texts.append(f"\n\n## Page {idx}\n\n{page_md}")

            markdown_path.write_text("".join(page_texts), encoding="utf-8")

            elapsed = time.time() - start
            return DeepSeekOCRResult(
                pdf_path=pdf_file,
                markdown_path=markdown_path,
                status="success",
                pages_processed=len(images),
                processing_time=elapsed,
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
