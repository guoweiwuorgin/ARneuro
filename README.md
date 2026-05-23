# ARneuro：基于 LLM 的神经影像文献定量综述工具

ARneuro 提供了从 **PMID 列表 → 文献筛选 → PDF 下载 → OCR 转 Markdown → 文档结构化 → 脑激活表提取** 的完整流程。

> 当前主线 OCR 已切换为 **DeepSeek OCR**，不再以 GLM OCR 作为主要方案。

---

## 核心流程（推荐）

1. **步骤1：摘要筛选 + PDF 下载**  
   `examples/step1_screen_abstract_and_download.py`
2. **步骤2：DeepSeek OCR 批量转换 PDF → Markdown**  
   `examples/step2_deepseek_ocr_from_downloaded_pdfs.py`
3. **步骤3：Markdown 文档分割并构建文档库**  
   `examples/step3_segment_markdown_and_build_library.py`
4. **步骤4：提取脑激活表格并导出结构化结果**  
   `examples/step4_extract_activation_tables_from_markdown.py`

---

## 安装

### 1) 环境要求
- Python >= 3.10
- 建议 CUDA 环境（可选，提升 DeepSeek OCR 速度）

### 2) 安装依赖

```bash
pip install -r requirements.txt
```

### 3) 准备 DeepSeek OCR 模型
请确保本地可用 DeepSeek OCR 模型目录，例如：

```text
/storage/work/wuguowei/Bigmodel/DeepSeek-OCR-2
```

---

## 输入数据

准备一个包含 PMID 的 CSV 文件（至少需要 `PMID` 列）：

```csv
PMID,Title
12345678,Sample Article A
23456789,Sample Article B
```

仓库示例输入：
- `examples/data/ARneuro_test.csv`

---

## 按模块运行（与 examples 保持一致）

## 步骤1：摘要筛选并下载 PDF

脚本：`examples/step1_screen_abstract_and_download.py`

```bash
python examples/step1_screen_abstract_and_download.py \
  --csv-file ./examples/data/ARneuro_test.csv \
  --criteria-yaml ./config/review_screening_criteria_template.yaml \
  --output-dir ./examples/data/workflow/step1 \
  --llm-client-type deepseek \
  --llm-model-name deepseek-chat
```

### 步骤1输出
- `screening_results.csv`
- `screening_results.json`
- `included_pmids.txt`
- `checkpoint.json`（断点续跑）

---

## 步骤2：DeepSeek OCR（主方案）

脚本：`examples/step2_deepseek_ocr_from_downloaded_pdfs.py`

```bash
python examples/step2_deepseek_ocr_from_downloaded_pdfs.py \
  --step1-results ./examples/data/workflow/step1/screening_results.csv \
  --pdf-dir ./examples/data/pdfs \
  --output-dir ./examples/data/workflow/step2_markdown \
  --model-path /storage/work/wuguowei/Bigmodel/DeepSeek-OCR-2 \
  --device cuda \
  --dpi 200 \
  --prompt "<image>\n<|grounding|>Convert the document to markdown."
```

### 步骤2输出
- 每篇文献对应 `.md`
- `ocr_report.json`
- `ocr_report.csv`

---

## 步骤3：分割 Markdown 并构建文档库

脚本：`examples/step3_segment_markdown_and_build_library.py`

```bash
python examples/step3_segment_markdown_and_build_library.py \
  --markdown-dir ./examples/data/workflow/step2_markdown \
  --output-dir ./examples/data/workflow/step3_library
```

### 步骤3输出
- `document_library_index.json`
- `segmentation_part_stats.json`
- `segmented/*_structured_content.json`
- `segmented/*_structured_meta.json`

---

## 步骤4：提取脑激活表格

脚本：`examples/step4_extract_activation_tables_from_markdown.py`

```bash
python examples/step4_extract_activation_tables_from_markdown.py \
  --step3-library-dir ./examples/data/workflow/step3_library \
  --output-dir ./examples/data/workflow/step4_activation_tables
```

### 步骤4输出
- 每篇文献独立目录（JSON + CSV）
- `activation_tables_summary.json`

---

## 配置建议（以 DeepSeek OCR 为主）

可在项目配置中使用类似字段（示意）：

```yaml
ocr_processing:
  backend: "local"
  primary_engine: "deepseek_ocr"
  model_path: "/storage/work/wuguowei/Bigmodel/DeepSeek-OCR-2"
  device: "cuda"
  dpi: 200
  prompt: "<image>\n<|grounding|>Convert the document to markdown."
  output_dir: "./examples/data/workflow/step2_markdown"
```

---

## 与旧版 GLM OCR 的关系

- GLM OCR 相关代码与调试脚本可保留用于兼容或对照实验。
- 但在当前 README 的推荐流程中，**默认与主推均为 DeepSeek OCR**。

---

## 目录建议

- `examples/step1_screen_abstract_and_download.py`：筛选与下载
- `examples/step2_deepseek_ocr_from_downloaded_pdfs.py`：OCR 主流程
- `examples/step3_segment_markdown_and_build_library.py`：结构化分割
- `examples/step4_extract_activation_tables_from_markdown.py`：表格提取
- `examples/data/workflow/`：完整流程输出样例

---

## 常见问题

1. **步骤2找不到 PDF**  
   优先检查 `--pdf-dir` 是否存在 PDF；否则检查 `--step1-results` 里的 `PDFPath` 列。

2. **GPU 不可用**  
   可切换 `--device cpu`，速度会明显下降。

3. **OCR 结果质量不稳定**  
   建议优先调整 `--dpi`（200/300）与 `--prompt`，再进行对比测试。
