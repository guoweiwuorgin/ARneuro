# ARneuro Toy Example / ARneuro 离线完整示例

**English** | [中文](#中文说明)

This directory is a complete two-paper fixture project. It is intentionally
offline: it neither downloads papers nor calls an LLM. It demonstrates the same
artifact contracts used by a real project, including a PubMed-style Excel input,
Markdown, structured segmentation, generic study information, SQLite, and the
local annotation UI.

## Run

```powershell
conda activate reviewer
cd D:\language_template\reviewer\current_data\step3_library\ARneuro\Toy_example
python run_all.py
python run_all.py --execute
python 10_launch_annotation_ui.py
```

`run_all.py` without `--execute` is only a plan display. The executed fixture
workflow creates files under `output/`; it makes no network or model call.

## Steps

| Script | Demonstrated contract |
| --- | --- |
| `00_create_input_excel.py` | Create a PubMed-style Excel input fixture. |
| `01_validate_pubmed_excel.py` | Validate aliases, PMIDs, duplicates, and canonicalize records. |
| `02_screen_records.py` | Screening checkpoint and decision artifact. |
| `03_acquire_or_import_pdfs.py` | PDF acquisition/import boundary. |
| `04_ocr_to_markdown.py` | OCR output boundary, supplied by Markdown fixtures. |
| `05_segment_and_validate.py` | Deterministic structured segmentation. |
| `06_extract_generic_study_info.py` | Generic study-extraction JSON contract. |
| `07_extract_tables_and_coordinates.py` | Table / coordinate extraction boundary. |
| `08_build_citation_network.py` | Citation-network stage contract. |
| `09_build_sqlite_database.py` | Portable corpus build. |
| `10_launch_annotation_ui.py` | Local document / task / table / section review UI. |
| `11_export_training_data.py` | Gold-standard JSONL export after annotation. |

For a real project, replace the fixtures in steps 02-06 with lawful PDF import,
OCR and provider-backed extraction. Do not put API keys in `arneuro_project.yaml`.

---

## 中文说明

该目录是一个完整的两篇虚拟文献项目。它完全离线，不会下载论文，也不会调用
大模型，但会演示真实项目完全一致的输入、输出和断点文件约定：PubMed 格式
Excel、Markdown、结构化分割、通用研究信息、SQLite 和本地人工校对界面。

```powershell
conda activate reviewer
cd D:\language_template\reviewer\current_data\step3_library\ARneuro\Toy_example
python run_all.py
python run_all.py --execute
python 10_launch_annotation_ui.py
```

不带 `--execute` 时只展示执行计划。带 `--execute` 时只在 `output/` 下生成
本地 fixture 结果，仍不会访问网络或调用模型。真实项目请将 02-06 步的 fixture
替换为合法 PDF 导入、OCR 与模型抽取，并且始终通过环境变量传递 API key。
