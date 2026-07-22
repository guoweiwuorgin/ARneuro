# ARneuro v2 
![ARneuro_log.png]

**English** | [中文](#中文说明)

ARneuro is a project-oriented, reproducible workflow for neuroscience literature
reviews. It begins with a PubMed Excel/CSV export and ends with a compact SQLite
corpus plus a local annotation interface. The default v2 workflow is deliberately
domain-neutral: it does not assume a language task, an fMRI study, or a
patient-versus-control design.

Existing language-neuroscience scripts and results remain in the repository. They
are supported as an optional profile, not as assumptions baked into the generic
workflow.

## What v2 produces

1. Validated, de-duplicated PubMed records with provenance.
2. Screening, PDF acquisition/import, OCR, and structured Markdown segmentation.
3. Generic study JSON: design, sample groups, paradigms, acquisition, analysis,
   outcomes, evidence, and explicitly missing fields.
4. Activation-table / coordinate and citation-network stages when relevant.
5. A portable SQLite corpus: full sections, heading-aware passages, FTS search,
   structured study metadata, experimental tasks, and source provenance.
6. A local human-review UI and JSONL exports for supervised fine-tuning.

## Project layout

```text
your_project/
  arneuro_project.yaml
  input/pubmed_export.xlsx
  output/
    00_input/             canonical records and validation report
    01_screening/         inclusion decisions and evidence
    02_pdfs/              lawfully obtained or imported PDFs
    03_markdown/          OCR Markdown
    04_segmented/         structured content and segmentation metadata
    05_study_info/        generic per-paper JSON
    06_tables/            table and coordinate extraction
    07_citations/         citation-network artifacts
    08_database/          SQLite corpus and annotation workspace
    09_exports/           training JSONL exports
    logs/run_manifest.json
```

The manifest records input/output artifacts and stage status, making incomplete
runs inspectable and resumable.

## Installation

Use Python 3.10+ and install the base dependencies required by the light-weight
project and database workflow:

```powershell
conda activate reviewer
pip install pandas openpyxl pyyaml flask requests beautifulsoup4 openai
```

OCR backends, local models, embeddings, and specific provider SDKs are optional.
Install them only for the stages you intend to run. API secrets must be provided as
environment variables, never committed into project YAML or examples.

## Minimal commands

```powershell
python -m ARneuro init --project .\arneuro_project.yaml
python -m ARneuro validate-input --project .\arneuro_project.yaml
python -m ARneuro segment --project .\arneuro_project.yaml
python -m ARneuro build-database --project .\arneuro_project.yaml
python -m ARneuro show-plan --project .\arneuro_project.yaml
```

`segment` uses deterministic segmentation by default. `--use-llm` only enables
the optional difficult-document fallback when a provider has been configured.

## Complete offline demonstration

[`Toy_example`](Toy_example/README.md) is a two-paper, no-network, no-model
demonstration. It starts by creating a PubMed-style `.xlsx`, installs two Markdown
fixtures as OCR output, performs deterministic segmentation, copies reviewed
generic study JSON, builds SQLite, and can launch the UI.

```powershell
cd ARneuro\Toy_example
python run_all.py                # show the plan only
python run_all.py --execute      # run local fixtures only
python 10_launch_annotation_ui.py
```

## AI assistant skill

The portable Codex skill is in
[`skills/arneuro-literature-review`](skills/arneuro-literature-review). It guides
an AI assistant through project setup, resumable stage selection, generic study
extraction, corpus/UI work, and the optional language-neuroscience profile.

To make it discoverable in a Codex installation, copy or link that directory into
your Codex skills directory, then invoke it as `$arneuro-literature-review`.

## Architecture and migration

- [v2 architecture audit](docs/ARCHITECTURE_AUDIT_V2.md)
- [v1-to-v2 migration notes](docs/MIGRATION_V1_TO_V2.md)
- [language-neuroscience optional profile](profiles/language_neuroscience/README.md)

## Safety and reproducibility

- Use only legally accessible or locally imported PDFs.
- Keep provider keys in environment variables such as `ARNEURO_LLM_API_KEY`.
- Preserve raw Markdown and structured content; human edits are stored as separate
  annotation records, never silently overwrite source text.
- Record model, prompt version, input hashes, and extraction failures for every
  model-enabled production stage.

---

## 中文说明

ARneuro 是一个面向神经科学文献综述的、可复现的项目化流程。它从 PubMed
导出的 Excel/CSV 表格开始，最终构建轻量级 SQLite 文本数据库和本地人工校对
界面。v2 默认流程是**领域通用的**：不会预设研究必须是语言任务、fMRI，或
患者-对照设计。

仓库中已有的语言神经科学脚本、结果和分析不会删除；它们被保留为可选的
领域 profile，而不再成为通用流程的隐含前提。

### v2 的主要输出

1. 可追溯、去重后的 PubMed 文献记录。
2. 筛选、PDF 获取/导入、OCR 与 Markdown 结构化分割。
3. 通用研究信息 JSON：研究设计、样本组、实验范式、采集、分析、结局、证据
   和缺失项。
4. 按项目需要执行的激活表/坐标提取和参考文献网络构建。
5. 可迁移的 SQLite 数据库：全文结构、带标题路径的段落、FTS 检索、研究信息、
   实验任务和原始来源。
6. 本地人工校对 UI，以及用于 LoRA / 监督微调的 JSONL 导出。

### 项目化使用

在每个新项目根目录放置 `arneuro_project.yaml`，再运行：

```powershell
python -m ARneuro init --project .\arneuro_project.yaml
python -m ARneuro validate-input --project .\arneuro_project.yaml
python -m ARneuro segment --project .\arneuro_project.yaml
python -m ARneuro build-database --project .\arneuro_project.yaml
```

`output/logs/run_manifest.json` 会记录每一步的输入、输出和状态，便于中断后
继续执行。默认分割不调用模型；只有在配置好服务商并显式加入 `--use-llm` 时，
才会使用困难文档的 LLM 补充分割。

### 离线完整示例

[`Toy_example`](Toy_example/README.md) 提供两个虚拟文献的完整、离线、无模型
示例。它会创建 PubMed 格式 Excel，放入 Markdown fixture，完成规则分割、
导入已人工核对的研究信息、构建 SQLite，并可启动人工标注界面。

```powershell
cd ARneuro\Toy_example
python run_all.py
python run_all.py --execute
python 10_launch_annotation_ui.py
```

### AI 助手 Skill

可移植的 Codex Skill 位于
[`skills/arneuro-literature-review`](skills/arneuro-literature-review)。它能让 AI
助手按照项目状态快速选择建库、恢复运行、通用研究信息抽取、数据库/UI 人工校对，
以及可选的语言神经科学 profile。将该目录复制或链接到 Codex 的 skills 目录后，
即可通过 `$arneuro-literature-review` 调用。

### 使用原则

- 仅处理合法公开获取或用户本地导入的 PDF。
- API key 必须通过环境变量提供，不能写入 YAML、示例脚本或版本控制文件。
- 原始 Markdown 与分割内容应永久保留；人工订正以独立 annotation 记录保存，
  不静默覆盖原始文献。
- 正式模型批处理应记录模型版本、prompt 版本、输入哈希及错误日志。
