# ARneuro v2 Architecture Audit / ARneuro v2 架构审阅

## Scope

The repository contains a mature, valuable language-neuroscience production
workflow alongside early general-purpose components. This audit introduces a
general review core without deleting, moving, or invalidating the existing
language-specialized outputs.

## Findings

| Area | Current strength | v2 issue addressed |
| --- | --- | --- |
| Segmentation | Strong rule + LLM recovery in `text_processing` | Old examples do not present it as a reusable project stage. |
| Study extraction | Rich language/fMRI schema | Core schema assumes language-specific variables and group labels. |
| Database/UI | SQLite, FTS, provenance, annotation UI | The default data model and labels are language-task centred. |
| OCR | Multiple practical backends | Heavy dependencies should be optional and lazily invoked. |
| Citation network | Reference parser, external matching, DeepSeek Bib fallback | It needs an explicit generic pipeline boundary and key hygiene. |
| Entry points | Many useful examples | Paths, provider settings, and some credentials are embedded in example-era scripts. |

## v2 decisions

1. `workflow/` owns project YAML, stage directories, and resumable manifests.
2. `data_fetch/project_input.py` accepts PubMed Excel, CSV, and TSV exports.
3. `generic_study_extractor.py` is the first-pass schema. Profiles enrich it;
   they never redefine the generic storage contract.
4. `review_corpus.py` creates one portable SQLite database with sections,
   passages, FTS, source records, generic study metadata, and paradigms.
5. The existing UI remains the manual-quality-control surface. Its storage model
   keeps source content and corrections separate for training-data provenance.
6. OCR, embeddings, provider SDKs and specific model backends remain optional.
7. New code receives secrets only through the environment. Legacy scripts should
   be migrated gradually, never used as secret templates.

## Generic stage contracts

```text
PubMed export -> canonical_articles.csv
  -> screening decisions -> local/OA PDF files
  -> Markdown -> structured content + metadata
  -> generic study JSON -> tables / coordinates / references
  -> SQLite corpus -> human corrections -> JSONL training sets
```

Each stage must be idempotent where possible, record artifacts in
`run_manifest.json`, and preserve the prior stage's raw output.

## Deliberately optional profiles

Language-task naming, linguistic feature vectors, task embeddings, language
network analyses, cluster relabelling, and hand-curated language task dictionaries
are highly useful but not universal neuroscience requirements. They remain
available under the language-neuroscience profile / legacy example surface.

## Validation expectation

The Toy project is the required no-network smoke workflow. A production project
adds tests for its own provider call, legal acquisition policy, OCR backend, and
human-review sampling protocol.

---

## 中文摘要

v2 不删除已有语言神经科学工作，而是在其上增加通用项目层。通用核心采用
“输入表格 -> 分割文献 -> 通用研究信息 -> SQLite -> 人工订正 -> 训练 JSONL”
的稳定契约；语言任务命名、特征、嵌入和聚类等保留为可选 profile。OCR、嵌入、
外部模型和参考文献补全均为按需安装、按需运行的可选能力。新的代码只能从环境
变量读取密钥，并以 `run_manifest.json` 保留全流程可追溯性。
