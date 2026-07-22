# Migration from v1 to v2 / 从 v1 迁移到 v2

## No destructive migration

v2 is additive. Existing `examples/`, `feature_extraction/` language-task tools,
current segmented corpora, method-information CSV files, task review tables, and
annotation workspaces are not altered by `workflow/` or `Toy_example/`.

## Recommended migration

1. Create a new project directory and copy `Toy_example/arneuro_project.yaml`.
2. Point `input.pubmed_table` to a PubMed Excel/CSV export.
3. Run `python -m ARneuro validate-input --project ...`.
4. Import lawful PDFs or retain an existing Markdown corpus in `03_markdown/`.
5. Use the existing final segmenter through `python -m ARneuro segment`.
6. Run generic study extraction with a provider adapter that records model and
   prompt versions; save one JSON per PMID in `05_study_info/`.
7. Build `08_database/review_corpus.sqlite3` and open the annotation UI.
8. Export only human-reviewed gold documents to JSONL for fine-tuning.

## Legacy language corpus

The current language corpus can remain operational with
`text_database/structured_corpus.py`. Do not force it into v2 unless you need a
fresh generic corpus: rebuilding from its segmented JSON and canonical PubMed
records preserves original source content while allowing new generic metadata.

## Credentials

Before sharing or publishing a project, rotate historical provider keys and remove
them from scripts, notebooks, logs, exported command histories, and YAML files.

---

## 中文摘要

v2 是增量升级，不会改写旧的语言任务数据库或已有结果。建议对新项目使用新的
`arneuro_project.yaml` 和项目目录；旧语料如需迁移，只需保留原始分割 JSON 与
PubMed 记录，然后重新构建通用 SQLite。公开或共享项目之前，请轮换历史密钥并
清理脚本、日志、命令历史和配置中的密钥。
