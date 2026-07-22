---
name: arneuro-literature-review
description: Build, resume, inspect, and curate reproducible neuroscience literature-review projects with ARneuro. Use when a user wants to start from a PubMed Excel/CSV/TSV export, import PDFs or Markdown, run OCR/segmentation/study extraction/table or citation analysis, build/search the SQLite corpus, launch the annotation UI, export human-reviewed training data, or apply the optional language-neuroscience profile.
---

# ARneuro Literature Review

Use ARneuro v2 as a project workflow, not as unrelated scripts. Read
`references/workflow.md` before selecting stages or creating a project.

## Start with the project state

1. Locate `arneuro_project.yaml`, the PubMed export, existing Markdown/PDFs, and
   `output/run_manifest.json`.
2. Inspect the manifest and existing artifacts before rerunning a stage. Preserve
   raw PDFs, OCR Markdown, segmented JSON, and human corrections.
3. For a new project, run `scripts/create_project.py` to create the layout and
   config. Do not put API keys in YAML; use environment variables.
4. Ask one concise question only when the input source, study scope, or intended
   model provider is genuinely unavailable. Otherwise proceed with sensible
   project-relative defaults.

## Choose the smallest applicable workflow

| User goal | Use |
| --- | --- |
| Validate a PubMed export | `python -m ARneuro validate-input --project <yaml>` |
| Segment existing Markdown | `python -m ARneuro segment --project <yaml>` |
| Build/rebuild a corpus | `python -m ARneuro build-database --project <yaml>` |
| Check progress/resume point | `python -m ARneuro show-plan --project <yaml>` |
| Review Methods, Results, tasks, tables, or coordinates | Launch the local annotation UI using the project SQLite database. |
| Produce fine-tuning data | Export only documents marked `gold`; retain source and correction provenance. |
| Analyse language task taxonomies | Read `references/language-profile.md` and use the optional profile only when requested. |

Use OCR, LLM extraction, citation enrichment, and embeddings only for stages the
user explicitly needs. Keep those dependencies optional and load them lazily.

## Execute safely

- Use the `reviewer` conda environment when it exists.
- Treat the manifest as the run ledger. Mark a stage completed only after output
  artifacts are present and validated.
- Prefer deterministic segmentation first. Enable the LLM fallback only for
  genuinely incomplete or ambiguous documents.
- Use one structured extraction JSON per PMID; record provider, model, prompt
  version, input source, and failure state beside it.
- Keep human edits in the annotation workspace. Never overwrite original section
  text or model output without preserving provenance.
- Obtain or import PDFs only through lawful access routes. Do not scrape or
  redistribute copyrighted full texts.

## Generic schema first

Use the generic study schema for study type, sample groups, experimental
paradigms, acquisition, analysis, outcomes, evidence, and uncertainty. Do not
assume language, fMRI, patient/control groups, or a particular scanner. Add a
domain profile only after the generic corpus is stable.

## Verify before handing off

Run focused tests or a small offline fixture whenever code changes. For a full
no-network reference workflow, use `ARneuro/Toy_example`; do not run external
download, OCR, or model stages unless the user asked for execution.
