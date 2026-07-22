# ARneuro v2 workflow reference

## Project files

- `arneuro_project.yaml`: project-relative input, output, provider and scope
  settings. Resolve `${ENV_VAR}` values only at runtime.
- `output/run_manifest.json`: stage ledger. Inspect it before resuming work.
- `output/00_input/canonical_articles.csv`: validated source records.
- `output/04_segmented/paper_<PMID>_structured_content.json`: preserved source
  for sections, tables, passages and later human review.
- `output/05_study_info/paper_<PMID>_study_info.json`: one generic extraction
  record plus provider/error metadata.
- `output/08_database/review_corpus.sqlite3`: portable corpus and FTS index.
- annotation workspace SQLite: separate human corrections and training status.

## Stage selection

1. Start with `validate-input`; accept `.xlsx`, `.xls`, `.csv`, or `.tsv`.
2. Use screening only when there is a scope decision to make.
3. Import local lawful PDFs or use an OA-aware acquisition adapter.
4. Convert PDFs to Markdown with a project-selected OCR backend. Keep OCR raw
   output separately from segmented JSON.
5. Segment Markdown using `FinalSeg`. Use deterministic logic by default and LLM
   recovery only if Methods or Results are incomplete.
6. Extract generic study metadata from Methods; supplement short Methods with
   Results, Introduction, or Other only when necessary.
7. Extract activation tables/coordinates and parse references when the review
   question needs them.
8. Build SQLite after the relevant source artifacts are stable.
9. Use the UI to create human-verified Methods/Results, task/paradigm and table
   annotations. Export only `gold` records for model fine-tuning.

## Generic extraction contract

Required top-level keys are `pmid`, `study_type`, `research_modalities`,
`study_design`, `sample_groups`, `experimental_tasks`, `acquisition`, `analysis`,
`key_outcomes`, `missing_or_uncertain_fields`, and `evidence`.

For each sample group preserve role, `n`, demographic/clinical context,
in-scanner status, and evidence. For each task/paradigm preserve a descriptive
name, category, description, modality, response, in-scanner status, contrast,
and evidence. Empty values are better than unsupported inference.

## Operational rules

- Process resumably: skip an artifact only if it exists and is valid.
- Log errors by PMID and preserve failed inputs for later repair.
- Use environment variables for keys and avoid printing them.
- Do not replace previous generated artifacts in place without a versioned output
  directory or an explicit user instruction.
- Before a broad paid-model run, test 10 to 20 representative papers and retain
  the test outputs for review.
