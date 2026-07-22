# ARneuro Corpus Annotation Workspace

Launch the local workspace from the `reviewer` environment:

```powershell
conda run --no-capture-output -n reviewer python -B ARneuro/examples/step8_9_run_corpus_annotation_ui.py
```

Open `http://127.0.0.1:8765` in a browser.

The interface reads the corpus database at
`structured_text_database/language_neuro_corpus.sqlite3` without changing it.
Manual annotations are saved in the rebuild-safe sidecar database:

`structured_text_database/annotation_workspace/annotation_workspace.sqlite3`

The workspace supports document filtering, segmented Methods/Results correction,
task review, activation-table review, task-to-table links, coordinate annotation,
annotation history, and export of documents marked `gold` as three JSONL datasets.

Tasks are shown and analyzed at the document level: one `PMID + normalized task
name` record merges patient and typical-human source records. The original rows
remain in `task_occurrences`; the default analysis table is `document_tasks`.
