# ARneuro Structured Text Database

`StructuredCorpusBuilder` creates one portable SQLite/FTS5 database from
segmented paper JSON files and the method-information table. It preserves PMID,
section names, Markdown heading paths, character offsets, source JSON paths,
method metadata, and available task annotations.

```python
from pathlib import Path
from ARneuro.text_database import StructuredCorpus

corpus = StructuredCorpus(Path("structured_text_database/language_neuro_corpus.sqlite3"))

# Ranked, section-aware full-text retrieval.
hits = corpus.search_phrase("language localization", section_name="Methods")

# Direct document and section access.
method_text = corpus.get_section("15703256", "Methods")

# Recover the source context of a task clue sentence.
source = corpus.locate_clue_sentence("15703256", "During scanning, subjects...")

# Corpus statistics with supporting PMIDs.
tfidf = corpus.term_tfidf("language localization", unit="document", section_name="Methods")
term_ppmi = corpus.ppmi_terms("IFG", "picture naming", unit="document")
task_ppmi = corpus.ppmi_task_term("Picture Naming", "IFG", section_name="Methods")
```

Task names are indexed in ``task_occurrences.normalized_task_name`` using
lowercase/casefolded whitespace-normalized keys. The original reviewed display
name remains in ``final_task_name``. ``find_task_occurrences`` and
``ppmi_task_term`` normalize their task-name input to this same key.

For task-specific source checking before adding a new annotation, use
``TaskContextMatcher`` and the 20-article review entry point
``ARneuro/examples/step8_4_review_task_inscanner_context_test.py``. It uses one
review unit per ``PMID + normalized_task_name``, locates a task by its original
name or clue sentence, and presents the matched source span with 500 characters
of surrounding text. Paper-level fMRI participation fields are shown only as
context and are never copied into a task-level decision.

The build entry point is
`ARneuro/examples/step8_build_structured_text_database.py`. Query and LoRA
export examples are in the other `step8_*.py` scripts.

The exported `passage` JSONL is source-traceable continued-pretraining/RAG
corpus data. The `method_info_sft` JSONL pairs Methods text with the existing
method-info JSON, rather than inventing new supervision labels.
