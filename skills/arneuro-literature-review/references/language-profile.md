# Optional language-neuroscience profile

Use this profile only for research questions requiring language-task taxonomy,
language-specific task naming, linguistic feature embeddings, language network
term analyses, or task clusters. It is not the generic ARneuro default.

Relevant existing modules:

- `ARneuro/feature_extraction/task_info_extractor.py`
- `ARneuro/feature_extraction/language_task_feature_extractor.py`
- `ARneuro/text_database/structured_corpus.py`
- `ARneuro/examples/step5_*`, `step7_*`, `step8_*`

First build or inspect the generic corpus, then create a separately named,
versioned language-profile output directory. Keep generic task/paradigm fields
and the profile-specific labels side by side so an original PMID and source
sentence remain traceable.
