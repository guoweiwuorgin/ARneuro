# Language Neuroscience Profile / 语言神经科学 Profile

The repository's established language-neuroscience workflow remains available as
an optional profile. It includes detailed language-task extraction, task name
review, feature annotation, embeddings, clustering, and task-specific tables.

These resources are not imported by the generic v2 command-line workflow. Use
them after generic segmentation and study extraction only when the review question
requires language-task taxonomy or language-network analysis.

Primary existing locations:

- `ARneuro/feature_extraction/task_info_extractor.py`
- `ARneuro/feature_extraction/language_task_feature_extractor.py`
- `ARneuro/text_database/structured_corpus.py`
- `ARneuro/examples/step5_*`, `step7_*`, and `step8_*`

中文：语言任务抽取、特征标记、embedding、聚类和人工校对仍然保留，但它们属于
可选领域扩展，不再是 ARneuro 通用综述流程的默认依赖。
