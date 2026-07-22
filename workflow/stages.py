"""Small, explicit adapters composing v2 projects from existing ARneuro tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

from ..data_fetch.project_input import validate_pubmed_export
from ..feature_extraction.generic_study_extractor import GenericStudyInfoExtractor
from ..text_database.review_corpus import ReviewCorpusBuilder, ReviewCorpusConfig
from ..text_processing.final_segmentation import FinalSeg
from .project import ProjectContext


def validate_input(context: ProjectContext) -> Dict[str, Any]:
    context.initialize()
    report = validate_pubmed_export(context.input_table_path, context.stage_dir("input"))
    context.record_stage("validate_input", "completed", inputs=[context.input_table_path], outputs=[context.stage_dir("input") / "canonical_articles.csv"], details=report)
    return report


def segment_markdown_directory(context: ProjectContext, *, use_llm: bool = False) -> Dict[str, Any]:
    """Segment imported Markdown files using the stable final-segmentation adapter."""

    context.initialize()
    source_dir = context.stage_dir("markdown")
    target_dir = context.stage_dir("segmented")
    config = context.config.providers.get("segmentation", {}) if use_llm else {}
    segmenter = FinalSeg(config=config, preferred_client=str(config.get("provider", "deepseek")))
    processed = 0
    errors: Dict[str, str] = {}
    for markdown_path in sorted(source_dir.glob("*.md")):
        pmid = markdown_path.stem.replace("paper_", "")
        content_path = target_dir / f"paper_{pmid}_structured_content.json"
        meta_path = target_dir / f"paper_{pmid}_structured_meta.json"
        if content_path.exists() and meta_path.exists():
            continue
        try:
            result = segmenter.segment_markdown(markdown_path)
            content_path.write_text(json.dumps(result.structured, ensure_ascii=False, indent=2), encoding="utf-8")
            meta_path.write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
        except Exception as exc:
            errors[markdown_path.name] = f"{type(exc).__name__}: {exc}"
    report = {"processed": processed, "errors": errors, "source_dir": str(source_dir), "target_dir": str(target_dir)}
    context.record_stage("segment_documents", "completed" if not errors else "completed_with_errors", inputs=[source_dir], outputs=[target_dir], details=report)
    return report


def extract_generic_study_info(
    context: ProjectContext,
    completion: Callable[[str, str], Any],
    *,
    provider: str = "",
    model: str = "",
) -> Dict[str, Any]:
    """Extract one generic study record per segmented paper, resumably."""

    context.initialize()
    segmented_dir = context.stage_dir("segmented")
    output_dir = context.stage_dir("study_info")
    extractor = GenericStudyInfoExtractor(completion=completion, provider=provider, model=model)
    processed = 0
    errors: Dict[str, str] = {}
    for content_path in sorted(segmented_dir.glob("paper_*_structured_content.json")):
        pmid = content_path.name.split("_")[1]
        output_path = output_dir / f"paper_{pmid}_study_info.json"
        if output_path.exists():
            continue
        try:
            sections = json.loads(content_path.read_text(encoding="utf-8"))
            result = extractor.extract(pmid, sections)
            output_path.write_text(json.dumps({"data": result.data, "metadata": {"provider": result.provider, "model": result.model, "error": result.error, "prompt_characters": result.prompt_characters}}, ensure_ascii=False, indent=2), encoding="utf-8")
            processed += 1
        except Exception as exc:
            errors[pmid] = f"{type(exc).__name__}: {exc}"
    report = {"processed": processed, "errors": errors, "output_dir": str(output_dir)}
    context.record_stage("extract_study_info", "completed" if not errors else "completed_with_errors", inputs=[segmented_dir], outputs=[output_dir], details=report)
    return report


def build_database(context: ProjectContext, *, rebuild: bool = True) -> Dict[str, Any]:
    context.initialize()
    config = ReviewCorpusConfig(
        canonical_articles_csv=context.stage_dir("input") / "canonical_articles.csv",
        segmented_dir=context.stage_dir("segmented"),
        study_info_dir=context.stage_dir("study_info"),
        output_database=context.stage_dir("database") / "review_corpus.sqlite3",
        rebuild=rebuild,
    )
    report = ReviewCorpusBuilder(config).build()
    context.record_stage("build_database", "completed", inputs=[config.canonical_articles_csv, config.segmented_dir, config.study_info_dir], outputs=[config.output_database], details=report)
    return report
