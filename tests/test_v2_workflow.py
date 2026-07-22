"""Offline smoke tests for the additive, domain-neutral v2 workflow."""

from __future__ import annotations

import json

import pandas as pd

from ARneuro.data_fetch.project_input import read_pubmed_export
from ARneuro.database_ui.app import create_app
from ARneuro.feature_extraction.generic_study_extractor import GenericStudyInfoExtractor
from ARneuro.text_database.review_corpus import ReviewCorpusBuilder, ReviewCorpusConfig


def test_excel_input_and_generic_corpus_build(tmp_path):
    source = tmp_path / "pubmed.xlsx"
    pd.DataFrame(
        [{"PMID": "900001", "Title": "Toy study", "Abstract": "Abstract"}]
    ).to_excel(source, index=False)
    records, report = read_pubmed_export(source)
    assert report["accepted_records"] == 1
    assert records[0].pmid == "900001"

    canonical = tmp_path / "canonical_articles.csv"
    pd.DataFrame([records[0].to_row()]).to_csv(canonical, index=False)
    segmented = tmp_path / "segmented"
    study_info = tmp_path / "study_info"
    segmented.mkdir()
    study_info.mkdir()
    content = {
        "Title": "Toy study",
        "Methods": "## Participants\nTen adults completed a button-press task in the scanner.",
        "Results": "The experimental condition increased response time.",
    }
    (segmented / "paper_900001_structured_content.json").write_text(json.dumps(content), encoding="utf-8")
    info = {
        "pmid": "900001",
        "study_type": "experimental",
        "research_modalities": ["fMRI"],
        "sample_groups": [{"name": "adults", "n": "10", "in_scanner": "yes"}],
        "experimental_tasks": [{"name": "Button press task", "category": "experimental_task", "description": "Respond to targets", "evidence": "button-press task"}],
    }
    (study_info / "paper_900001_study_info.json").write_text(json.dumps(info), encoding="utf-8")
    database = tmp_path / "review.sqlite3"
    report = ReviewCorpusBuilder(
        ReviewCorpusConfig(canonical_articles_csv=canonical, segmented_dir=segmented, study_info_dir=study_info, output_database=database)
    ).build()
    assert report["documents"] == 1
    assert report["experimental_tasks"] == 1
    assert database.exists()
    client = create_app(database, tmp_path / "workspace.sqlite3").test_client()
    assert client.get("/api/stats").status_code == 200
    detail = client.get("/api/document/900001")
    assert detail.status_code == 200
    assert detail.get_json()["tasks"][0]["final_task_name"] == "Button press task"


def test_generic_extractor_uses_results_when_methods_is_short():
    seen = {}

    def completion(_system, prompt):
        seen["prompt"] = prompt
        return {"study_type": "experimental", "experimental_tasks": []}

    result = GenericStudyInfoExtractor(completion=completion).extract(
        "1", {"Methods": "Brief methods.", "Results": "Participants completed a task."}
    )
    assert result.data["study_type"] == "experimental"
    assert "## Results" in seen["prompt"]
