import csv
import json
from pathlib import Path

from ARneuro.text_database import CorpusBuildConfig, StructuredCorpus, StructuredCorpusBuilder


def _write_csv(path: Path, rows, delimiter=","):
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]), delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def test_build_query_and_statistics(tmp_path: Path):
    segmented = tmp_path / "segmented"
    segmented.mkdir()
    method_csv = tmp_path / "method_info.csv"
    task_csv = tmp_path / "tasks.csv"
    content = {
        "Title": "Example language study",
        "Methods": "### Participants\nTwenty healthy adults completed picture naming in the IFG localizer task.",
        "Results": "The IFG response was higher during picture naming.",
    }
    (segmented / "paper_123_structured_content.json").write_text(json.dumps(content), encoding="utf-8")
    (segmented / "paper_123_structured_meta.json").write_text(json.dumps({"source_file": "paper_123.md"}), encoding="utf-8")
    _write_csv(
        method_csv,
        [{"PMID": "123", "typical_human_total": "10, 10", "patient_total": "", "scanner_type": "3T"}],
    )
    _write_csv(
        task_csv,
        [
            {
                "PMID": "123",
                "task_index": "1",
                "task_id": "123_1",
                "source_group": "typical_human",
                "hier_cluster_k258": "4",
                "structured_task_name": "Picture Naming Task",
                "description": "Name pictured objects.",
                "clue_sentences": "completed picture naming",
            }
        ],
        delimiter=";",
    )
    result = StructuredCorpusBuilder(
        CorpusBuildConfig(
            method_info_csv=method_csv,
            segmented_dir=segmented,
            output_dir=tmp_path / "database",
            task_occurrences_csv=task_csv,
        )
    ).build()

    assert result.eligible_pmids == 1
    assert result.indexed_documents == 1
    corpus = StructuredCorpus(result.database_path)
    assert corpus.get_section("123", "Methods")["content"].startswith("### Participants")
    assert corpus.search_phrase("picture naming", section_name="Methods")[0]["pmid"] == "123"
    assert corpus.find_task_occurrences("Picture Naming Task", exact=True)[0]["pmid"] == "123"
    assert corpus.locate_clue_sentence("123", "Twenty healthy adults completed picture naming.")["pmid"] == "123"
    assert corpus.term_tfidf("IFG", unit="document")["document_frequency"] == 1
    assert corpus.ppmi_terms("IFG", "picture naming")["cooccurrence_unit_count"] == 1
    assert corpus.ppmi_task_term("Picture Naming", "IFG")["cooccurrence_document_count"] == 1
