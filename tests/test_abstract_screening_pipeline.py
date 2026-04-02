from dataclasses import dataclass
from pathlib import Path

from feature_extraction.abstract_screening import AbstractScreeningHarness


class FakeAbstractProvider:
    def fetch_abstract(self, pmid: str):
        if pmid == "1002":
            return "This fMRI study tests language networks in healthy adults."
        return None


class FakeLLMJudge:
    def evaluate_abstract(self, criteria, abstract: str, pmid: str):
        include = "fmri" in abstract.lower() and "language" in abstract.lower()
        return {
            "include": include,
            "confidence": 0.92 if include else 0.25,
            "reason": "method+topic fit" if include else "not in scope",
            "labels": {
                "method": "fmri" if "fmri" in abstract.lower() else "unknown",
                "topic": "language" if "language" in abstract.lower() else "other",
            },
        }


@dataclass
class FakeDownloadResult:
    file_path: Path
    status: str = "success"


class FakePDFDownloader:
    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path

    def download(self, pmid: str):
        pdf_path = self.tmp_path / f"{pmid}.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n%fake\n")
        return FakeDownloadResult(file_path=pdf_path)


@dataclass
class FakeOCRResult:
    status: str


class FakeDeepSeekOCR:
    def process_pdf(self, pdf_path: str):
        return FakeOCRResult(status="success")


def test_abstract_screening_harness_end_to_end(tmp_path: Path):
    csv_path = tmp_path / "input.csv"
    csv_path.write_text(
        "PMID,Abstract\n"
        "1001,This is a behavior-only attention study.\n"
        "1002,\n",
        encoding="utf-8",
    )

    criteria_yaml = tmp_path / "criteria.yaml"
    criteria_yaml.write_text(
        """
review_name: Demo Review
research_field: [psychology, neuroscience]
research_topic_statement: Test language neural mechanism studies
methodology:
  include: [fmri, eeg, behavior]
  exclude: [animal_only]
participant_scope:
  species: [human]
  age_range: {min: 18, max: 65}
disease_and_case_report_policy:
  include_disease_studies: true
  allow_case_report: false
publication_filters:
  year_from: 2010
  languages: [English]
decision_policy:
  min_confidence_to_include: 0.70
  uncertain_bucket: manual_review
""",
        encoding="utf-8",
    )

    peft_path = tmp_path / "peft_dataset.jsonl"
    harness = AbstractScreeningHarness(
        abstract_provider=FakeAbstractProvider(),
        llm_judge=FakeLLMJudge(),
        pdf_downloader=FakePDFDownloader(tmp_path=tmp_path),
        deepseek_ocr=FakeDeepSeekOCR(),
    )

    result = harness.run(
        csv_path=csv_path,
        criteria_yaml=criteria_yaml,
        peft_export_path=peft_path,
    )

    assert result.total_records == 2
    assert result.abstract_available == 2
    assert result.included_count == 1
    assert result.excluded_count == 1
    assert result.downloaded_count == 1
    assert result.ocr_success_count == 1
    assert len(result.peft_records) == 2
    assert peft_path.exists()
    assert "Qwen3-3B" in peft_path.read_text(encoding="utf-8")
