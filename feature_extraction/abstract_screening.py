"""Abstract screening test harness for review scope filtering."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import yaml


@dataclass
class ScreeningCriteria:
    """User-defined review scope criteria loaded from YAML."""

    review_name: str
    research_field: List[str]
    research_topic_statement: str
    methodology_include: List[str]
    methodology_exclude: List[str]
    participant_scope: Dict[str, Any]
    disease_and_case_report_policy: Dict[str, Any]
    publication_filters: Dict[str, Any]
    priority_signals: List[str] = field(default_factory=list)
    exclusion_signals: List[str] = field(default_factory=list)
    decision_policy: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AbstractScreeningDecision:
    pmid: str
    include: bool
    confidence: float
    reason: str
    labels: Dict[str, Any] = field(default_factory=dict)
    raw_llm_output: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScreeningRunResult:
    total_records: int = 0
    abstract_available: int = 0
    included_count: int = 0
    excluded_count: int = 0
    downloaded_count: int = 0
    ocr_success_count: int = 0
    decisions: List[AbstractScreeningDecision] = field(default_factory=list)
    peft_records: List[Dict[str, Any]] = field(default_factory=list)


class AbstractProvider(Protocol):
    def fetch_abstract(self, pmid: str) -> Optional[str]:
        """Fetch abstract from a legal source (e.g., PubMed/EuropePMC API)."""


class LLMJudge(Protocol):
    def evaluate_abstract(self, criteria: ScreeningCriteria, abstract: str, pmid: str) -> Dict[str, Any]:
        """Return structured decision JSON from an LLM."""


class PDFDownloaderAdapter(Protocol):
    def download(self, pmid: str) -> Any:
        """Download a PDF and return an object with file_path/status fields."""


class PDFOCRAdapter(Protocol):
    def process_pdf(self, pdf_path: str) -> Any:
        """Convert PDF to Markdown via DeepSeek OCR."""


class ReviewCriteriaLoader:
    """Load and validate YAML criteria."""

    REQUIRED_FIELDS = {
        "review_name",
        "research_field",
        "research_topic_statement",
        "methodology",
        "participant_scope",
        "disease_and_case_report_policy",
        "publication_filters",
        "decision_policy",
    }

    @classmethod
    def load(cls, yaml_path: Path) -> ScreeningCriteria:
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        missing = cls.REQUIRED_FIELDS - set(data.keys())
        if missing:
            raise ValueError(f"criteria YAML缺少关键字段: {sorted(missing)}")

        methodology = data.get("methodology", {})
        return ScreeningCriteria(
            review_name=str(data["review_name"]),
            research_field=list(data.get("research_field", [])),
            research_topic_statement=str(data.get("research_topic_statement", "")),
            methodology_include=list(methodology.get("include", [])),
            methodology_exclude=list(methodology.get("exclude", [])),
            participant_scope=dict(data.get("participant_scope", {})),
            disease_and_case_report_policy=dict(data.get("disease_and_case_report_policy", {})),
            publication_filters=dict(data.get("publication_filters", {})),
            priority_signals=list(data.get("priority_signals", [])),
            exclusion_signals=list(data.get("exclusion_signals", [])),
            decision_policy=dict(data.get("decision_policy", {})),
        )


class AbstractScreeningHarness:
    """Integration-style harness for CSV -> abstract screening -> download -> DeepSeek OCR."""

    def __init__(
        self,
        abstract_provider: AbstractProvider,
        llm_judge: LLMJudge,
        pdf_downloader: PDFDownloaderAdapter,
        deepseek_ocr: PDFOCRAdapter,
    ):
        self.abstract_provider = abstract_provider
        self.llm_judge = llm_judge
        self.pdf_downloader = pdf_downloader
        self.deepseek_ocr = deepseek_ocr

    def run(self, csv_path: Path, criteria_yaml: Path, peft_export_path: Optional[Path] = None) -> ScreeningRunResult:
        criteria = ReviewCriteriaLoader.load(criteria_yaml)
        rows = self._read_csv_rows(csv_path)

        result = ScreeningRunResult(total_records=len(rows))
        for row in rows:
            pmid = str(row.get("PMID", "")).strip()
            abstract = str(row.get("Abstract", "")).strip() or self.abstract_provider.fetch_abstract(pmid)
            if not abstract:
                continue

            result.abstract_available += 1
            llm_output = self.llm_judge.evaluate_abstract(criteria, abstract, pmid)
            decision = self._to_decision(pmid=pmid, llm_output=llm_output)
            result.decisions.append(decision)
            result.peft_records.append(self._build_peft_record(criteria, abstract, decision))

            if not decision.include:
                result.excluded_count += 1
                continue

            result.included_count += 1
            dl_result = self.pdf_downloader.download(pmid)
            if getattr(dl_result, "file_path", None):
                result.downloaded_count += 1
                ocr_result = self.deepseek_ocr.process_pdf(str(dl_result.file_path))
                if str(getattr(ocr_result, "status", "")).lower() == "success":
                    result.ocr_success_count += 1

        if peft_export_path:
            self.export_peft_dataset(result.peft_records, peft_export_path)

        return result

    @staticmethod
    def export_peft_dataset(records: List[Dict[str, Any]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _read_csv_rows(csv_path: Path) -> List[Dict[str, str]]:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    @staticmethod
    def _to_decision(pmid: str, llm_output: Dict[str, Any]) -> AbstractScreeningDecision:
        include = bool(llm_output.get("include", False))
        confidence = float(llm_output.get("confidence", 0.0))
        reason = str(llm_output.get("reason", ""))
        labels = dict(llm_output.get("labels", {}))
        return AbstractScreeningDecision(
            pmid=pmid,
            include=include,
            confidence=confidence,
            reason=reason,
            labels=labels,
            raw_llm_output=llm_output,
        )

    @staticmethod
    def _build_peft_record(
        criteria: ScreeningCriteria,
        abstract: str,
        decision: AbstractScreeningDecision,
    ) -> Dict[str, Any]:
        """Build a PEFT-friendly SFT record for future Qwen-3B fine-tuning."""
        instruction = (
            "You are a literature screener. Judge whether the study fits the review scope "
            "and return JSON with include/confidence/reason/labels."
        )
        criteria_payload = {
            "review_name": criteria.review_name,
            "research_field": criteria.research_field,
            "research_topic_statement": criteria.research_topic_statement,
            "methodology_include": criteria.methodology_include,
            "methodology_exclude": criteria.methodology_exclude,
            "participant_scope": criteria.participant_scope,
            "disease_and_case_report_policy": criteria.disease_and_case_report_policy,
            "publication_filters": criteria.publication_filters,
            "priority_signals": criteria.priority_signals,
            "exclusion_signals": criteria.exclusion_signals,
            "decision_policy": criteria.decision_policy,
        }
        return {
            "instruction": instruction,
            "input": {
                "criteria": criteria_payload,
                "abstract": abstract,
            },
            "output": {
                "include": decision.include,
                "confidence": decision.confidence,
                "reason": decision.reason,
                "labels": decision.labels,
            },
            "metadata": {
                "pmid": decision.pmid,
                "target_model": "Qwen3-3B",
                "recommended_finetune_framework": "PEFT",
                "task": "abstract_screening",
            },
        }
