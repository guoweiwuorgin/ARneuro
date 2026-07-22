"""
Method-section information extraction for structured ARneuro papers.

This module extracts participant, task, scanner, language, and analysis-method
metadata from the already segmented Methods section. The extractor is designed
to be economical: each paper is sent to the LLM once, using only structured
Methods text plus a small amount of paper context.
"""

from __future__ import annotations

import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from core.llm_client import LLMClientManager
except Exception:
    LLMClientManager = None


METHOD_INFO_COLUMNS = [
    "Second Language Research",
    "patient_patients",
    "typical_human_number_of_excluded_subjects",
    "patient_male_number",
    "patient_involved_tasks_names",
    "scanner_type",
    "PMID",
    "patient_disease_type",
    "patient_number_of_excluded_subjects",
    "patient_age_range",
    "typical_human_right_handed",
    "patient_all_participated_in_fMRI_task",
    "ROI_analysis",
    "typical_human_involved_tasks_names",
    "patient_right_handed",
    "typical_human_all_participated_in_fMRI_task",
    "multivariate_analysis",
    "Language Used in current research",
    "patient_total",
    "univariate_analysis",
    "language_localizer_task_analysis",
    "typical_human_total",
    "case_report",
    "whole_brain_analysis",
    "typical_human_male_number",
    "Case report",
    "typical_human_age_range",
]


RAW_SCHEMA = {
    "PMID": "",
    "Second Language Research": "unknown",
    "Language Used in current research": "",
    "case_report": "unknown",
    "Case report": "unknown",
    "scanner_type": "",
    "patient_patients": "unknown",
    "patient_disease_type": "",
    "patient_total": "",
    "patient_male_number": "",
    "patient_age_range": "",
    "patient_right_handed": "unknown",
    "patient_number_of_excluded_subjects": "",
    "patient_involved_tasks_names": [],
    "patient_all_participated_in_fMRI_task": "unknown",
    "typical_human_total": "",
    "typical_human_male_number": "",
    "typical_human_age_range": "",
    "typical_human_right_handed": "unknown",
    "typical_human_number_of_excluded_subjects": "",
    "typical_human_involved_tasks_names": [],
    "typical_human_all_participated_in_fMRI_task": "unknown",
    "whole_brain_analysis": "unknown",
    "ROI_analysis": "unknown",
    "univariate_analysis": "unknown",
    "multivariate_analysis": [],
    "language_localizer_task_analysis": "unknown",
    "missing_or_uncertain_fields": [],
    "evidence": {},
}


LEGACY_CONTROL_FIELD_MAP = {
    "control_total": "typical_human_total",
    "control_male_number": "typical_human_male_number",
    "control_age_range": "typical_human_age_range",
    "control_right_handed": "typical_human_right_handed",
    "control_number_of_excluded_subjects": (
        "typical_human_number_of_excluded_subjects"
    ),
    "control_involved_tasks_names": "typical_human_involved_tasks_names",
    "control_all_participated_in_fMRI_task": (
        "typical_human_all_participated_in_fMRI_task"
    ),
}


@dataclass
class MethodInfoExtractionResult:
    pmid: str
    raw_json: Dict[str, Any]
    flat_record: Dict[str, Any]
    metadata: Dict[str, Any]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=METHOD_INFO_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record.get(column, "") for column in METHOD_INFO_COLUMNS})


def pmid_from_path(path: Path) -> str:
    stem = path.stem.replace("_structured_content", "")
    return stem.replace("paper_", "", 1) if stem.startswith("paper_") else stem


def as_bool_string(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    value_str = str(value or "").strip()
    if not value_str:
        return "unknown"
    lower = value_str.lower()
    if lower in {"true", "yes", "y", "1", "present"}:
        return "True"
    if lower in {"false", "no", "n", "0", "absent"}:
        return "False"
    if lower in {"unknown", "unclear", "not reported", "not specified", "n/a", "na"}:
        return "unknown"
    return value_str


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    value_str = str(value).strip()
    if not value_str:
        return []
    return [item.strip() for item in re.split(r"[;,]\s*", value_str) if item.strip()]


def join_list(value: Any) -> str:
    return ", ".join(normalize_list(value))


def normalize_experiment_ordered_value(value: Any) -> str:
    """Store one value per experiment as an ordered comma-separated string."""

    if value is None:
        return ""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        return ", ".join(
            str(item).strip() for item in value if str(item).strip()
        )
    return str(value).strip()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(text[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse a JSON object from model response.")


class MethodSectionInfoExtractor:
    """
    Extract structured method information from segmented paper JSON files.

    The extractor sends one request per paper to MiMo by default. It writes a
    per-paper JSON record and can aggregate all flat records into one CSV table.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        client_type: str = "mimo",
        model_name: str = "mimo-v2.5-pro",
        max_method_chars: int = 60000,
        request_sleep_seconds: float = 0.4,
        max_retries: int = 2,
        short_methods_threshold: int = 3000,
        supplementary_section_chars: int = 12000,
        max_tokens: int = 8192,
    ) -> None:
        self.config = config or {}
        self.client_type = client_type
        self.model_name = model_name
        self.max_method_chars = max_method_chars
        self.request_sleep_seconds = request_sleep_seconds
        self.max_retries = max_retries
        self.short_methods_threshold = int(short_methods_threshold)
        self.supplementary_section_chars = int(supplementary_section_chars)
        self.max_tokens = int(max_tokens)
        self.call_count = 0
        self._client = None
        self._resolved_model_name = None

    def get_client(self):
        if self._client is not None:
            return self._client, self._resolved_model_name

        if LLMClientManager is None:
            raise RuntimeError("LLMClientManager could not be imported from ARneuro.core.")

        manager = LLMClientManager(self.config)
        client, model_name = manager.get_client(
            client_type=self.client_type,
            model_name=self.model_name,
        )
        self._client = client
        self._resolved_model_name = model_name
        return client, model_name

    @staticmethod
    def compact_text(text: str, max_chars: int) -> str:
        text = str(text or "").strip()
        if len(text) <= max_chars:
            return text
        separator = "\n\n[... section compacted ...]\n\n"
        available = max_chars - len(separator) * 2
        head_chars = int(available * 0.45)
        middle_chars = int(available * 0.20)
        tail_chars = available - head_chars - middle_chars
        middle_start = max(0, (len(text) - middle_chars) // 2)
        return (
            text[:head_chars]
            + separator
            + text[middle_start : middle_start + middle_chars]
            + separator
            + text[-tail_chars:]
        )

    def build_source_payload(
        self,
        pmid: str,
        content: Dict[str, Any],
    ) -> Tuple[Dict[str, str], List[str]]:
        methods = str(content.get("Methods", "") or "").strip()
        payload = {
            "PMID": str(pmid),
            "Title_context": str(content.get("Title", "") or "")[:1000],
            "Abstract_context": str(content.get("Abstract", "") or "")[:2500],
            "Methods_section": self.compact_text(methods, self.max_method_chars),
        }
        sections_used = ["Methods"]

        if len(methods) < self.short_methods_threshold:
            results = self.compact_text(
                str(content.get("Results", "") or ""),
                self.supplementary_section_chars,
            )
            other = self.compact_text(
                str(content.get("Other", "") or ""),
                self.supplementary_section_chars,
            )
            if results:
                payload["Results_supplement"] = results
                sections_used.append("Results")
            if other:
                payload["Other_supplement"] = other
                sections_used.append("Other")
        return payload, sections_used

    def build_prompt(self, pmid: str, content: Dict[str, Any]) -> List[Dict[str, str]]:
        user_payload, _sections_used = self.build_source_payload(pmid, content)
        schema_text = json.dumps(RAW_SCHEMA, ensure_ascii=False, indent=2)
        system_prompt = f"""
Extract neuroscience study information into exactly one JSON object.
Return JSON only. Use every schema key exactly once and add no keys.

1. Methods is primary evidence. Results/Other are supplementary only when
   provided because Methods is shorter than {self.short_methods_threshold} characters.
2. Never guess. Missing text/count/range="", list=[], uncertain boolean="unknown".
   Boolean values are only "True", "False", or "unknown".
3. patient_* is only for diagnosed clinical, neurological, psychiatric, lesion,
   or disease groups.
4. typical_human_* covers every non-patient human participant in the analyzed
   sample, including healthy-only studies and healthy comparison groups. The
   paper need not call them controls.
5. Totals and male counts describe final analyzed samples. Exclusion counts are
   only explicit exclusions. Never calculate an unstated value. If separate
   experiments have separate participant groups, preserve experiment order and
   return one value per experiment separated by commas. Example: Experiment 1
   n=9, Experiment 2 n=8, Experiment 3 n=7 must be typical_human_total="9, 8, 7",
   not 24. Apply the same ordered format to patient totals, male counts,
   exclusion counts, and age ranges when each experiment has its own value.
   Do not merge overlapping or reused samples.
6. Use a reported age range only; mean±SD is not a range. Right-handed=True only
   if every participant in that category is right-handed.
7. Task fields contain experimental/scanning tasks actually performed.
   all_participated_in_fMRI_task=True only when all analyzed participants in
   that category completed the relevant neuroimaging task.
8. List research languages comma-separated. Second Language Research=True only
   when L2/bilingualism/multilingualism/translation/non-native or cross-language
   processing is central.
9. scanner_type includes modality, field strength and scanner model if reported.
10. whole_brain_analysis requires whole-brain/whole-cortex inference, not merely
    acquisition. ROI_analysis requires ROI/mask/seed inference.
    univariate_analysis includes GLM/contrasts/t-tests/ANOVA/mass-univariate tests.
    multivariate_analysis lists exact methods such as MVPA, decoding, RSA, SVM,
    encoding, connectivity, ICA/PCA, or machine learning. Standard GLM is not one.
11. language_localizer_task_analysis=True only for a dedicated language localizer.
12. case_report and "Case report" must match; True only for a single case.
13. missing_or_uncertain_fields lists unresolved keys. evidence contains short
    source phrases only for important non-empty fields.

Schema:
{schema_text}
""".strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

    def call_model(self, messages: List[Dict[str, str]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        client, model_name = self.get_client()
        last_error = ""
        for retry_idx in range(self.max_retries + 1):
            try:
                self.call_count += 1
                response = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"},
                    stream=False,
                )
                time.sleep(self.request_sleep_seconds)
                raw_text = response.choices[0].message.content
                return extract_json_object(raw_text), {
                    "model": model_name,
                    "client_type": self.client_type,
                    "error": "",
                    "retry_count": retry_idx,
                }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if retry_idx < self.max_retries:
                    time.sleep(min(2 ** retry_idx, 8))

        return {}, {
            "model": model_name,
            "client_type": self.client_type,
            "error": last_error,
            "retry_count": self.max_retries,
        }

    def coerce_raw_json(self, pmid: str, raw_json: Dict[str, Any]) -> Dict[str, Any]:
        raw_json = dict(raw_json or {})
        for legacy_key, new_key in LEGACY_CONTROL_FIELD_MAP.items():
            if new_key not in raw_json and legacy_key in raw_json:
                raw_json[new_key] = raw_json[legacy_key]

        missing_fields = normalize_list(raw_json.get("missing_or_uncertain_fields"))
        raw_json["missing_or_uncertain_fields"] = [
            LEGACY_CONTROL_FIELD_MAP.get(field, field) for field in missing_fields
        ]
        evidence = raw_json.get("evidence")
        if isinstance(evidence, dict):
            raw_json["evidence"] = {
                LEGACY_CONTROL_FIELD_MAP.get(str(key), str(key)): value
                for key, value in evidence.items()
            }

        data = dict(RAW_SCHEMA)
        for key, default_value in RAW_SCHEMA.items():
            if key in raw_json:
                data[key] = raw_json[key]
            else:
                data[key] = default_value

        data["PMID"] = str(pmid)

        for key in [
            "Second Language Research",
            "case_report",
            "Case report",
            "patient_patients",
            "patient_right_handed",
            "patient_all_participated_in_fMRI_task",
            "typical_human_right_handed",
            "typical_human_all_participated_in_fMRI_task",
            "whole_brain_analysis",
            "ROI_analysis",
            "univariate_analysis",
            "language_localizer_task_analysis",
        ]:
            data[key] = as_bool_string(data.get(key))

        for key in [
            "patient_involved_tasks_names",
            "typical_human_involved_tasks_names",
            "multivariate_analysis",
            "missing_or_uncertain_fields",
        ]:
            data[key] = normalize_list(data.get(key))

        for key in [
            "patient_total",
            "patient_male_number",
            "patient_age_range",
            "patient_number_of_excluded_subjects",
            "typical_human_total",
            "typical_human_male_number",
            "typical_human_age_range",
            "typical_human_number_of_excluded_subjects",
        ]:
            data[key] = normalize_experiment_ordered_value(data.get(key))

        if not isinstance(data.get("evidence"), dict):
            data["evidence"] = {}

        patient_signal = any(
            str(data.get(key, "") or "").strip()
            for key in ["patient_total", "patient_disease_type"]
        )
        typical_human_signal = bool(
            str(data.get("typical_human_total", "") or "").strip()
        )
        if patient_signal and data["patient_patients"] == "unknown":
            data["patient_patients"] = "True"
        elif (
            typical_human_signal
            and not patient_signal
            and data["patient_patients"] == "unknown"
        ):
            data["patient_patients"] = "False"

        if data["patient_patients"] == "False":
            for key in [
                "patient_disease_type",
                "patient_total",
                "patient_male_number",
                "patient_age_range",
                "patient_number_of_excluded_subjects",
            ]:
                data[key] = ""
            data["patient_involved_tasks_names"] = []
            data["patient_right_handed"] = "unknown"
            data["patient_all_participated_in_fMRI_task"] = "unknown"

        return data

    def flatten_record(self, pmid: str, raw_json: Dict[str, Any]) -> Dict[str, Any]:
        record = {column: "" for column in METHOD_INFO_COLUMNS}
        for column in METHOD_INFO_COLUMNS:
            value = raw_json.get(column, "")
            if isinstance(value, list):
                record[column] = join_list(value)
            elif isinstance(value, dict):
                record[column] = json.dumps(value, ensure_ascii=False)
            else:
                record[column] = value
        record["PMID"] = str(pmid)
        record["patient_involved_tasks_names"] = join_list(raw_json.get("patient_involved_tasks_names"))
        record["typical_human_involved_tasks_names"] = join_list(
            raw_json.get("typical_human_involved_tasks_names")
        )
        record["multivariate_analysis"] = join_list(raw_json.get("multivariate_analysis"))
        return record

    def extract_from_content(
        self,
        pmid: str,
        content: Dict[str, Any],
        source_file: Optional[str] = None,
    ) -> MethodInfoExtractionResult:
        methods = str(content.get("Methods", "") or "").strip()
        _payload, sections_used = self.build_source_payload(pmid, content)
        metadata = {
            "pmid": pmid,
            "source_file": source_file or "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "method_chars": len(methods),
            "short_methods_threshold": self.short_methods_threshold,
            "source_sections_used": sections_used,
            "client_type": self.client_type,
            "model": self.model_name,
            "call_count_before": self.call_count,
            "status": "pending",
            "error": "",
        }

        if not methods and sections_used == ["Methods"]:
            raw_json = self.coerce_raw_json(pmid, {"PMID": pmid})
            raw_json["missing_or_uncertain_fields"] = METHOD_INFO_COLUMNS[:]
            flat = self.flatten_record(pmid, raw_json)
            metadata["status"] = "empty_methods"
            metadata["error"] = "Methods section is empty."
            return MethodInfoExtractionResult(pmid, raw_json, flat, metadata)

        messages = self.build_prompt(pmid, content)
        extracted, call_meta = self.call_model(messages)
        metadata.update(call_meta)
        metadata["call_count_after"] = self.call_count
        metadata["status"] = "success" if extracted else "error"

        if not extracted:
            extracted = {"PMID": pmid}
            metadata["error"] = call_meta.get("error", "Model returned no JSON.")

        raw_json = self.coerce_raw_json(pmid, extracted)
        flat = self.flatten_record(pmid, raw_json)
        return MethodInfoExtractionResult(pmid, raw_json, flat, metadata)

    def extract_from_content_file(
        self,
        content_path: Path,
        output_json_dir: Optional[Path] = None,
        overwrite: bool = False,
    ) -> MethodInfoExtractionResult:
        pmid = pmid_from_path(content_path)
        output_json_path = None
        if output_json_dir is not None:
            output_json_path = output_json_dir / f"paper_{pmid}_method_info.json"
            if output_json_path.exists() and not overwrite:
                existing = read_json(output_json_path)
                return MethodInfoExtractionResult(
                    pmid=pmid,
                    raw_json=existing.get("extracted_json", {}),
                    flat_record=existing.get("flat_record", {}),
                    metadata=existing.get("metadata", {}),
                )

        content = read_json(content_path)
        result = self.extract_from_content(pmid=pmid, content=content, source_file=str(content_path))

        if output_json_path is not None:
            write_json(
                {
                    "pmid": result.pmid,
                    "extracted_json": result.raw_json,
                    "flat_record": result.flat_record,
                    "metadata": result.metadata,
                },
                output_json_path,
            )

        return result

    def extract_directory(
        self,
        segmented_dir: Path,
        output_json_dir: Path,
        output_csv_path: Path,
        limit: Optional[int] = None,
        overwrite: bool = False,
        skip_pmids: Optional[Iterable[str]] = None,
        workers: int = 1,
    ) -> List[Dict[str, Any]]:
        skip_set = {str(pmid) for pmid in (skip_pmids or [])}
        content_files = [
            path for path in sorted(segmented_dir.glob("paper_*_structured_content.json"))
            if pmid_from_path(path) not in skip_set
        ]
        if limit is not None:
            content_files = content_files[:limit]

        records: List[Dict[str, Any]] = []

        if workers <= 1:
            for idx, content_path in enumerate(content_files, start=1):
                print(f"[{idx}/{len(content_files)}] Extracting method info: {content_path.name}")
                result = self.extract_from_content_file(
                    content_path=content_path,
                    output_json_dir=output_json_dir,
                    overwrite=overwrite,
                )
                records.append(result.flat_record)
                write_csv(records, output_csv_path)
            return records

        def run_one(content_path: Path) -> Dict[str, Any]:
            local_extractor = MethodSectionInfoExtractor(
                config=self.config,
                client_type=self.client_type,
                model_name=self.model_name,
                max_method_chars=self.max_method_chars,
                request_sleep_seconds=self.request_sleep_seconds,
                max_retries=self.max_retries,
                short_methods_threshold=self.short_methods_threshold,
                supplementary_section_chars=self.supplementary_section_chars,
                max_tokens=self.max_tokens,
            )
            result = local_extractor.extract_from_content_file(
                content_path=content_path,
                output_json_dir=output_json_dir,
                overwrite=overwrite,
            )
            return result.flat_record

        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(run_one, content_path): content_path
                for content_path in content_files
            }
            for future in as_completed(future_to_path):
                completed += 1
                content_path = future_to_path[future]
                print(f"[{completed}/{len(content_files)}] Finished method info: {content_path.name}")
                try:
                    records.append(future.result())
                except Exception as exc:
                    pmid = pmid_from_path(content_path)
                    records.append({
                        column: "" for column in METHOD_INFO_COLUMNS
                    })
                    records[-1]["PMID"] = pmid
                    print(f"  ERROR {content_path.name}: {type(exc).__name__}: {exc}")
                write_csv(records, output_csv_path)

        return records
