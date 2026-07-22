"""Domain-neutral extraction of structured study information from paper sections.

This extractor intentionally avoids assumptions such as a language task, a
patient/control design, or fMRI-only acquisition.  It is suitable as ARneuro's
default first-pass schema; domain profiles may enrich its output later.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional


GENERIC_STUDY_SCHEMA: Dict[str, Any] = {
    "pmid": "",
    "study_type": "unknown",
    "research_modalities": [],
    "study_design": "",
    "sample_groups": [],
    "experimental_tasks": [],
    "acquisition": {
        "modalities": [],
        "scanner_or_instrument": "",
        "protocol": "",
    },
    "analysis": {
        "approaches": [],
        "software": [],
        "statistical_model": "",
    },
    "key_outcomes": [],
    "missing_or_uncertain_fields": [],
    "evidence": {},
}

SYSTEM_PROMPT = """You are extracting a reproducible, domain-neutral neuroscience study record.

Return one valid JSON object only. Do not add Markdown or commentary. Do not infer
facts that are absent. Preserve uncertainty with empty strings, empty arrays, or
the literal value \"unknown\". Evidence must quote or closely paraphrase a short
source sentence and identify its section.

Required JSON shape:
{
  "pmid":"string",
  "study_type":"experimental|observational|clinical|case_report|review|methods|unknown",
  "research_modalities":["fMRI|EEG|MEG|behavioural|lesion|TMS|other"],
  "study_design":"string",
  "sample_groups":[{
    "name":"string", "role":"participant|patient|comparison|control|case|other",
    "n":"string", "age":"string", "sex":"string", "clinical_status":"string",
    "in_scanner":"yes|no|mixed|unknown", "evidence":"string"
  }],
  "experimental_tasks":[{
    "name":"string", "category":"experimental_task|control_condition|clinical_assessment|other",
    "description":"string", "stimulus_modality":["visual|auditory|somatosensory|other"],
    "response_mode":"string", "in_scanner":"yes|no|mixed|unknown",
    "contrast":"string", "evidence":"string"
  }],
  "acquisition":{"modalities":["string"],"scanner_or_instrument":"string","protocol":"string"},
  "analysis":{"approaches":["string"],"software":["string"],"statistical_model":"string"},
  "key_outcomes":["string"],
  "missing_or_uncertain_fields":["string"],
  "evidence":{"field_path":"short source evidence"}
}
"""


def _empty_schema(pmid: str) -> Dict[str, Any]:
    result = json.loads(json.dumps(GENERIC_STUDY_SCHEMA))
    result["pmid"] = str(pmid)
    return result


def parse_json_object(value: Any) -> Dict[str, Any]:
    """Tolerantly recover a JSON object from a provider response."""

    if isinstance(value, dict):
        return value
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                return parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                pass
    raise ValueError("Could not parse a JSON object from model response.")


def _merge_schema(pmid: str, extracted: Mapping[str, Any]) -> Dict[str, Any]:
    result = _empty_schema(pmid)
    for key in result:
        if key in extracted and extracted[key] is not None:
            result[key] = extracted[key]
    result["pmid"] = str(extracted.get("pmid") or pmid)
    for list_key in ("research_modalities", "sample_groups", "experimental_tasks", "key_outcomes", "missing_or_uncertain_fields"):
        if not isinstance(result[list_key], list):
            result[list_key] = []
    for map_key in ("acquisition", "analysis", "evidence"):
        if not isinstance(result[map_key], dict):
            result[map_key] = _empty_schema(pmid)[map_key]
    return result


@dataclass
class GenericStudyExtractionResult:
    pmid: str
    data: Dict[str, Any]
    prompt_characters: int
    provider: str = ""
    model: str = ""
    error: str = ""


@dataclass
class GenericStudyInfoExtractor:
    """One-call extractor with provider injection for testability and portability.

    ``completion`` receives ``(system_prompt, user_prompt)`` and must return a
    JSON string or mapping.  Keeping it injectable makes this class usable with
    DeepSeek, OpenAI-compatible APIs, local models, or deterministic fixtures.
    """

    completion: Optional[Callable[[str, str], Any]] = None
    max_method_characters: int = 45000
    short_method_threshold: int = 3000
    provider: str = ""
    model: str = ""

    def build_source_text(self, sections: Mapping[str, Any]) -> str:
        methods = str(sections.get("Methods", "") or "").strip()
        parts = ["## Methods\n" + methods] if methods else []
        if len(methods) < self.short_method_threshold:
            for section_name in ("Results", "Introduction", "Other"):
                text = str(sections.get(section_name, "") or "").strip()
                if text:
                    parts.append(f"## {section_name}\n{text}")
        text = "\n\n".join(parts).strip()
        return text[: self.max_method_characters]

    def extract(self, pmid: str, sections: Mapping[str, Any]) -> GenericStudyExtractionResult:
        source = self.build_source_text(sections)
        if not source:
            data = _empty_schema(pmid)
            data["missing_or_uncertain_fields"] = ["Methods section unavailable"]
            return GenericStudyExtractionResult(pmid=str(pmid), data=data, prompt_characters=0, error="Methods section unavailable")
        if self.completion is None:
            raise RuntimeError("No completion callable configured for generic study extraction.")
        user_prompt = f"PMID: {pmid}\n\nStructured paper text:\n{source}"
        try:
            data = _merge_schema(str(pmid), parse_json_object(self.completion(SYSTEM_PROMPT, user_prompt)))
            return GenericStudyExtractionResult(
                pmid=str(pmid), data=data, prompt_characters=len(user_prompt), provider=self.provider, model=self.model
            )
        except Exception as exc:
            data = _empty_schema(pmid)
            data["missing_or_uncertain_fields"] = ["Initial extraction failed"]
            return GenericStudyExtractionResult(
                pmid=str(pmid), data=data, prompt_characters=len(user_prompt), provider=self.provider, model=self.model,
                error=f"{type(exc).__name__}: {exc}",
            )
