"""Detailed task extraction from a paper's complete Methods section.

This module combines task discovery, content-based task renaming, and rich
linguistic/task-design annotation in one LLM request per paper. Its output keeps
the legacy task-table keys while adding structured properties for downstream
feature extraction.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from core.llm_client import LLMClientManager
except Exception:
    try:
        from ..core.llm_client import LLMClientManager
    except Exception:
        arneuro_dir = Path(__file__).resolve().parents[1]
        if str(arneuro_dir) not in sys.path:
            sys.path.insert(0, str(arneuro_dir))
        from core.llm_client import LLMClientManager


LINGUISTIC_DOMAINS = [
    "Phonological",
    "Semantic",
    "Syntactic",
    "Orthographic",
    "Morphological",
    "Phonetic",
    "Prosodic",
    "Pragmatic/Discourse",
    "Lexical",
    "Non-linguistic",
]

PROCESSING_TYPES = [
    "Recognition/Identification",
    "Production",
    "Discrimination",
    "Judgment",
    "Generation",
    "Comprehension",
    "Repetition",
    "Reading",
    "Writing",
    "Listening",
    "Naming",
    "Memory",
    "Attention",
    "Motor response",
]

TASK_TYPES = [
    "Language sub-function test task",
    "Control experiment task",
    "Non-language cognitive task",
    "Clinical or behavioral assessment task",
]

PRODUCTION_COMPREHENSION = [
    "Production",
    "Comprehension",
    "Mixed",
    "Neither",
    "Unknown",
]

ATTENTION_ENGAGEMENT = ["Active", "Passive", "Repetition", "Unknown"]

CONTRAST_TEMPLATE = {
    "contrast_name": "",
    "conditions_compared": [],
    "direction": "",
    "exact_text": "",
}

TASK_TEMPLATE: Dict[str, Any] = {
    "name": "",
    "raw_name": "",
    "canonical_task_family": "",
    "type": "",
    "is_language_task": "unknown",
    "is_control_task": "unknown",
    "description": "",
    "clue_sentences": "",
    "purpose": "",
    "measured_constructs": [],
    "task_naming_rationale": "",
    "task_dictionary_mapping": {
        "recommended_dictionary_name": "",
        "mapping_status": "new",
        "mapping_rationale": "",
    },
    "contrast_exact_applied": "",
    "Task_property_on_Production_or_Comprehension": "Unknown",
    "Task_property_on_attention_engagement": "Unknown",
    "Task_property_on_Linguistic_Domain": [],
    "Task_property_on_Processing_Type": [],
    "Task_property_on_Stimulus_modality": [],
    "linguistic_properties": {
        "domains": [],
        "subdomains": [],
        "language_units": [],
        "linguistic_operations": [],
    },
    "stimulus_properties": {
        "modalities": [],
        "units": [],
        "types": [],
        "lexical_status": [],
        "languages": [],
        "presentation": "",
        "linguistic_complexity_or_manipulation": [],
    },
    "response_properties": {
        "response_mode": [],
        "response_device": "",
        "overt_or_covert": "Unknown",
        "response_mapping": "",
        "behavioral_measures": [],
    },
    "experimental_design": {
        "scanner_execution": "unknown",
        "imaging_modalities": [],
        "design_type": "",
        "timing": "",
        "conditions": [],
        "baseline_or_control": "",
        "randomization_or_counterbalancing": "",
        "task_instructions": "",
        "trial_structure": "",
    },
    "contrasts": [CONTRAST_TEMPLATE],
    "participant_groups": [],
    "cognitive_demands": [],
    "downstream_feature_hints": {
        "representational_features": [],
        "processing_features": [],
        "task_design_features": [],
    },
    "verbal_fluency": {
        "is_verbal_fluency_task": "False",
        "cue_type": "",
        "time_limited": "unknown",
    },
    "evidence": {
        "task_identity": [],
        "participant_action": [],
        "stimuli": [],
        "contrast": [],
    },
    "missing_or_uncertain_properties": [],
}

TASK_INFO_SCHEMA: Dict[str, Any] = {
    "PMID": "",
    "tasks": [TASK_TEMPLATE],
    "Language_use": [],
    "task_count": 0,
}


@dataclass
class TaskInfoExtractionResult:
    pmid: str
    task_info: Dict[str, Any]
    metadata: Dict[str, Any]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def write_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)
    temp_path.replace(path)


def pmid_from_path(path: Path) -> str:
    name = path.name
    match = re.match(r"^paper_(.+?)_structured_content\.json$", name)
    return match.group(1) if match else path.stem


def extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Could not parse a JSON object from model response.")


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    elif isinstance(value, tuple):
        values = list(value)
    elif isinstance(value, str):
        values = re.split(r"\s*[;|]\s*", value.strip()) if value.strip() else []
    else:
        values = [value]
    output: List[str] = []
    seen = set()
    for item in values:
        text = str(item).strip()
        key = text.casefold()
        if text and key not in seen:
            output.append(text)
            seen.add(key)
    return output


def _as_bool_string(value: Any) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    text = str(value or "").strip().lower()
    if text in {"true", "yes", "1", "present"}:
        return "True"
    if text in {"false", "no", "0", "absent"}:
        return "False"
    return "unknown"


def _dict_with_defaults(value: Any, defaults: Dict[str, Any]) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    result: Dict[str, Any] = {}
    for key, default in defaults.items():
        item = source.get(key, default)
        if isinstance(default, dict):
            result[key] = _dict_with_defaults(item, default)
        elif isinstance(default, list):
            result[key] = _as_list(item)
        else:
            result[key] = item if item is not None else default
    return result


def _normalize_contrasts(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    output: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        contrast = _dict_with_defaults(item, CONTRAST_TEMPLATE)
        contrast["conditions_compared"] = _as_list(
            contrast.get("conditions_compared")
        )
        if any(
            contrast.get(key)
            for key in [
                "contrast_name",
                "conditions_compared",
                "direction",
                "exact_text",
            ]
        ):
            output.append(contrast)
    return output


def normalize_task_info(pmid: str, raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize provider variations into the stable task information schema."""

    raw = dict(raw or {})
    if "Language_use" not in raw:
        for alias in ["Language use", "Language Used", "Language Use"]:
            if alias in raw:
                raw["Language_use"] = raw[alias]
                break

    normalized_tasks: List[Dict[str, Any]] = []
    seen_tasks = set()
    for item in raw.get("tasks", []) if isinstance(raw.get("tasks"), list) else []:
        if not isinstance(item, dict):
            continue
        aliases = {
            "contrast exact applied": "contrast_exact_applied",
            "Task property on Production or Comprehension": (
                "Task_property_on_Production_or_Comprehension"
            ),
            "Task property on attention engagement": (
                "Task_property_on_attention_engagement"
            ),
            "Task property on Linguistic Domain": (
                "Task_property_on_Linguistic_Domain"
            ),
            "Task property on   Linguistic Domain": (
                "Task_property_on_Linguistic_Domain"
            ),
            "Task property on Processing Type": (
                "Task_property_on_Processing_Type"
            ),
            "Task property on Stimulus modality": (
                "Task_property_on_Stimulus_modality"
            ),
            "clue\\_sentences": "clue_sentences",
            "clue sentences": "clue_sentences",
        }
        task_source = dict(item)
        for old_key, new_key in aliases.items():
            if old_key in task_source and new_key not in task_source:
                task_source[new_key] = task_source[old_key]
        task = _dict_with_defaults(task_source, TASK_TEMPLATE)
        task["contrasts"] = _normalize_contrasts(task_source.get("contrasts"))
        task["name"] = str(task.get("name", "")).strip()
        task["raw_name"] = str(task.get("raw_name", "")).strip()
        task["canonical_task_family"] = str(
            task.get("canonical_task_family", "")
        ).strip()
        task["type"] = str(task.get("type", "")).strip()
        if task["type"] not in TASK_TYPES:
            task["type"] = (
                "Language sub-function test task"
                if _as_bool_string(task.get("is_language_task")) == "True"
                else "Non-language cognitive task"
            )
        task["is_language_task"] = _as_bool_string(task.get("is_language_task"))
        task["is_control_task"] = _as_bool_string(task.get("is_control_task"))
        if task["is_language_task"] == "unknown":
            task["is_language_task"] = (
                "True"
                if task["type"] == "Language sub-function test task"
                else "False"
            )
        if task["is_control_task"] == "unknown":
            task["is_control_task"] = (
                "True" if task["type"] == "Control experiment task" else "False"
            )
        if task["is_control_task"] == "True":
            task["type"] = "Control experiment task"
        for key in [
            "Task_property_on_Linguistic_Domain",
            "Task_property_on_Processing_Type",
            "Task_property_on_Stimulus_modality",
            "participant_groups",
            "cognitive_demands",
            "measured_constructs",
            "missing_or_uncertain_properties",
        ]:
            task[key] = _as_list(task.get(key))
        pc_value = str(
            task.get("Task_property_on_Production_or_Comprehension", "Unknown")
        ).strip()
        task["Task_property_on_Production_or_Comprehension"] = (
            pc_value if pc_value in PRODUCTION_COMPREHENSION else "Unknown"
        )
        attention = str(
            task.get("Task_property_on_attention_engagement", "Unknown")
        ).strip()
        task["Task_property_on_attention_engagement"] = (
            attention if attention in ATTENTION_ENGAGEMENT else "Unknown"
        )
        domains = _as_list(task["Task_property_on_Linguistic_Domain"])
        nested_domains = _as_list(task["linguistic_properties"].get("domains"))
        task["Task_property_on_Linguistic_Domain"] = _as_list(
            domains + nested_domains
        )
        task["linguistic_properties"]["domains"] = list(
            task["Task_property_on_Linguistic_Domain"]
        )
        modalities = _as_list(task["Task_property_on_Stimulus_modality"])
        nested_modalities = _as_list(
            task["stimulus_properties"].get("modalities")
        )
        task["Task_property_on_Stimulus_modality"] = _as_list(
            modalities + nested_modalities
        )
        task["stimulus_properties"]["modalities"] = list(
            task["Task_property_on_Stimulus_modality"]
        )
        clue = task.get("clue_sentences", "")
        if isinstance(clue, list):
            task["clue_sentences"] = " ... ".join(_as_list(clue))
        else:
            task["clue_sentences"] = str(clue or "").strip()
        task["verbal_fluency"]["is_verbal_fluency_task"] = _as_bool_string(
            task["verbal_fluency"].get("is_verbal_fluency_task")
        )
        task["verbal_fluency"]["time_limited"] = _as_bool_string(
            task["verbal_fluency"].get("time_limited")
        )
        task["experimental_design"]["scanner_execution"] = _as_bool_string(
            task["experimental_design"].get("scanner_execution")
        )
        identity = re.sub(r"[^a-z0-9]+", " ", task["name"].lower()).strip()
        if not task["name"] or identity in seen_tasks:
            continue
        seen_tasks.add(identity)
        normalized_tasks.append(task)

    language_use = _as_list(raw.get("Language_use"))
    for task in normalized_tasks:
        for language in task["stimulus_properties"]["languages"]:
            if language.casefold() not in {x.casefold() for x in language_use}:
                language_use.append(language)

    return {
        "PMID": str(pmid),
        "tasks": normalized_tasks,
        "Language_use": language_use,
        "task_count": len(normalized_tasks),
    }


class MethodTaskInfoExtractor:
    """Provider-independent detailed task extractor."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        client_type: str = "deepseek",
        model_name: str = "deepseek-v4-flash",
        max_retries: int = 1,
        max_tokens: int = 32768,
        request_interval_seconds: float = 0.0,
        fallback_model_name: Optional[str] = None,
        fallback_max_retries: int = 2,
        context_window_tokens: int = 384000,
    ) -> None:
        self.config = config or {}
        self.client_type = client_type
        self.model_name = model_name
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)
        self.request_interval_seconds = float(request_interval_seconds)
        self.fallback_model_name = fallback_model_name
        self.fallback_max_retries = int(fallback_max_retries)
        self.context_window_tokens = int(context_window_tokens)
        self.call_count = 0
        self._client_manager = LLMClientManager(self.config)
        self._clients: Dict[str, Tuple[Any, str]] = {}
        self._last_request_finished_at = 0.0

    def get_client(self, model_name: Optional[str] = None):
        requested_model = model_name or self.model_name
        if requested_model not in self._clients:
            self._clients[requested_model] = self._client_manager.get_client(
                client_type=self.client_type,
                model_name=requested_model,
            )
        return self._clients[requested_model]

    def build_messages(
        self,
        pmid: str,
        methods_text: str,
        title_context: str = "",
    ) -> List[Dict[str, str]]:
        schema = json.dumps(TASK_INFO_SCHEMA, ensure_ascii=False, indent=2)
        system_prompt = f"""
You are an expert in cognitive neuroscience, psycholinguistics, and experimental
task ontology. Read the COMPLETE Methods section, discover every distinct task,
and rename and annotate each task in one pass.

TASK SCOPE
1. Include scanner tasks, behavioral tasks, training/practice tasks, localizers,
   clinical assessments, and explicit baseline/control conditions when the
   participant performs or experiences a defined procedure.
2. Do not invent a task from image preprocessing, statistical analysis, generic
   recruitment, or reported results.
3. Keep genuinely distinct experiments/conditions as separate tasks when their
   participant action, stimulus, purpose, or contrast differs. Merge repeated
   descriptions of the same task.

CONTENT-BASED NAMING
1. name must be a grammatical, specific task name based on what participants do,
   not the paper's abbreviation. raw_name preserves the paper's label.
2. Include linguistic unit and operation when supported: letter, phoneme,
   syllable, word, pseudoword, phrase, sentence, discourse, picture/object.
3. Include modality in name only when explicitly reported. Never infer visual,
   auditory, or written modality from a generic label such as "Word
   Comprehension Task".
4. A speeded or time-limited request to generate as many words as possible is a
   Verbal Fluency Task. Preserve cue type: letter/phonemic, semantic/category,
   action/verb, or other reported cue.
5. Naming examples:
   - Subjects pressed buttons for real words -> Word Recognition Task
   - Classify artificial grammar strings -> Artificial Grammar Judgment Task
   - Identify visual vowels -> Letter Identification Task
   - Write words from audio -> Auditory Word Writing Task
   - Generate words with a given character -> Character-Cued Word Generation Task
   - Generate as many words as possible in one minute -> Verbal Fluency Task
   - Judge whether letters rhyme -> Letter Rhyming Judgment Task

ANNOTATION RULES
1. Linguistic domains use only supported values from: {LINGUISTIC_DOMAINS}.
2. Processing types use supported values from: {PROCESSING_TYPES}.
3. Stimulus modalities must be explicit: visual, auditory, written, audiovisual,
   tactile, or other reported modality. Use [] when unreported.
4. stimulus_properties.units records linguistic/material units; types records
   pictures, tones, symbols, natural speech, text, etc.; lexical_status records
   real words, pseudowords/nonwords, artificial grammar, or other status.
5. description must comprehensively state participant action, stimuli,
   language, response, timing/design, and conditions when reported.
6. clue_sentences must reproduce the minimum Methods wording that proves the
   task. Do not fabricate quotations.
7. purpose states the task's experimental role. contrast_exact_applied preserves
   the exact reported contrast text, otherwise "".
8. contrasts is a list of structured objects with keys:
   contrast_name, conditions_compared, direction, exact_text.
9. Production/Comprehension is Production, Comprehension, Mixed, Neither, or
   Unknown. Attention engagement is Active, Passive, Repetition, or Unknown.
10. scanner_execution is True only when explicitly performed during imaging,
    False when explicitly outside the scanner, otherwise unknown.
11. Language_use contains unique human languages used by tasks. Do not include
    programming languages or the word "language" without a named language.
12. Missing evidence must remain empty/unknown and be listed in
    missing_or_uncertain_properties. Never guess.
13. measured_constructs names what the task measures, such as lexical access,
    phoneme discrimination, syntactic parsing, semantic integration, verbal
    fluency, working memory, or motor control.
14. task_dictionary_mapping recommends a reusable dictionary label. mapping_status
    is exact, near, or new. task_naming_rationale briefly explains why the final
    content-based name is accurate.
15. behavioral_measures contains only explicitly collected measures such as
    accuracy, reaction time, error rate, speech output, or button response.
16. downstream_feature_hints maps evidence into three lists compatible with
    ExtractFeatureJson: representational_features, processing_features, and
    task_design_features. Use concise ontology-like labels, not prose.
17. Do not use words such as "likely", "probably", or "presumably" to fill
    missing task details. Record uncertainty instead.

Return exactly one JSON object using every key in this schema. Add no prose:
{schema}
""".strip()
        user_payload = {
            "PMID": str(pmid),
            "Title_context": str(title_context or "")[:1500],
            "Complete_Methods_section": str(methods_text or ""),
        }
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]

    def _wait_for_slot(self) -> None:
        remaining = self.request_interval_seconds - (
            time.monotonic() - self._last_request_finished_at
        )
        if remaining > 0:
            time.sleep(remaining)

    def call_model(
        self,
        messages: List[Dict[str, str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        primary_errors: List[str] = []
        fallback_errors: List[str] = []
        total_attempts = 0
        raw_preview = ""

        def run_tier(
            requested_model: str,
            retry_limit: int,
            errors: List[str],
            tier_name: str,
        ) -> Tuple[Dict[str, Any], str, int]:
            nonlocal total_attempts, raw_preview
            client, resolved_model = self.get_client(requested_model)
            for retry_index in range(retry_limit + 1):
                try:
                    self._wait_for_slot()
                    self.call_count += 1
                    total_attempts += 1
                    kwargs: Dict[str, Any] = {
                        "model": resolved_model,
                        "messages": messages,
                        "temperature": 0.0,
                        "max_tokens": self.max_tokens,
                        "response_format": {"type": "json_object"},
                        "stream": False,
                    }
                    if self.client_type == "deepseek":
                        kwargs["extra_body"] = {
                            "thinking": {"type": "disabled"}
                        }
                    response = client.chat.completions.create(**kwargs)
                    self._last_request_finished_at = time.monotonic()
                    raw_text = response.choices[0].message.content or ""
                    raw_preview = raw_text[:1500]
                    return (
                        extract_json_object(raw_text),
                        resolved_model,
                        retry_index,
                    )
                except Exception as exc:
                    self._last_request_finished_at = time.monotonic()
                    error = f"{type(exc).__name__}: {exc}"
                    errors.append(error)
                    if retry_index >= retry_limit:
                        break
                    lower = error.lower()
                    rate_limited = "429" in lower or "rate limit" in lower
                    delay = min(
                        (30 if rate_limited else 2) * (2**retry_index),
                        300 if rate_limited else 30,
                    )
                    delay += random.uniform(0.0, 1.0)
                    print(
                        f"  {tier_name} extraction failed ({error}); retry "
                        f"{retry_index + 1}/{retry_limit} in {delay:.1f}s"
                    )
                    time.sleep(delay)
            return {}, resolved_model, retry_limit

        raw, used_model, primary_retry_count = run_tier(
            self.model_name,
            self.max_retries,
            primary_errors,
            "Primary model",
        )
        fallback_used = False
        fallback_retry_count = 0
        if not raw and self.fallback_model_name:
            fallback_used = True
            print(
                f"  Primary model failed after {self.max_retries + 1} attempts; "
                f"switching to {self.fallback_model_name} with thinking disabled."
            )
            raw, used_model, fallback_retry_count = run_tier(
                self.fallback_model_name,
                self.fallback_max_retries,
                fallback_errors,
                "Fallback model",
            )

        all_errors = primary_errors + fallback_errors
        return raw, {
            "model": used_model,
            "primary_model": self.model_name,
            "fallback_model": self.fallback_model_name or "",
            "fallback_used": fallback_used,
            "client_type": self.client_type,
            "thinking": "disabled" if self.client_type == "deepseek" else "",
            "primary_retry_count": primary_retry_count,
            "fallback_retry_count": fallback_retry_count,
            "total_attempts": total_attempts,
            "max_output_tokens": self.max_tokens,
            "context_window_tokens": self.context_window_tokens,
            "error": "" if raw else (all_errors[-1] if all_errors else ""),
            "attempt_errors": all_errors,
            "raw_generation_preview": raw_preview,
        }

    def extract_from_content(
        self,
        pmid: str,
        content: Dict[str, Any],
        source_file: str = "",
    ) -> TaskInfoExtractionResult:
        methods = str(content.get("Methods", "") or "").strip()
        metadata = {
            "pmid": str(pmid),
            "source_file": source_file,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "methods_chars": len(methods),
            "source_sections_used": ["Methods"],
            "status": "pending",
            "error": "",
            "model": self.model_name,
            "primary_model": self.model_name,
            "fallback_model": self.fallback_model_name or "",
            "context_window_tokens": self.context_window_tokens,
            "client_type": self.client_type,
        }
        if not methods:
            metadata["status"] = "empty_methods"
            metadata["error"] = "Methods section is empty."
            return TaskInfoExtractionResult(
                str(pmid),
                normalize_task_info(str(pmid), {}),
                metadata,
            )
        raw, call_meta = self.call_model(
            self.build_messages(
                pmid=str(pmid),
                methods_text=methods,
                title_context=str(content.get("Title", "") or ""),
            )
        )
        metadata.update(call_meta)
        metadata["status"] = "success" if raw else "error"
        if not raw:
            metadata["error"] = call_meta.get("error", "Model returned no JSON.")
        task_info = normalize_task_info(str(pmid), raw)
        metadata["task_count"] = task_info["task_count"]
        return TaskInfoExtractionResult(str(pmid), task_info, metadata)

    def extract_from_content_file(
        self,
        content_path: Path,
        output_json_dir: Optional[Path] = None,
        overwrite: bool = False,
    ) -> TaskInfoExtractionResult:
        pmid = pmid_from_path(content_path)
        output_path = (
            output_json_dir / f"paper_{pmid}_task_info.json"
            if output_json_dir is not None
            else None
        )
        if output_path and output_path.exists() and not overwrite:
            existing = read_json(output_path)
            metadata = existing.get("metadata", {})
            task_info = existing.get("task_info", {})
            if metadata.get("status") == "success" and isinstance(task_info, dict):
                return TaskInfoExtractionResult(pmid, task_info, metadata)
        try:
            content = read_json(content_path)
            result = self.extract_from_content(
                pmid=pmid,
                content=content,
                source_file=str(content_path),
            )
        except Exception as exc:
            result = TaskInfoExtractionResult(
                pmid=pmid,
                task_info=normalize_task_info(pmid, {}),
                metadata={
                    "pmid": pmid,
                    "source_file": str(content_path),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "task_count": 0,
                },
            )
        if output_path:
            write_json(
                {
                    "pmid": result.pmid,
                    "task_info": result.task_info,
                    "metadata": result.metadata,
                },
                output_path,
            )
        return result

    def extract_pmids_serial(
        self,
        segmented_dir: Path,
        pmids: Iterable[str],
        output_json_dir: Path,
        overwrite: bool = False,
    ) -> List[TaskInfoExtractionResult]:
        ordered = sorted(
            {str(pmid).strip() for pmid in pmids if str(pmid).strip()},
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        )
        results: List[TaskInfoExtractionResult] = []
        for index, pmid in enumerate(ordered, start=1):
            content_path = segmented_dir / f"paper_{pmid}_structured_content.json"
            print(f"[{index}/{len(ordered)}] Extracting task information: PMID {pmid}")
            if not content_path.exists():
                result = TaskInfoExtractionResult(
                    pmid,
                    normalize_task_info(pmid, {}),
                    {
                        "pmid": pmid,
                        "source_file": str(content_path),
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "error",
                        "error": "Structured content file not found.",
                        "task_count": 0,
                    },
                )
                write_json(
                    {
                        "pmid": result.pmid,
                        "task_info": result.task_info,
                        "metadata": result.metadata,
                    },
                    output_json_dir / f"paper_{pmid}_task_info.json",
                )
            else:
                result = self.extract_from_content_file(
                    content_path,
                    output_json_dir=output_json_dir,
                    overwrite=overwrite,
                )
            results.append(result)
            print(
                f"  status={result.metadata.get('status')} "
                f"tasks={result.task_info.get('task_count', 0)}"
            )
        return results


class DeepSeekMethodTaskInfoExtractor(MethodTaskInfoExtractor):
    """DeepSeek task extractor with non-thinking Pro fallback."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "deepseek-v4-flash",
        fallback_model_name: Optional[str] = "deepseek-v4-pro",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config={"deepseek_api_key": api_key},
            client_type="deepseek",
            model_name=model_name,
            fallback_model_name=fallback_model_name,
            **kwargs,
        )
