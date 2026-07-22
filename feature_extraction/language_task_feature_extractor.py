"""Language-task feature embedding from detailed task information.

This module converts one extracted task description into a binary feature
embedding. Values follow the legacy convention:

0 = feature is present
1 = feature is absent

The LLM judges leaf features from the structured task information produced in
step 5. Parent features are induced by a deterministic hierarchy so the final
embedding stays internally consistent.
"""

from __future__ import annotations

import csv
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


REPRESENTATIONAL_FEATURES = [
    "phonological_processing",
    "prosody_tone_stress_processing",
    "orthographic_processing",
    "morphological_processing",
    "syntactic_processing",
    "semantic_processing",
    "discourse_pragmatic_processing",
    "lexical_access",
    "letter_sound_mapping",
    "word_form_processing",
    "sentence_composition",
    "semantic_anomaly_plausibility",
    "bilingual_multilingual_processing",
]

PROCESSING_FEATURES = [
    "comprehension",
    "production",
    "reading",
    "speech_perception_listening",
    "repetition",
    "naming",
    "generation_fluency",
    "judgment_decision",
    "recognition_identification",
    "discrimination",
    "lexical_decision",
    "translation_or_language_switching",
]

STIMULUS_FEATURES = [
    "visual_text_stimulus",
    "auditory_speech_stimulus",
    "picture_object_stimulus",
    "real_word_material",
    "pseudoword_nonword_material",
    "false_font_symbol_material",
    "artificial_grammar_material",
    "letter_grapheme_unit",
    "syllable_phoneme_unit",
    "word_unit",
    "sentence_unit",
    "discourse_unit",
    "semantic_relatedness_manipulation",
    "phonological_similarity_rhyme_manipulation",
    "word_frequency_length_manipulation",
]

RESPONSE_DESIGN_FEATURES = [
    "spoken_response",
    "button_press_response",
    "written_typed_response",
    "covert_response",
    "time_pressure_rt_measure",
    "scanner_task",
    "baseline_or_control_task",
    "nonlinguistic_lowlevel_control",
]

FEATURE_GROUPS: Dict[str, List[str]] = {
    "representational": REPRESENTATIONAL_FEATURES,
    "processing": PROCESSING_FEATURES,
    "stimulus_material": STIMULUS_FEATURES,
    "response_design": RESPONSE_DESIGN_FEATURES,
}

LEAF_FEATURES: List[str] = [
    feature for group in FEATURE_GROUPS.values() for feature in group
]

FEATURE_GRAPH: Dict[str, List[str]] = {feature: [] for feature in LEAF_FEATURES}


FEATURE_DESCRIPTIONS: Dict[str, str] = {
    "phonological_processing": "0 when sounds/phonemes/syllables/rhyme/pronunciation are central; 1 for purely semantic or visual control tasks.",
    "prosody_tone_stress_processing": "0 for lexical tone, accent, stress, intonation, rhythm, or prosody; 1 for ordinary word reading without prosodic manipulation.",
    "orthographic_processing": "0 for written letters/words/spelling/visual word forms; 1 for purely auditory speech.",
    "morphological_processing": "0 for inflection, derivation, compounds, morphemes, or word-structure judgments; 1 for simple whole-word recognition.",
    "syntactic_processing": "0 for grammar, word order, sentence structure, dependency, or grammaticality; 1 for isolated words/pictures.",
    "semantic_processing": "0 for meaning, lexical semantics, categories, relatedness, plausibility; 1 for low-level perceptual controls.",
    "discourse_pragmatic_processing": "0 for stories, discourse, narrative, anaphora, pragmatics; 1 for single words or sentences without discourse relation.",
    "lexical_access": "0 when retrieving/recognizing real word identities or names is central; 1 for false-font or tone-only controls.",
    "letter_sound_mapping": "0 for reading aloud, spelling, grapheme-phoneme conversion; 1 for word meaning decisions without letter-sound mapping.",
    "word_form_processing": "0 for orthographic or phonological word-form analysis; 1 for picture-only semantic tasks.",
    "sentence_composition": "0 for combining word meanings or sentence-level interpretation; 1 for isolated word tasks.",
    "semantic_anomaly_plausibility": "0 for anomalous/plausible/incongruent semantic decisions; 1 for ordinary semantic categorization.",
    "bilingual_multilingual_processing": "0 for L2, bilingual, translation, or multiple named languages; 1 for one-language tasks.",
    "comprehension": "0 when understanding perceived language/stimuli is the participant goal; 1 for pure production without comprehension demand.",
    "production": "0 when participants generate/speak/write/name/repeat language; 1 for passive listening or button judgments only.",
    "reading": "0 for reading text aloud/silently or visual word reading; 1 for auditory word repetition.",
    "speech_perception_listening": "0 for auditory spoken-language perception; 1 for visual text or picture-only tasks.",
    "repetition": "0 when participants repeat heard/seen items; 1 for naming, reading, or judgments.",
    "naming": "0 for picture/object/sound naming; 1 for reading words aloud or repeating words.",
    "generation_fluency": "0 for verbal fluency or open word generation; 1 for fixed-response recognition tasks.",
    "judgment_decision": "0 for explicit yes/no, same/different, grammatical, semantic, or lexical decisions; 1 for simple reading/repetition.",
    "recognition_identification": "0 for identifying/recognizing a presented item or category; 1 for unconstrained generation.",
    "discrimination": "0 for distinguishing alternatives such as vowel length, same/different, word vs nonword; 1 for simple naming.",
    "lexical_decision": "0 only for deciding real word vs pseudoword/nonword; 1 for word recognition without this decision.",
    "translation_or_language_switching": "0 for translation or switching between languages; 1 for monolingual tasks.",
    "visual_text_stimulus": "0 for visually presented letters/words/sentences; 1 for auditory-only or picture-only tasks.",
    "auditory_speech_stimulus": "0 for spoken words/sentences/syllables; 1 for visual text-only tasks.",
    "picture_object_stimulus": "0 for pictures/photos/objects as stimuli; 1 for words/sounds only.",
    "real_word_material": "0 when real words are central materials; 1 for only pseudowords, symbols, tones, or pictures.",
    "pseudoword_nonword_material": "0 for pseudowords/nonwords/nonsense syllables; 1 for only real words.",
    "false_font_symbol_material": "0 for false fonts/symbol strings/non-letter controls; 1 otherwise.",
    "artificial_grammar_material": "0 for artificial grammar/rule strings; 1 otherwise.",
    "letter_grapheme_unit": "0 for letters/graphemes/characters as units; 1 for whole words/pictures only.",
    "syllable_phoneme_unit": "0 for syllables/phonemes/vowels/consonants; 1 for whole words/pictures only.",
    "word_unit": "0 for word-level stimuli or responses; 1 for sentence-only, sound-only, or picture-only tasks.",
    "sentence_unit": "0 for sentence-level stimuli; 1 for isolated word/letter/picture tasks.",
    "discourse_unit": "0 for paragraph/story/narrative/discourse stimuli; 1 otherwise.",
    "semantic_relatedness_manipulation": "0 when semantic relatedness/category/thematic relation is manipulated; 1 for ordinary meaningful words with no relation manipulation.",
    "phonological_similarity_rhyme_manipulation": "0 for rhyme/phonological similarity/sound contrast manipulation; 1 otherwise.",
    "word_frequency_length_manipulation": "0 for explicit word frequency, familiarity, length, syllable-count, or complexity manipulation; 1 otherwise.",
    "spoken_response": "0 for overt spoken response; 1 for button-only or passive tasks.",
    "button_press_response": "0 for button/key/manual choice response; 1 otherwise.",
    "written_typed_response": "0 for written or typed language response; 1 otherwise.",
    "covert_response": "0 for silent/covert/mental response; 1 for overt or unspecified response.",
    "time_pressure_rt_measure": "0 for speeded response, deadline, or RT as key measure; 1 otherwise.",
    "scanner_task": "0 when performed during fMRI/PET/MEG/EEG scanning or imaging acquisition; 1 otherwise.",
    "baseline_or_control_task": "0 if this row itself is a baseline/control task; 1 for experimental language tasks.",
    "nonlinguistic_lowlevel_control": "0 if this row itself uses nonlinguistic low-level control material such as fixation, tones, symbols, false fonts; 1 otherwise.",
}


@dataclass
class LanguageTaskFeatureResult:
    """One task's feature embedding and extraction metadata."""

    pmid: str
    task_index: str
    task_name: str
    grouped_features: Dict[str, Dict[str, int]]
    feature_embedding: Dict[str, int]
    metadata: Dict[str, Any]


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def write_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, ensure_ascii=False, indent=2)


def safe_json_loads(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    text = str(value or "").strip()
    if not text:
        return ""
    if text[0:1] in {"[", "{"}:
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def compact_task_record(task_record: Dict[str, Any]) -> Dict[str, Any]:
    """Parse JSON-like cells and keep only evidence-rich task fields."""

    keep_fields = [
        "PMID",
        "task_index",
        "name",
        "raw_name",
        "canonical_task_family",
        "type",
        "is_language_task",
        "is_control_task",
        "description",
        "clue_sentences",
        "purpose",
        "measured_constructs",
        "Task_property_on_Production_or_Comprehension",
        "Task_property_on_attention_engagement",
        "Task_property_on_Linguistic_Domain",
        "Task_property_on_Processing_Type",
        "Task_property_on_Stimulus_modality",
        "linguistic_properties",
        "stimulus_properties",
        "response_properties",
        "cognitive_demands",
        "downstream_feature_hints",
        "verbal_fluency",
        "evidence",
        "Language_use",
    ]
    compact: Dict[str, Any] = {}
    for field in keep_fields:
        if field in task_record:
            compact[field] = safe_json_loads(task_record.get(field))
    design = safe_json_loads(task_record.get("experimental_design"))
    if isinstance(design, dict):
        compact["experimental_design"] = {
            key: design.get(key, "")
            for key in [
                "scanner_execution",
                "imaging_modalities",
                "design_type",
                "timing",
                "task_instructions",
                "trial_structure",
            ]
        }
    return compact


def extract_json_object(text: str) -> Dict[str, Any]:
    content = str(text or "").strip()
    if not content:
        raise ValueError("Empty model response.")
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", content, flags=re.DOTALL)
    if not match:
        raise ValueError("Could not parse a JSON object from model response.")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("Parsed response is not a JSON object.")
    return parsed


def normalize_binary(value: Any) -> int:
    if isinstance(value, bool):
        return 0 if value else 1
    try:
        return 0 if int(value) == 0 else 1
    except Exception:
        text = str(value or "").strip().lower()
        if text in {"0", "present", "yes", "true", "y"}:
            return 0
        return 1


def sanitize_filename(value: str, max_len: int = 110) -> str:
    text = re.sub(r"[^a-zA-Z0-9_\-\s\u4e00-\u9fff]", "", str(value or "task"))
    text = re.sub(r"\s+", "_", text.strip())
    return (text or "task")[:max_len]


class ExtractLanguage_taskFeature:
    """Extract binary language-task feature embeddings with an LLM."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        client_type: str = "deepseek",
        model_name: str = "deepseek-v4-flash",
        max_retries: int = 2,
        max_tokens: int = 8192,
        request_interval_seconds: float = 0.0,
    ) -> None:
        self.config = config or {}
        self.client_type = client_type
        self.model_name = model_name
        self.max_retries = int(max_retries)
        self.max_tokens = int(max_tokens)
        self.request_interval_seconds = float(request_interval_seconds)
        self.feature_groups = FEATURE_GROUPS
        self.leaf_features = list(LEAF_FEATURES)
        self.feature_graph = dict(FEATURE_GRAPH)
        self._client_manager = LLMClientManager(self.config)
        self._client: Optional[Any] = None
        self._resolved_model_name: Optional[str] = None
        self._last_request_finished_at = 0.0
        self.call_count = 0

    @classmethod
    def deepseek(
        cls,
        api_key: str,
        model_name: str = "deepseek-v4-flash",
        **kwargs: Any,
    ) -> "ExtractLanguage_taskFeature":
        return cls(
            config={"deepseek_api_key": api_key},
            client_type="deepseek",
            model_name=model_name,
            **kwargs,
        )

    def get_client(self) -> Tuple[Any, str]:
        if self._client is None:
            self._client, self._resolved_model_name = self._client_manager.get_client(
                client_type=self.client_type,
                model_name=self.model_name,
            )
        return self._client, str(self._resolved_model_name or self.model_name)

    def _wait_for_slot(self) -> None:
        remaining = self.request_interval_seconds - (
            time.monotonic() - self._last_request_finished_at
        )
        if remaining > 0:
            time.sleep(remaining)

    def build_messages(self, task_record: Dict[str, Any]) -> List[Dict[str, str]]:
        schema = {feature: 1 for feature in self.leaf_features}
        feature_notes = "\n".join(
            f"- {key}: {value}" for key, value in sorted(FEATURE_DESCRIPTIONS.items())
        )
        system_prompt = f"""
You are a precise language-neuroscience task feature annotator.
Convert the provided structured task information into a binary feature
embedding.

CODING
- 0 means PRESENT.
- 1 means ABSENT.
- Mark PRESENT only when the feature is a central task property, an explicit
  stimulus/response/design factor, or a directly measured linguistic/cognitive
  process.
- Annotate only the CURRENT task described by name/description/stimuli/response.
  Ignore comparison conditions, baseline conditions, contrast names, and other
  tasks mentioned only to describe the experimental design.
- Incidental consequences of any language task do not count. For example, do
  not mark syntax present for an isolated word-reading task.
- If evidence is missing or uncertain, choose 1.
- Non-language control tasks may have control/design features, but language
  representational features should remain absent unless linguistic material is
  explicitly used.
- Preserve the exact feature keys. Values must be integers 0 or 1.

IMPORTANT FEATURE NOTES
{feature_notes}

FEATURE GROUPS
{json.dumps(self.feature_groups, ensure_ascii=False, indent=2)}

Return exactly one JSON object:
{{
  "features": {json.dumps(schema, ensure_ascii=False)},
  "uncertain_features": ["feature_name"],
  "evidence": {{"feature_name": "short evidence from task info"}}
}}
""".strip()
        user_payload = compact_task_record(task_record)
        return [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(user_payload, ensure_ascii=False),
            },
        ]

    def call_model(
        self,
        messages: Sequence[Dict[str, str]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        client, model_name = self.get_client()
        last_error = ""
        raw_preview = ""
        for retry_index in range(self.max_retries + 1):
            try:
                self._wait_for_slot()
                self.call_count += 1
                kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": list(messages),
                    "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "response_format": {"type": "json_object"},
                    "stream": False,
                }
                if self.client_type == "deepseek":
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                response = client.chat.completions.create(**kwargs)
                self._last_request_finished_at = time.monotonic()
                raw_text = response.choices[0].message.content or ""
                raw_preview = raw_text[:1500]
                return extract_json_object(raw_text), {
                    "model": model_name,
                    "client_type": self.client_type,
                    "thinking": "disabled" if self.client_type == "deepseek" else "",
                    "retry_count": retry_index,
                    "error": "",
                    "raw_generation_preview": raw_preview,
                }
            except Exception as exc:
                self._last_request_finished_at = time.monotonic()
                last_error = f"{type(exc).__name__}: {exc}"
                if retry_index >= self.max_retries:
                    break
                lower = last_error.lower()
                rate_limited = "429" in lower or "rate limit" in lower
                delay = min((30 if rate_limited else 3) * (2**retry_index), 180)
                delay += random.uniform(0.0, 1.0)
                print(
                    f"  Feature extraction failed ({last_error}); retry "
                    f"{retry_index + 1}/{self.max_retries} in {delay:.1f}s"
                )
                time.sleep(delay)
        return {}, {
            "model": model_name,
            "client_type": self.client_type,
            "thinking": "disabled" if self.client_type == "deepseek" else "",
            "retry_count": self.max_retries,
            "error": last_error,
            "raw_generation_preview": raw_preview,
        }

    def normalize_leaf_features(self, raw: Dict[str, Any]) -> Dict[str, int]:
        feature_source = raw.get("features", raw)
        if not isinstance(feature_source, dict):
            feature_source = {}
        return {
            feature: normalize_binary(feature_source.get(feature, 1))
            for feature in self.leaf_features
        }

    def apply_task_info_rules(
        self,
        features: Dict[str, int],
        task_record: Dict[str, Any],
    ) -> Dict[str, int]:
        """Use explicit structured fields to stabilize common feature labels."""

        compact = compact_task_record(task_record)
        stimulus = compact.get("stimulus_properties")
        response = compact.get("response_properties")
        linguistic = compact.get("linguistic_properties")
        verbal_fluency = compact.get("verbal_fluency")
        design = compact.get("experimental_design")
        task_specific = {
            "name": compact.get("name", ""),
            "raw_name": compact.get("raw_name", ""),
            "canonical_task_family": compact.get("canonical_task_family", ""),
            "type": compact.get("type", ""),
            "description": compact.get("description", ""),
            "purpose": compact.get("purpose", ""),
            "measured_constructs": compact.get("measured_constructs", ""),
            "linguistic_properties": linguistic,
            "stimulus_properties": stimulus,
            "response_properties": response,
            "verbal_fluency": verbal_fluency,
            "evidence": compact.get("evidence", ""),
            "downstream_feature_hints": compact.get("downstream_feature_hints", ""),
            "Language_use": compact.get("Language_use", ""),
        }
        text = json.dumps(task_specific, ensure_ascii=False).lower()
        stimulus_text = json.dumps(stimulus, ensure_ascii=False).lower()
        response_text = json.dumps(response, ensure_ascii=False).lower()
        design_text = json.dumps(design, ensure_ascii=False).lower()
        name = str(compact.get("name", "")).lower()
        task_type = str(compact.get("type", "")).lower()

        def present(key: str) -> None:
            if key in features:
                features[key] = 0

        def absent(key: str) -> None:
            if key in features:
                features[key] = 1

        keyword_map = {
            "real_word_material": ["real word", "real-word", '"real words"'],
            "pseudoword_nonword_material": ["pseudoword", "pseudo-word", "nonword", "non-word"],
            "false_font_symbol_material": ["false font", "symbol string", "non-letter"],
            "artificial_grammar_material": ["artificial grammar"],
            "word_frequency_length_manipulation": ["word frequency", "frequency", "word length", "length", "imageability", "concreteness"],
            "semantic_relatedness_manipulation": ["semantic prim", "semantic related", "category decision", "thematic"],
            "phonological_similarity_rhyme_manipulation": ["phonological prim", "rhyme", "rhyming"],
            "lexical_decision": ["lexical decision"],
            "generation_fluency": ["verbal fluency", "generate as many", "word generation"],
            "translation_or_language_switching": ["language switching", "switching between", "translation", "translate"],
            "bilingual_multilingual_processing": ["bilingual", "second language", " l2 ", "multilingual", "translation"],
        }
        for key, needles in keyword_map.items():
            if any(needle in text for needle in needles):
                present(key)

        response_mode_text = ""
        overt_value = ""
        if isinstance(response, dict):
            response_mode_text = json.dumps(
                response.get("response_mode", ""), ensure_ascii=False
            ).lower()
            overt_value = str(response.get("overt_or_covert", "")).lower()
        else:
            response_mode_text = response_text
        if any(word in response_text for word in ["spoken", "speak", "say aloud", "read aloud", "repeat aloud", "voice"]):
            present("spoken_response")
        if any(word in response_text for word in ["button", "keypress", "key press"]):
            present("button_press_response")
        if any(word in response_text for word in ["written", "typed", "write", "typing"]):
            present("written_typed_response")
        if (
            "covert" in response_mode_text
            or overt_value == "covert"
            or any(word in response_text for word in ["silently", "mental"])
        ):
            present("covert_response")
        elif overt_value == "overt" or features.get("spoken_response", 1) == 0:
            absent("covert_response")
        if any(word in design_text for word in ["fmri", "pet", "scanner", "imaging", "true"]):
            present("scanner_task")
        if any(word in text for word in ["speeded", "reaction time", "as quickly"]):
            present("time_pressure_rt_measure")

        if "control" in task_type or "baseline" in name:
            present("baseline_or_control_task")
        else:
            absent("baseline_or_control_task")
        if (
            ("control" in task_type or "baseline" in name or "control" in name)
            and ("false font" in stimulus_text or "non-linguistic" in stimulus_text or "nonlinguistic" in stimulus_text)
        ):
            present("nonlinguistic_lowlevel_control")
        else:
            absent("nonlinguistic_lowlevel_control")
        if (
            "visual" in stimulus_text
            and any(word in stimulus_text for word in ["text", "word", "letter", "grapheme", "string"])
        ):
            present("visual_text_stimulus")
        elif "modalities" in stimulus_text and "visual" not in stimulus_text:
            absent("visual_text_stimulus")
        if any(word in stimulus_text for word in ["auditory", "heard", "spoken", "speech", "natural speech"]):
            present("auditory_speech_stimulus")
        elif "modalities" in stimulus_text and "visual" in stimulus_text:
            absent("auditory_speech_stimulus")
        if any(word in stimulus_text for word in ["picture", "object", "photograph"]):
            present("picture_object_stimulus")
        else:
            absent("picture_object_stimulus")
        if "false font" not in stimulus_text and "non-letter" not in stimulus_text:
            absent("false_font_symbol_material")
        if "lexical_status" in stimulus_text:
            if "real words" not in stimulus_text and "real word" not in stimulus_text:
                absent("real_word_material")
            if not any(word in stimulus_text for word in ["pseudoword", "pseudo-word", "nonword", "non-word"]):
                absent("pseudoword_nonword_material")
        if "read aloud" in text or "reading task" in name:
            present("reading")
            if features.get("spoken_response", 1) == 0:
                present("letter_sound_mapping")
        if not any(
            phrase in text
            for phrase in [
                "naming task",
                "picture naming",
                "object naming",
                "sound naming",
                "name pictures",
                "name objects",
                "naming and pointing",
            ]
        ):
            absent("naming")
        units_text = stimulus_text + " " + json.dumps(linguistic, ensure_ascii=False).lower()
        if any(word in units_text for word in ["letter", "grapheme", "character"]):
            present("letter_grapheme_unit")
        if any(word in units_text for word in ["syllable", "phoneme", "vowel", "consonant"]):
            present("syllable_phoneme_unit")
        if "word" in units_text:
            present("word_unit")
        if "sentence" in units_text:
            present("sentence_unit")
        if any(word in units_text for word in ["discourse", "paragraph", "narrative", "story"]):
            present("discourse_unit")
        return features

    def upward_close(self, leaf_assignments: Dict[str, int]) -> Dict[str, int]:
        all_nodes = set(self.feature_graph.keys()) | set(leaf_assignments.keys())
        assignments = {feature: 1 for feature in all_nodes}
        for key, value in leaf_assignments.items():
            assignments[key] = normalize_binary(value)
        changed = True
        while changed:
            changed = False
            for node, parents in self.feature_graph.items():
                if assignments.get(node, 1) == 0:
                    for parent in parents:
                        if assignments.get(parent, 1) != 0:
                            assignments[parent] = 0
                            changed = True
        return assignments

    def enforce_consistency(self, assignments: Dict[str, int]) -> Dict[str, int]:
        def present(key: str) -> None:
            if key in assignments:
                assignments[key] = 0

        if assignments.get("pseudoword_nonword_material", 1) == 0:
            present("phonological_processing")
            present("word_form_processing")
        if assignments.get("real_word_material", 1) == 0:
            present("lexical_access")
            present("word_form_processing")
        if assignments.get("lexical_decision", 1) == 0:
            present("recognition_identification")
            present("judgment_decision")
            present("discrimination")
        if assignments.get("sentence_unit", 1) == 0:
            present("syntactic_processing")
            present("semantic_processing")
            present("sentence_composition")
        if assignments.get("discourse_unit", 1) == 0:
            present("discourse_pragmatic_processing")
        if assignments.get("semantic_anomaly_plausibility", 1) == 0:
            present("semantic_processing")
            present("judgment_decision")
        if assignments.get("reading", 1) == 0:
            present("orthographic_processing")
            present("visual_text_stimulus")
        if assignments.get("naming", 1) == 0:
            present("production")
        if assignments.get("generation_fluency", 1) == 0:
            present("production")
            present("lexical_access")
        if assignments.get("translation_or_language_switching", 1) == 0:
            present("bilingual_multilingual_processing")
        if assignments.get("spoken_response", 1) == 0:
            present("production")
        return self.upward_close(assignments)

    def group_features(self, assignments: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        return {
            group_name: {feature: assignments.get(feature, 1) for feature in features}
            for group_name, features in self.feature_groups.items()
        }

    def extract_task_features(
        self,
        task_record: Dict[str, Any],
    ) -> LanguageTaskFeatureResult:
        pmid = str(task_record.get("PMID", "")).strip()
        task_index = str(task_record.get("task_index", "")).strip()
        task_name = str(task_record.get("name", "") or "task").strip()
        metadata = {
            "pmid": pmid,
            "task_index": task_index,
            "task_name": task_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "pending",
            "model": self.model_name,
            "client_type": self.client_type,
            "error": "",
        }
        raw, call_meta = self.call_model(self.build_messages(task_record))
        metadata.update(call_meta)
        metadata["status"] = "success" if raw else "error"
        if not raw:
            leaf_features = {feature: 1 for feature in self.leaf_features}
            metadata["error"] = call_meta.get("error", "Model returned no JSON.")
        else:
            leaf_features = self.normalize_leaf_features(raw)
        leaf_features = self.apply_task_info_rules(leaf_features, task_record)
        closed = self.upward_close(leaf_features)
        consistent = self.enforce_consistency(closed)
        grouped = self.group_features(consistent)
        output_embedding = {
            key: consistent.get(key, 1) for key in self.leaf_features
        }
        metadata["present_feature_count"] = sum(
            1 for value in output_embedding.values() if value == 0
        )
        return LanguageTaskFeatureResult(
            pmid=pmid,
            task_index=task_index,
            task_name=task_name,
            grouped_features=grouped,
            feature_embedding=output_embedding,
            metadata=metadata,
        )

    def extract_task_to_file(
        self,
        task_record: Dict[str, Any],
        output_json_dir: Path,
        overwrite: bool = False,
    ) -> LanguageTaskFeatureResult:
        pmid = str(task_record.get("PMID", "")).strip()
        task_index = str(task_record.get("task_index", "")).strip()
        task_name = str(task_record.get("name", "") or "task")
        filename = f"paper_{pmid}_task_{task_index}_{sanitize_filename(task_name)}_features.json"
        output_path = output_json_dir / filename
        if output_path.exists() and not overwrite:
            existing = read_json(output_path)
            metadata = existing.get("metadata", {})
            if metadata.get("status") == "success":
                return LanguageTaskFeatureResult(
                    pmid=str(existing.get("pmid", pmid)),
                    task_index=str(existing.get("task_index", task_index)),
                    task_name=str(existing.get("task_name", task_name)),
                    grouped_features=existing.get("grouped_features", {}),
                    feature_embedding=existing.get("feature_embedding", {}),
                    metadata=metadata,
                )
        result = self.extract_task_features(task_record)
        write_json(
            {
                "pmid": result.pmid,
                "task_index": result.task_index,
                "task_name": result.task_name,
                "grouped_features": result.grouped_features,
                "feature_embedding": result.feature_embedding,
                "metadata": result.metadata,
                "source_task_record": compact_task_record(task_record),
            },
            output_path,
        )
        return result


def load_task_rows(path: Path, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        for row in reader:
            rows.append(dict(row))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_feature_tables(
    results: Iterable[LanguageTaskFeatureResult],
    output_dir: Path,
) -> None:
    result_list = list(results)
    feature_fields = sorted({key for result in result_list for key in result.feature_embedding})
    table_path = output_dir / "language_task_feature_embedding_table.csv"
    summary_path = output_dir / "language_task_feature_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with table_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        fields = ["PMID", "task_index", "task_name", "status", "model", *feature_fields]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for result in result_list:
            row = {
                "PMID": result.pmid,
                "task_index": result.task_index,
                "task_name": result.task_name,
                "status": result.metadata.get("status", ""),
                "model": result.metadata.get("model", ""),
            }
            row.update(result.feature_embedding)
            writer.writerow({field: row.get(field, "") for field in fields})
    with summary_path.open("w", encoding="utf-8-sig", newline="") as file_obj:
        fields = [
            "PMID",
            "task_index",
            "task_name",
            "status",
            "present_feature_count",
            "error",
        ]
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for result in result_list:
            writer.writerow(
                {
                    "PMID": result.pmid,
                    "task_index": result.task_index,
                    "task_name": result.task_name,
                    "status": result.metadata.get("status", ""),
                    "present_feature_count": result.metadata.get(
                        "present_feature_count", ""
                    ),
                    "error": result.metadata.get("error", ""),
                }
            )
