"""
Final supplementary segmentation for difficult Step3 Markdown documents.

The final segmenter is intended for papers that remain missing Methods and/or
Results after the original and supplementary segmentation passes. It combines:

1. Conservative heading/paragraph rules for old OCR Markdown.
2. Semantic fallbacks for unlabeled short reports and case studies.
3. Optional LLM extraction for mixed Methods/Results writing.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from openai import OpenAI

try:
    from core.llm_client import LLMClientManager
except Exception:
    LLMClientManager = None


TARGET_CATEGORIES = [
    "Title",
    "Author",
    "Keywords",
    "Abstract",
    "Introduction",
    "Methods",
    "Results",
    "Discussion",
    "References",
    "Acknowledgements",
    "Other",
]

METHOD_HEADINGS = {
    "methods",
    "method",
    "materials and methods",
    "materials & methods",
    "subjects and methods",
    "patients and methods",
    "participants and methods",
    "data acquisition",
    "data analysis",
    "statistical analysis",
    "statistical analyses",
    "experimental procedure",
    "experimental procedures",
    "experimental design",
    "procedure",
    "procedures",
    "case description",
    "case presentation",
    "case report",
    "materials",
    "stimulation mapping procedure",
    "scan eeg and related measures",
    "standardized assessment",
}

RESULT_HEADINGS = {
    "results",
    "result",
    "findings",
    "visualization of language data",
    "visualizing deep structures",
    "projecting fmri data onto the surface of the brain",
    "activation",
    "brain activation",
    "clinical findings",
    "outcome",
    "outcomes",
}

DISCUSSION_HEADINGS = {
    "discussion",
    "conclusion",
    "conclusions",
    "comment",
    "summary",
    "limitations",
}

REFERENCE_HEADINGS = {"references", "references and notes", "bibliography"}

METHOD_MARKERS = [
    "subjects were",
    "participants were",
    "patients were",
    "volunteers",
    "written informed consent",
    "protocol was approved",
    "we obtained",
    "we acquired",
    "we used",
    "we performed",
    "was performed",
    "were performed",
    "the task was",
    "stimuli were",
    "data acquisition",
    "statistical analysis",
    "mri scans",
    "fmri",
    "pet",
    "meg",
    "eeg",
    "spm",
    "regression model",
    "linear regression",
    "control condition",
    "experimental condition",
]

RESULT_MARKERS = [
    "we found",
    "we observed",
    "revealed significant",
    "showed significant",
    "activated",
    "activation",
    "compared with",
    "significantly",
    "p <",
    "p<",
    "t =",
    "z =",
    "correlation",
    "results showed",
    "was greater",
    "were greater",
]

DISCUSSION_MARKERS = [
    "these findings suggest",
    "our findings suggest",
    "in conclusion",
    "we conclude",
    "this suggests",
    "may reflect",
    "consistent with",
    "further studies",
    "future",
]


@dataclass
class FinalSegResult:
    structured: Dict[str, Any]
    metadata: Dict[str, Any]


def normalize_heading(text: str) -> str:
    text = re.sub(r"^#+\s*", "", text or "").strip()
    text = re.sub(r"^[\d.\-–—\s]+", "", text)
    text = re.sub(r"[^\w\s&]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_markdown_text(text: str) -> str:
    text = re.sub(r"\*\*Total Pages:\*\*.*", "", text)
    text = re.sub(r"\*\*Status:\*\*.*", "", text)
    text = re.sub(r"^={10,}$|^-{10,}$", "", text, flags=re.M)
    text = re.sub(r"<!--\s*Page\s+\d+\s*-->", "", text, flags=re.I)
    text = re.sub(r"^##\s+Page\s+\d+\s*$", "", text, flags=re.I | re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_tables(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    tables: List[Dict[str, Any]] = []

    def replace_html(match: re.Match) -> str:
        table_id = f"table_{len(tables) + 1}"
        tables.append({"id": table_id, "content": match.group(0), "table_type": "html"})
        return f"\n[TABLE: {table_id}]\n"

    text = re.sub(r"<table\b.*?</table>", replace_html, text, flags=re.I | re.S)
    return text, tables


def split_into_blocks(text: str) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    current_heading = "Document_Root"
    current_level = 0
    current_lines: List[str] = []
    current_start = 1

    def flush(end_line: int) -> None:
        nonlocal current_lines, current_start
        content = "\n".join(current_lines).strip()
        if content:
            blocks.append(
                {
                    "heading": current_heading,
                    "heading_level": current_level,
                    "content": content,
                    "line_start": current_start,
                    "line_end": end_line,
                    "category": "Other",
                }
            )
        current_lines = []

    lines = text.splitlines()
    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        heading_match = re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if heading_match:
            flush(idx - 1)
            current_heading = heading_match.group(2).strip()
            current_level = len(heading_match.group(1))
            current_start = idx + 1
            continue

        if not stripped:
            if current_lines:
                current_lines.append("")
            continue

        if not current_lines:
            current_start = idx
        current_lines.append(raw)

    flush(len(lines))
    return blocks


def category_from_heading(heading: str) -> str:
    normalized = normalize_heading(heading)
    if normalized in METHOD_HEADINGS:
        return "Methods"
    if normalized in RESULT_HEADINGS:
        return "Results"
    if normalized in DISCUSSION_HEADINGS:
        return "Discussion"
    if normalized in REFERENCE_HEADINGS or normalized.startswith("references"):
        return "References"
    if "abstract" == normalized:
        return "Abstract"
    if "keyword" in normalized:
        return "Keywords"
    if "introduction" in normalized or normalized == "background":
        return "Introduction"
    if "acknowledg" in normalized or "funding" in normalized:
        return "Acknowledgements"
    return "Other"


def marker_score(text: str, markers: Iterable[str]) -> int:
    lower = text.lower()
    return sum(1 for marker in markers if marker in lower)


def has_meaningful_section(value: Any, min_chars: int = 80) -> bool:
    text = str(value or "")
    text = re.sub(r"^\s*#{1,6}\s+.*$", "", text, flags=re.M)
    text = re.sub(r"\[TABLE:[^\]]+\]", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    return len(text) >= min_chars


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = str(text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(cleaned[start : end + 1])
            return data if isinstance(data, dict) else {}
        raise


def classify_blocks_with_rules(blocks: List[Dict[str, Any]]) -> None:
    for idx, block in enumerate(blocks):
        category = category_from_heading(block.get("heading", ""))
        if category == "Other" and idx == 0:
            category = "Title"
        block["category"] = category

    for block in blocks:
        if block["category"] != "Other":
            continue
        content = block.get("content", "")
        scores = {
            "Methods": marker_score(content, METHOD_MARKERS),
            "Results": marker_score(content, RESULT_MARKERS),
            "Discussion": marker_score(content, DISCUSSION_MARKERS),
        }
        best = max(scores, key=scores.get)
        if scores[best] >= 2:
            block["category"] = best

    # Preserve natural order for unlabeled short reports.
    seen_method = any(block["category"] == "Methods" for block in blocks)
    seen_result = any(block["category"] == "Results" for block in blocks)
    if seen_method and not seen_result:
        for block in blocks:
            if block["category"] == "Other" and marker_score(block.get("content", ""), RESULT_MARKERS) >= 1:
                block["category"] = "Results"
    if seen_result and not seen_method:
        for block in blocks:
            if block["category"] == "Other" and marker_score(block.get("content", ""), METHOD_MARKERS) >= 1:
                block["category"] = "Methods"


def formatted_section(blocks: List[Dict[str, Any]]) -> str:
    pieces: List[str] = []
    emitted = set()
    for block in blocks:
        heading = block.get("heading", "Document_Root")
        content = block.get("content", "").strip()
        key = (heading, block.get("heading_level", 0))
        if heading != "Document_Root" and key not in emitted:
            emitted.add(key)
            pieces.append(f"### {heading}\n{content}".strip())
        elif content:
            pieces.append(content)
    return "\n\n".join(piece for piece in pieces if piece).strip()


def build_structured(blocks: List[Dict[str, Any]], tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_cat: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for block in blocks:
        by_cat[block.get("category", "Other")].append(block)
    structured = {category: formatted_section(by_cat.get(category, [])) for category in TARGET_CATEGORIES}
    structured["Tables"] = tables
    structured["Methods_Hierarchy"] = [
        {
            "heading": block.get("heading", ""),
            "level": block.get("heading_level", 0),
            "content": block.get("content", ""),
            "children": [],
        }
        for block in by_cat.get("Methods", [])
    ]
    return structured


def body_without_references(text: str) -> str:
    match = re.search(r"^#{1,6}\s+(references|references and notes|bibliography)\b", text, flags=re.I | re.M)
    if match:
        return text[: match.start()].strip()
    return text


def heuristic_method_fallback(text: str) -> str:
    body = body_without_references(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 80]
    method_like = [
        p for p in paragraphs
        if marker_score(p, METHOD_MARKERS) >= 1
    ]
    if method_like:
        return "\n\n".join(method_like[:4]).strip()

    # If no explicit method prose exists, use the central empirical procedure
    # portion rather than leaving Methods empty.
    if len(paragraphs) >= 4:
        return "\n\n".join(paragraphs[2:4]).strip()
    if paragraphs:
        return paragraphs[min(1, len(paragraphs) - 1)]
    return body[:2500].strip()


def heuristic_results_fallback(text: str) -> str:
    body = body_without_references(text)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if len(p.strip()) > 80]
    result_like = [
        p for p in paragraphs
        if marker_score(p, RESULT_MARKERS) >= 1
    ]
    if result_like:
        return "\n\n".join(result_like[:5]).strip()
    if len(paragraphs) >= 5:
        return "\n\n".join(paragraphs[3:5]).strip()
    return ""


class FinalSeg:
    """Final difficult-case segmenter with optional LLM extraction."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        preferred_client: str = "deepseek",
        max_chars: int = 60000,
        sleep_seconds: float = 0.5,
    ) -> None:
        self.config = config or {}
        self.preferred_client = preferred_client
        self.max_chars = max_chars
        self.sleep_seconds = sleep_seconds
        self.llm_calls = 0

    def get_client(self, client_type: str) -> Tuple[OpenAI, str]:
        if LLMClientManager is not None:
            manager = LLMClientManager(self.config)
            configured_model = self.config.get(f"{client_type}_model_name")
            return manager.get_client(
                client_type=client_type,
                model_name=configured_model,
            )

        if client_type == "mimo":
            return (
                OpenAI(
                    api_key=self.config["mimo_api_key"],
                    base_url="https://token-plan-cn.xiaomimimo.com/v1",
                ),
                self.config.get("mimo_model_name", "mimo-v2.5-pro"),
            )
        return (
            OpenAI(
                api_key=self.config["deepseek_api_key"],
                base_url="https://api.deepseek.com/v1",
            ),
            self.config.get("deepseek_model_name", "deepseek-chat"),
        )

    def should_use_llm(self, review_record: Optional[Dict[str, Any]], structured: Dict[str, Any]) -> bool:
        if not has_meaningful_section(structured.get("Methods")) or not has_meaningful_section(
            structured.get("Results")
        ):
            return True
        if not review_record:
            return False
        note = str(review_record.get("note", "")).lower()
        return "方法结果混合" in note or "mixed" in note

    def llm_extract_sections(
        self,
        markdown_text: str,
        review_record: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        prompt_text = markdown_text[: self.max_chars]
        issue_note = json.dumps(review_record or {}, ensure_ascii=False)
        system_prompt = """You extract Methods and Results text from difficult OCR Markdown academic papers.

Return valid JSON only with keys:
Methods, Results.

Rules:
- Copy or lightly normalize text from the Markdown; do not invent facts.
- Methods must contain study design, participants/cases, task/stimuli, acquisition, preprocessing, analysis, assessment, or case/procedure description.
- If Methods and Results are written together, split procedural parts into Methods and findings/outcomes/statistics into Results.
- For case studies, put patient/case description, assessment, procedure, and mapping protocol into Methods.
- If the article lacks an explicit Methods title, infer Methods from procedural paragraphs.
- Results must contain findings, outcomes, activation results, behavioral results,
  statistical comparisons, or reported observations, not only a heading.
- Do not include References in either section.
- Every value must be a string."""

        user_prompt = (
            "Review metadata:\n"
            f"{issue_note}\n\n"
            "Markdown:\n"
            f"{prompt_text}"
        )

        errors = []
        client_order = [self.preferred_client]
        for fallback in ("mimo", "deepseek"):
            if fallback not in client_order and f"{fallback}_api_key" in self.config:
                client_order.append(fallback)

        for client_type in client_order:
            for retry_idx in range(3):
                try:
                    client, model_name = self.get_client(client_type)
                    self.llm_calls += 1
                    request_kwargs = {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": 8192,
                        "response_format": {"type": "json_object"},
                    }
                    if (
                        client_type == "deepseek"
                        and self.config.get("deepseek_thinking_disabled", False)
                    ):
                        request_kwargs["extra_body"] = {
                            "thinking": {"type": "disabled"}
                        }
                    response = client.chat.completions.create(**request_kwargs)
                    time.sleep(self.sleep_seconds)
                    data = parse_json_object(response.choices[0].message.content)
                    return {
                        category: str(data.get(category, "") or "").strip()
                        for category in TARGET_CATEGORIES
                    }, {
                        "llm_client": client_type,
                        "llm_model": model_name,
                        "llm_error": "",
                        "llm_retry_count": retry_idx,
                    }
                except Exception as exc:
                    errors.append(
                        f"{client_type}[{retry_idx}]: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    if retry_idx < 2:
                        time.sleep(2 ** retry_idx)

        return {}, {"llm_client": "", "llm_model": "", "llm_error": " | ".join(errors)}

    def segment_markdown(
        self,
        markdown_path: Path,
        review_record: Optional[Dict[str, Any]] = None,
        existing_content: Optional[Dict[str, Any]] = None,
    ) -> FinalSegResult:
        raw_text = markdown_path.read_text(encoding="utf-8", errors="replace")
        cleaned = clean_markdown_text(raw_text)
        table_cleaned, tables = extract_tables(cleaned)
        blocks = split_into_blocks(table_cleaned)
        classify_blocks_with_rules(blocks)
        structured = build_structured(blocks, tables)

        llm_meta: Dict[str, Any] = {}
        if self.should_use_llm(review_record, structured):
            llm_structured, llm_meta = self.llm_extract_sections(table_cleaned, review_record)
            for category in ("Methods", "Results"):
                if (
                    llm_structured.get(category)
                    and not has_meaningful_section(structured.get(category))
                ):
                    structured[category] = llm_structured[category]

        if not has_meaningful_section(structured.get("Methods")):
            structured["Methods"] = heuristic_method_fallback(table_cleaned)
        if not has_meaningful_section(structured.get("Results")):
            structured["Results"] = heuristic_results_fallback(table_cleaned)

        if existing_content:
            for category in ("Title", "Author", "Keywords", "Abstract", "References", "Acknowledgements", "Other"):
                if not structured.get(category) and existing_content.get(category):
                    structured[category] = existing_content[category]
            if not structured.get("Tables") and existing_content.get("Tables"):
                structured["Tables"] = existing_content["Tables"]

        structured["Methods_Hierarchy"] = [
            {
                "heading": "Final inferred Methods",
                "level": 2,
                "content": structured.get("Methods", ""),
                "children": [],
            }
        ] if structured.get("Methods") else []

        detected_sections = [category for category in TARGET_CATEGORIES if str(structured.get(category, "")).strip()]
        metadata = {
            "source_file": str(markdown_path),
            "strategy": "final_seg_rules_plus_llm",
            "review_record": review_record or {},
            "sections_detected": detected_sections,
            "required_sections_complete": all(
                has_meaningful_section(structured.get(x))
                for x in ("Methods", "Results")
            ),
            "segmentation_success": has_meaningful_section(structured.get("Methods")),
            "has_methods": has_meaningful_section(structured.get("Methods")),
            "has_results": has_meaningful_section(structured.get("Results")),
            "llm_calls_total": self.llm_calls,
            "llm_meta": llm_meta,
            "block_count": len(blocks),
            "table_count": len(structured.get("Tables", [])),
            "blocks": [{k: v for k, v in block.items() if k != "content"} for block in blocks],
        }
        return FinalSegResult(structured=structured, metadata=metadata)
