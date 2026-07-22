"""Recover method-like blocks that were incorrectly left in ``Other``.

This module is deliberately independent from the original segmentation pass.
It repairs an already structured paper by splitting ``Other`` at Markdown
headings and paragraph boundaries, then moving strongly method-like blocks into
``Methods``. The operation is deterministic and idempotent.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


HEADING_RE = re.compile(r"(?m)^(#{1,6})[ \t]+(.+?)[ \t]*$")
PAGE_MARKER_RE = re.compile(r"<!--\s*Page\s+\d+\s*-->", re.I)

# A heading match is sufficient evidence because headings define the author's
# document structure. Patterns cover participant descriptions, experimental
# procedures, acquisition, preprocessing, and inferential analysis.
METHOD_HEADING_PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = (
    ("method_heading", re.compile(r"\bmethods?\b|\bmethodology\b", re.I)),
    (
        "task_heading",
        re.compile(r"\btask(?:s)?\s*$", re.I),
    ),
    (
        "participant_heading",
        re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s*)?"
            r"(?:(?:healthy|typical|control|clinical|stroke|aphasic|epilepsy|"
            r"patient|experimental)\s+)?"
            r"(?:subjects?|participants?|patients?|volunteers?|controls?)"
            r"(?:\s+(?:and|&)\s+(?:subjects?|participants?|patients?|controls?))?"
            r"(?:\s+(?:characteristics?|demographics?|selection|recruitment))?\s*$|"
            r"^(?:\d+(?:\.\d+)*\.?\s*)?"
            r"(?:subjects?|participants?|patients?|volunteers?|controls?)"
            r"\s*(?:,|and|&|/|-)\s*"
            r"(?:experimental\s+)?(?:procedure|procedures|design|methods?|"
            r"assessment|assessments|testing|recruitment|selection|"
            r"characteristics|demographics)\s*$|"
            r"^(?:\d+(?:\.\d+)*\.?\s*)?"
            r"(?:experimental\s+)?(?:procedure|procedures|design|methods?|"
            r"assessment|assessments|testing|recruitment|selection)"
            r"\s*(?:,|and|&|/|-)\s*"
            r"(?:subjects?|participants?|patients?|volunteers?|controls?)\s*$|"
            r"\b(?:study|patient) (?:population|sample|cohort)\b|"
            r"\bsample characteristics?\b|\bdemographics?\b|"
            r"\brecruitment\b|\beligibility\b|\binclusion\b|\bexclusion\b",
            re.I,
        ),
    ),
    (
        "experiment_heading",
        re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s*)?"
            r"(?:experiments?(?:\s+\d+)?(?:\s*[:\-].*)?|"
            r"(?:(?:first|second|third|fourth|fifth|\d+(?:st|nd|rd|th))\s+)"
            r"experiments?(?:\s*[:\-].*)?|"
            r"experimental (?:design|procedure|procedures|paradigm|protocol)|"
            r"(?:general\s+)?(?:study\s+)?design(?:\s*\(.*\))?|"
            r"(?:general\s+)?procedures?|protocol|paradigms?|apparatus|"
            r"materials?(?:\s+and\s+(?:methods?|procedure))?|"
            r"stimuli?(?:\s*\(.*\))?"
            r"(?:\s+(?:and|,)\s+(?:design|task|tasks|procedure|materials|"
            r"randomization))?|stimuli? randomization|"
            r"(?:(?:visual|auditory|behavioral|behavioural|language|psychological|"
            r"scanning|in-scanner|lexical decision|functional localizer|"
            r"forced-choice|main)\s+)?tasks?"
            r"(?:\s+(?:and|&)\s+(?:procedure|design|stimuli|performance))?)\s*$|"
            r"\btask and procedure\b|\bdesign and procedure\b|"
            r"\bmaterials and procedure\b|\bstimuli and design\b",
            re.I,
        ),
    ),
    (
        "acquisition_heading",
        re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s*)?"
            r"(?:functional\s+)?(?:mri|fmri|pet|meg|eeg|bold|neuroimaging|imaging)"
            r"(?:\s+(?:data|image|scan))?"
            r"(?:\s+(?:acquisition|scanning|recording|parameters?|protocol|"
            r"pre-?processing|processing|analysis|analyses|design|experiment|"
            r"procedure|methods?|technique))?"
            r"(?:\s*(?:,|and|&|/|-)\s*(?:acquisition|scanning|recording|"
            r"parameters?|protocol|pre-?processing|processing|analysis|analyses|"
            r"design|procedure|methods?|technique))*\s*$|"
            r"\b(?:data|image|scan) acquisition\b|\bscanning (?:procedure|session)s?\b|"
            r"\bimaging parameters?\b|\bscanning parameters?\b",
            re.I,
        ),
    ),
    (
        "analysis_heading",
        re.compile(
            r"^(?:\d+(?:\.\d+)*\.?\s*)?(?:analysis|analyses|statistics)\s*$|"
            r"\bdata analys(?:is|es)\b|\bimage analys(?:is|es)\b|"
            r"\bstatistical analys(?:is|es)\b|\bbehavioral analys(?:is|es)\b|"
            r"\bbehavioural analys(?:is|es)\b|\bpre-?processing\b|"
            r"\bnormalization\b|\brealignment\b|\bregion of interest\b|"
            r"\broi analys(?:is|es)\b|\bwhole-?brain (?:f?mri )?analys(?:is|es)\b|"
            r"\b(?:f?mri|erp|eeg|meg|pet) statistics\b|"
            r"\bsource reconstruction analys(?:is|es)\b|"
            r"\bimage processing and subtraction analys(?:is|es)\b|"
            r"\bacoustic analys(?:is|es)\b|\bcontent analys(?:is|es)\b|"
            r"\banalys(?:is|es) comparing experiments?\b|"
            r"\bremoval of confounding effect\b",
            re.I,
        ),
    ),
    (
        "clinical_heading",
        re.compile(
            r"\bclinical assessment\b|\bneuropsychological assessment\b|"
            r"\bdiagnostic assessment\b|\bethics?\b|\binformed consent\b",
            re.I,
        ),
    ),
)

NON_METHOD_HEADING_RE = re.compile(
    r"\b(results?|findings?|discussion|conclusions?|references?|bibliography|"
    r"acknowledg(?:e)?ments?|limitations?|supplementary references?)\b",
    re.I,
)
SUPPLEMENTARY_HEADING_RE = re.compile(
    r"\b(?:appendix\b.*\b)?supplementary (?:material|materials|information|data)\b",
    re.I,
)
TITLE_LIKE_IMAGING_RE = re.compile(
    r"\b(?:an?|the)\s+(?:functional\s+)?(?:mri|fmri|pet|meg|eeg)\s+study\b|"
    r"\b(?:er-?)?(?:mri|fmri|pet|meg|eeg)\s+study\b|"
    r"\bstudy of\b|^a comparison of\b|"
    r"\b(?:identified|revealed|assessed|investigated|mapped|predicted)\s+"
    r"(?:by|with|using)\b|"
    r"^(?:neural correlates?|functional differentiation|predicting growth|"
    r"language dominance|cerebrocerebellar networks?)\b",
    re.I,
)
RESULT_LIKE_HEADING_RE = re.compile(
    r"\b(?:significant|activat(?:ion|ions|ed)|effects?|performances?|accuracy|reaction times?|"
    r"correlations?|interactions?|responses?|differences?|findings?|prefer(?:s|ence)?|"
    r"task-related activity|network activity|between-group|within-group|"
    r"neural correlates?|main network|global network|functional overlap|"
    r"changes in|consistency of|concordance of|areas? activated|"
    r"factors? held constant)\b|"
    r"\b(?:versus|vs\.)\b|^analyses? of\b",
    re.I,
)
STRONG_METHOD_STRUCTURE_RE = re.compile(
    r"\b(?:methods?|acquisition|analys(?:is|es)|pre-?processing|procedure|"
    r"design|stimuli|materials|parameters|protocol|recruitment|participants?|"
    r"subjects?|patients?)\b",
    re.I,
)
METHOD_HEADING_ATOM_RE = re.compile(
    r"\b(?:methods?|methodology|subjects?|participants?|patients?|volunteers?|"
    r"controls?|sample|cohort|recruitment|eligibility|inclusion|exclusion|"
    r"demographics?|characteristics?|procedure|procedures|design|protocol|"
    r"experiment|experimental|paradigm|tasks?|testing|stimuli?|materials?|"
    r"apparatus|f?mri|pet|meg|eeg|bold|neuroimaging|imaging|scanner|scanning|"
    r"acquisition|recording|parameters?|pre-?processing|processing|analysis|"
    r"analyses|statistical|behavioral|behavioural|roi|region of interest|"
    r"assessment|assessments|clinical|neuropsychological|ethics?|consent|"
    r"practice|session|randomization|localizer)\b",
    re.I,
)
METHOD_HEADING_NOISE_RE = re.compile(
    r"\b(?:the|a|an|of|for|in|on|with|without|using|and|or|to|from|by|"
    r"study|data|image|images|functional|magnetic|resonance|statement)\b|"
    r"[\d.\-–—,:;/&()\[\]]+",
    re.I,
)
AMBIGUOUS_METHOD_CONTEXT_HEADING_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*\.?\s*)?"
    r"(?:(?:study|session|experimental|methodological|technical|scanning|"
    r"imaging|behavioral|behavioural|data|image)\s+)"
    r"(?:details?|overview|setup|information)\s*$",
    re.I,
)

PROSE_EVIDENCE: Sequence[Tuple[str, re.Pattern[str]]] = (
    (
        "participants",
        re.compile(
            r"\b(?:subjects?|participants?|patients?|volunteers?)\s+"
            r"(?:were|included|underwent|performed|completed|received)|"
            r"\b(?:recruited|enrolled|included|excluded)\b|"
            r"\b(?:healthy|right-handed|left-handed)\s+(?:subjects?|participants?|"
            r"volunteers?|controls?)\b|\bage range\b|\bwritten informed consent\b",
            re.I,
        ),
    ),
    (
        "procedure",
        re.compile(
            r"\b(?:subjects?|participants?|patients?)\s+(?:were asked|performed|"
            r"completed|underwent|viewed|listened|responded)|"
            r"\b(?:stimuli|trials?|blocks?|runs?)\s+(?:were|consisted|included)|"
            r"\btask (?:was|consisted|included)|\bexperimental (?:design|procedure)|"
            r"\bresponse (?:button|key|box)\b",
            re.I,
        ),
    ),
    (
        "acquisition",
        re.compile(
            r"\b(?:f?mri|pet|meg|eeg|bold)\b.{0,240}\b(?:acquir|record|scan|measure)|"
            r"\b(?:acquir|record|scan)\w*\b.{0,240}\b(?:f?mri|pet|meg|eeg|bold)\b|"
            r"\b(?:scanner|magnet)\b|\b[137]\s*(?:t|tesla)\b|"
            r"\b(?:repetition time|echo time|flip angle|voxel size|slice thickness|"
            r"tr\s*=|te\s*=)\b",
            re.I | re.S,
        ),
    ),
    (
        "preprocessing",
        re.compile(
            r"\b(?:pre-?process|realign|normaliz|coregister|segment|smooth)\w*\b|"
            r"\b(?:spm|fsl|afni|freesurfer|matlab)\b",
            re.I,
        ),
    ),
    (
        "statistics",
        re.compile(
            r"\b(?:general linear model|glm|anova|ancova|t-?tests?|"
            r"mixed effects?|random effects?|regression model)\b|"
            r"\bstatistical (?:analysis|analyses|threshold|significance)\b|"
            r"\b(?:whole-?brain|voxelwise|voxel-wise|roi|region of interest)\b",
            re.I,
        ),
    ),
)

RESULT_PROSE_RE = re.compile(
    r"\b(?:we found|we observed|results? (?:showed|revealed|indicated)|"
    r"was significantly (?:greater|higher|lower)|were significantly|"
    r"these findings|our findings|in conclusion)\b",
    re.I,
)
REFERENCE_LIKE_RE = re.compile(
    r"(?:\b[A-Z][A-Za-z'-]+(?:\s+[A-Z]\.){1,3}.*\b(?:19|20)\d{2}\b)|"
    r"(?:^\s*\[\d+\]\s+)|(?:\bdoi\s*:)|(?:\bet al\.\s*\(\d{4}\))",
    re.I | re.M,
)


@dataclass
class OtherBlock:
    heading: str
    level: int
    content: str
    source_index: int

    def markdown(self) -> str:
        parts: List[str] = []
        if self.heading:
            parts.append(f"{'#' * self.level} {self.heading}")
        if self.content.strip():
            parts.append(self.content.strip())
        return "\n".join(parts).strip()


@dataclass
class OtherMethodRecoveryResult:
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    changed: bool
    recovered_blocks: List[Dict[str, Any]]
    recovered_chars: int


def _split_long_unheaded_text(text: str, max_chars: int) -> List[str]:
    """Split oversized prose without cutting Markdown headings or sentences."""

    text = text.strip()
    if not text:
        return []
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n|(?=<!--\s*Page\s+\d+\s*-->)", text)
        if part.strip()
    ]
    output: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            output.append(paragraph)
            continue
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", paragraph)
        current: List[str] = []
        current_chars = 0
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current and current_chars + len(sentence) + 1 > max_chars:
                output.append(" ".join(current))
                current = []
                current_chars = 0
            if len(sentence) > max_chars:
                if current:
                    output.append(" ".join(current))
                    current = []
                    current_chars = 0
                output.extend(
                    sentence[start : start + max_chars]
                    for start in range(0, len(sentence), max_chars)
                )
            else:
                current.append(sentence)
                current_chars += len(sentence) + 1
        if current:
            output.append(" ".join(current))
    return output


def split_other_into_blocks(other_text: str, max_unheaded_chars: int = 4000) -> List[OtherBlock]:
    """Split ``Other`` by headings, then paragraphs/sentences when unheaded."""

    text = str(other_text or "").strip()
    if not text:
        return []

    matches = list(HEADING_RE.finditer(text))
    blocks: List[OtherBlock] = []
    source_index = 0

    root_text = text[: matches[0].start()] if matches else text
    for part in _split_long_unheaded_text(root_text, max_unheaded_chars):
        blocks.append(OtherBlock("", 0, part, source_index))
        source_index += 1

    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(2).strip()
        level = len(match.group(1))
        body = text[body_start:body_end].strip()
        # A heading already provides a reliable structural boundary, so keep its
        # complete body together even when it is long.
        blocks.append(OtherBlock(heading, level, body, source_index))
        source_index += 1
    return blocks


def _heading_method_reason(heading: str) -> Optional[str]:
    heading = PAGE_MARKER_RE.sub(" ", heading or "").strip()
    method_structure_terms = re.search(
        r"\b(methods?|participants?|subjects?|patients?|acquisition|analysis|"
        r"pre-?processing|procedure|design|task|stimuli|scanner|scanning)\b",
        heading,
        re.I,
    )
    if (
        not heading
        or NON_METHOD_HEADING_RE.search(heading)
        or SUPPLEMENTARY_HEADING_RE.search(heading)
        or TITLE_LIKE_IMAGING_RE.search(heading)
        or re.match(r"^(?:appendix|fig(?:ure)?\.?)\b", heading, re.I)
        or (len(heading) > 110 and not method_structure_terms)
        or re.search(r"\bdiscrepanc(?:y|ies) between experiments?\b", heading, re.I)
        or RESULT_LIKE_HEADING_RE.search(heading)
    ):
        return None
    for reason, pattern in METHOD_HEADING_PATTERNS:
        if pattern.search(heading):
            return reason
    atoms = METHOD_HEADING_ATOM_RE.findall(heading)
    residual = METHOD_HEADING_ATOM_RE.sub(" ", heading)
    residual = METHOD_HEADING_NOISE_RE.sub(" ", residual)
    residual = re.sub(r"\s+", " ", residual).strip()
    if len(atoms) >= 2 and not residual:
        return "compound_method_heading"
    return None


def _heading_allows_prose_recovery(heading: str) -> bool:
    heading = PAGE_MARKER_RE.sub(" ", heading or "").strip()
    if not heading:
        return True
    if (
        NON_METHOD_HEADING_RE.search(heading)
        or SUPPLEMENTARY_HEADING_RE.search(heading)
        or TITLE_LIKE_IMAGING_RE.search(heading)
        or RESULT_LIKE_HEADING_RE.search(heading)
    ):
        return False
    return bool(AMBIGUOUS_METHOD_CONTEXT_HEADING_RE.fullmatch(heading))


def _prose_method_reasons(text: str) -> List[str]:
    clean = PAGE_MARKER_RE.sub(" ", text or "")
    if len(re.sub(r"\s+", " ", clean).strip()) < 80:
        return []
    reasons = [name for name, pattern in PROSE_EVIDENCE if pattern.search(clean)]
    citation_hits = len(REFERENCE_LIKE_RE.findall(clean))
    word_count = max(1, len(re.findall(r"\b\w+\b", clean)))
    reference_heavy = citation_hits >= 3 and citation_hits * 35 > word_count
    if reference_heavy:
        return []
    if RESULT_PROSE_RE.search(clean) and len(reasons) < 2:
        return []
    # Two independent evidence families are required for unheaded prose. A
    # scanner/acquisition paragraph is sufficiently specific on its own.
    if len(reasons) >= 2 or "acquisition" in reasons:
        return reasons
    return []


def _append_unique(existing: str, additions: Sequence[str]) -> str:
    result = str(existing or "").strip()
    for addition in additions:
        addition = addition.strip()
        if not addition or addition in result:
            continue
        result = f"{result}\n\n{addition}".strip() if result else addition
    return result


class OtherMethodSectionRecoverer:
    """Repair method blocks in an already structured paper."""

    def __init__(self, max_unheaded_chars: int = 4000) -> None:
        self.max_unheaded_chars = max(500, int(max_unheaded_chars))

    def recover(
        self,
        content: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OtherMethodRecoveryResult:
        updated_content = dict(content or {})
        updated_metadata = dict(metadata or {})
        original_other = str(updated_content.get("Other", "") or "").strip()
        original_methods = str(updated_content.get("Methods", "") or "").strip()
        blocks = split_other_into_blocks(original_other, self.max_unheaded_chars)

        recovered: List[Tuple[OtherBlock, List[str]]] = []
        remaining: List[OtherBlock] = []
        for block in blocks:
            heading_reason = _heading_method_reason(block.heading)
            prose_reasons = (
                _prose_method_reasons(block.content)
                if not heading_reason and _heading_allows_prose_recovery(block.heading)
                else []
            )
            reasons = [heading_reason] if heading_reason else prose_reasons
            block_markdown = block.markdown()
            # Empty parent headings such as "Experiment 1" do not add evidence;
            # their concrete child headings are evaluated independently.
            if reasons and block.content.strip() and block_markdown:
                recovered.append((block, reasons))
            else:
                remaining.append(block)

        recovered_markdown = [block.markdown() for block, _ in recovered]
        updated_methods = _append_unique(original_methods, recovered_markdown)
        updated_other = "\n\n".join(
            block.markdown() for block in remaining if block.markdown()
        ).strip()
        changed = updated_methods != original_methods or updated_other != original_other

        recovered_records = [
            {
                "heading": block.heading,
                "heading_level": block.level,
                "source_index": block.source_index,
                "reasons": reasons,
                "chars": len(block.markdown()),
                "preview": re.sub(r"\s+", " ", block.markdown())[:240],
            }
            for block, reasons in recovered
        ]
        recovered_chars = sum(item["chars"] for item in recovered_records)

        if changed:
            updated_content["Methods"] = updated_methods
            updated_content["Other"] = updated_other
            hierarchy = list(updated_content.get("Methods_Hierarchy") or [])
            for block, reasons in recovered:
                hierarchy.append(
                    {
                        "heading": block.heading or "Recovered method-like prose",
                        "level": block.level or 3,
                        "content": block.content.strip(),
                        "children": [],
                        "recovered_from": "Other",
                        "recovery_reasons": reasons,
                    }
                )
            updated_content["Methods_Hierarchy"] = hierarchy

        audit = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "version": 2,
            "changed": changed,
            "other_sha256_before": hashlib.sha256(
                original_other.encode("utf-8")
            ).hexdigest(),
            "other_chars_before": len(original_other),
            "other_chars_after": len(updated_other),
            "methods_chars_before": len(original_methods),
            "methods_chars_after": len(updated_methods),
            "recovered_block_count": len(recovered_records),
            "recovered_chars": recovered_chars,
            "recovered_blocks": recovered_records,
        }
        previous_audit = updated_metadata.get("other_method_recovery")
        if changed and isinstance(previous_audit, dict):
            history = list(updated_metadata.get("other_method_recovery_history") or [])
            if not history or history[-1] != previous_audit:
                history.append(previous_audit)
            updated_metadata["other_method_recovery_history"] = history
        if changed or "other_method_recovery" not in updated_metadata:
            updated_metadata["other_method_recovery"] = audit

        return OtherMethodRecoveryResult(
            content=updated_content,
            metadata=updated_metadata,
            changed=changed,
            recovered_blocks=recovered_records,
            recovered_chars=recovered_chars,
        )


def recover_methods_from_other(
    content: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    max_unheaded_chars: int = 4000,
) -> OtherMethodRecoveryResult:
    """Convenience function for callers that do not need a persistent object."""

    return OtherMethodSectionRecoverer(max_unheaded_chars=max_unheaded_chars).recover(
        content=content,
        metadata=metadata,
    )
