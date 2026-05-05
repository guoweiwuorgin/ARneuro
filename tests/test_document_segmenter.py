from pathlib import Path

from text_processing.document_segmentation import DocumentSegmenter


def test_segment_document_rule_based_success(tmp_path: Path):
    md = tmp_path / "paper.md"
    md.write_text(
        """
# A Great Study
## Authors
Alice; Bob
## Keywords
fMRI; memory
## Abstract
This is abstract.
## Introduction
Intro text.
## Methods
Method overview.
### Subjects
20 participants.
### Statistical Analysis
ANOVA.
## Results
Result text.
## Discussion
Discussion text.
## References
[1] Ref
""",
        encoding="utf-8",
    )

    seg = DocumentSegmenter()
    structured, meta = seg.segment_document(str(md))

    assert meta["required_sections_complete"] is True
    assert meta["strategy"] == "rule_based"
    assert structured["Methods"]
    assert len(structured["Methods_Hierarchy"]) >= 1


def test_segment_document_keyword_fallback(tmp_path: Path):
    md = tmp_path / "messy.md"
    md.write_text(
        """
# Weird Title
## Section A
In this part, we describe methods and participants.
## Section B
Here are our results and findings in detail.
## Section C
General discussion and conclusion.
## Section D
References
""",
        encoding="utf-8",
    )

    seg = DocumentSegmenter()
    structured, meta = seg.segment_document(str(md), llm_classifier=None)

    assert meta["strategy"] in {"keyword_fallback", "rule_based"}
    assert meta["table_count"] == 0
    assert "Results" in structured
