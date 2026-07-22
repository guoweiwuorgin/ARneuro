"""Document segmentation module for ARneuro."""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)


SECTION_HEADING_PATTERNS = [
    r"abstract",
    r"summary",
    r"keywords?",
    r"key\s+words?",
    r"introduction",
    r"background",
    r"materials?\s+and\s+methods?",
    r"methods?",
    r"methodology",
    r"experimental\s+procedures?",
    r"participants?",
    r"subjects?",
    r"patients?",
    r"data\s+acquisition",
    r"statistical\s+analys(e|i)s",
    r"results?",
    r"findings?",
    r"discussion",
    r"conclusions?",
    r"limitations?",
    r"references?",
    r"bibliography",
    r"acknowledg(e)?ments?",
    r"funding",
    r"author\s+contributions?",
    r"conflicts?\s+of\s+interest",
    r"competing\s+interests?",
    r"supplementary\s+materials?",
]


def _clean_heading(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"^#{1,6}\s*", "", text or "")).strip()


def _normalize_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"^[\d.\-–—?)\[\]\s]+", "", text)
    return re.sub(r"\s+", " ", text.lower()).strip()


def _detect_unmarked_heading(text: str) -> Optional[Tuple[str, int]]:
    stripped = (text or "").strip()
    if not stripped or len(stripped) > 180 or stripped.startswith("|"):
        return None
    if stripped.endswith(".") and not re.match(r"^\d+(\.\d+)*\.?\s+", stripped):
        return None
    if stripped.count(",") >= 2:
        return None

    normalized = _normalize_text(stripped)
    pattern_text = "|".join(SECTION_HEADING_PATTERNS)
    if not normalized:
        return None

    if re.fullmatch(rf"({pattern_text})", normalized, flags=re.I):
        return _clean_heading(stripped), 2
    if re.fullmatch(rf"({pattern_text})\s*:", normalized, flags=re.I):
        return _clean_heading(stripped.rstrip(":")), 2
    if re.fullmatch(
        rf"({pattern_text})(\s+(and|&|/|-)\s+({pattern_text}))*",
        normalized,
        flags=re.I,
    ) and len(normalized.split()) <= 8:
        return _clean_heading(stripped.rstrip(":")), 2

    numbered = re.match(rf"^\d+(\.\d+)*\.?\s+({pattern_text})(\b|:)", normalized, flags=re.I)
    if numbered:
        level = 3 if "." in normalized.split()[0] else 2
        return _clean_heading(stripped), level

    alpha_chars = re.sub(r"[^A-Za-z]", "", stripped)
    if 4 <= len(alpha_chars) <= 60 and alpha_chars.isupper() and re.search(pattern_text, normalized, flags=re.I):
        return _clean_heading(stripped), 2

    return None


def _join_paragraph_lines(lines: List[str]) -> str:
    paragraph = ""
    for line in [x.strip() for x in lines if x.strip()]:
        if not paragraph:
            paragraph = line
        elif paragraph.endswith("-") and not paragraph.endswith("--") and re.match(r"^[a-z]", line):
            paragraph = paragraph[:-1] + line
        else:
            paragraph += " " + line
    return re.sub(r"\s+", " ", paragraph).strip()


def _starts_new_logical_paragraph(previous_line: str, current_line: str, paragraph_lines: List[str]) -> bool:
    current = (current_line or "").strip()
    previous = (previous_line or "").strip()
    if not current or not previous or not paragraph_lines:
        return False
    if re.match(r"^(\d+[\.)]|[\[\(]\d+[\]\)]|[-*])\s+", current):
        return True
    if re.match(r"^(figure|fig\.|table)\s+\d+[\.:]", current, flags=re.I):
        return True
    if re.match(r"^(background|objective|objectives|methods|results|conclusions?|keywords?)\s*:", current, flags=re.I):
        return True
    if previous.endswith((".", "!", "?", ";", ":")) and re.match(r"^[A-Z0-9(]", current):
        return len(" ".join(paragraph_lines)) >= 80
    return False


def _format_blocks_once(blocks: List[Dict]) -> str:
    pieces = []
    emitted_headings = set()
    for block in sorted(blocks, key=lambda x: (x.get("line_start", 0), x.get("line_end", 0))):
        heading = (block.get("heading") or "Document_Root").strip()
        content = (block.get("content") or "").strip()
        block_kind = block.get("block_kind", "section")
        heading_key = (heading, block.get("heading_level", 0))

        if heading == "Document_Root":
            if content:
                pieces.append(content)
            continue

        if block_kind == "heading_marker":
            if heading_key not in emitted_headings:
                emitted_headings.add(heading_key)
                pieces.append(f"### {heading}")
            continue

        if heading_key not in emitted_headings:
            emitted_headings.add(heading_key)
            text = f"### {heading}\n{content}".strip()
        else:
            text = content

        if text:
            pieces.append(text)

    return "\n\n".join(pieces).strip()


class DocumentSegmenter:
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
    REQUIRED_CORE = {"Introduction", "Methods", "Results", "Discussion", "References"}

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    def _is_page_marker(self, text: str) -> bool:
        return bool(re.match(r"^##\s+Page\s+\d+", text, flags=re.I))

    def _extract_blocks(self, lines: List[str]) -> Tuple[List[Dict], List[Dict], Dict]:
        section_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
        total_pages_re = re.compile(r"\*\*Total Pages:\*\*\s*(\d+)", re.I)
        sep_re = re.compile(r"^={10,}$|^-{10,}$")
        html_start_re = re.compile(r"<table\b[^>]*>", re.I)
        html_end_re = re.compile(r"</table>", re.I)

        blocks, tables = [], []
        meta = {
            "declared_total_pages": None,
            "parsed_page_markers": 0,
            "line_count": len(lines),
            "extraction_strategy": "paragraph_blocks_with_heading_detection",
        }
        current_heading = "Document_Root"
        current_heading_level = 0
        paragraph_lines = []
        paragraph_start_line = None

        def append_block(line_start: int, line_end: int, content: str, block_kind: str) -> None:
            blocks.append({
                "heading": current_heading,
                "heading_level": current_heading_level,
                "line_start": line_start,
                "line_end": line_end,
                "content": (content or "").strip(),
                "block_kind": block_kind,
                "category": "Other",
            })

        def flush_paragraph(end_line: int) -> None:
            nonlocal paragraph_lines, paragraph_start_line
            if not paragraph_lines:
                return
            paragraph = _join_paragraph_lines(paragraph_lines)
            if paragraph:
                append_block(paragraph_start_line or end_line, end_line, paragraph, "paragraph")
            paragraph_lines = []
            paragraph_start_line = None

        i = 0
        while i < len(lines):
            raw = lines[i].rstrip("\n")
            s = raw.strip()
            if not s or sep_re.match(s):
                flush_paragraph(i)
                i += 1
                continue
            m = total_pages_re.search(s)
            if m:
                meta["declared_total_pages"] = int(m.group(1))
                i += 1
                continue
            if self._is_page_marker(s):
                flush_paragraph(i)
                meta["parsed_page_markers"] += 1
                i += 1
                continue
            if html_start_re.match(s):
                flush_paragraph(i)
                t_lines = []
                start = i
                while i < len(lines):
                    t_lines.append(lines[i].rstrip("\n"))
                    if html_end_re.search(lines[i]):
                        i += 1
                        break
                    i += 1
                table_id = f"table_{len(tables)+1}"
                tables.append({"id": table_id, "content": "\n".join(t_lines), "line_start": start + 1, "line_end": i, "table_type": "html"})
                append_block(start + 1, i, f"[TABLE: {table_id}]", "table_placeholder")
                continue
            if s.startswith("|"):
                flush_paragraph(i)
                t_lines = []
                start = i
                while i < len(lines) and lines[i].strip().startswith("|"):
                    t_lines.append(lines[i].rstrip("\n"))
                    i += 1
                table_id = f"table_{len(tables)+1}"
                tables.append({"id": table_id, "content": "\n".join(t_lines), "line_start": start + 1, "line_end": i, "table_type": "markdown_pipe"})
                append_block(start + 1, i, f"[TABLE: {table_id}]", "table_placeholder")
                continue

            head = section_re.match(s)
            unmarked_head = None if head else _detect_unmarked_heading(s)
            if head or unmarked_head:
                flush_paragraph(i)
                if head:
                    current_heading = _clean_heading(head.group(2))
                    current_heading_level = len(head.group(1))
                else:
                    current_heading, current_heading_level = unmarked_head
                append_block(i + 1, i + 1, "", "heading_marker")
                i += 1
                continue

            if paragraph_start_line is None:
                paragraph_start_line = i + 1
            if paragraph_lines and _starts_new_logical_paragraph(paragraph_lines[-1], raw, paragraph_lines):
                flush_paragraph(i)
                paragraph_start_line = i + 1
            paragraph_lines.append(raw)
            i += 1
        flush_paragraph(len(lines))
        return blocks, tables, meta

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"^[\d.\-()\s]+", "", text.lower())).strip()

    def _rule_category(self, heading: str) -> str:
        t = self._normalize(heading)
        rules = {
            "Keywords": ["keywords", "key words"],
            "Abstract": ["abstract", "summary"],
            "Introduction": ["introduction", "background"],
            "Methods": ["methods", "materials and methods", "methodology", "experimental", "participants", "subjects", "statistical"],
            "Results": ["results", "findings"],
            "Discussion": ["discussion", "conclusion", "limitations", "implications"],
            "References": ["references", "bibliography"],
            "Acknowledgements": ["acknowledg", "funding", "conflict of interest"],
            "Author": ["author", "affiliation"],
        }
        for k, vals in rules.items():
            if any(v in t for v in vals):
                return k
        return "Other"

    def _classify_blocks(self, blocks: List[Dict], llm_classifier=None) -> Tuple[List[Dict], str]:
        categories = [self._rule_category(b["heading"]) for b in blocks]
        # heuristics for first block/title
        if blocks and categories[0] == "Other":
            categories[0] = "Title"
        for i, b in enumerate(blocks):
            if b["heading"] == "Document_Root" and categories[i] == "Other":
                categories[i] = "Title"

        core = {c for c in categories if c in self.REQUIRED_CORE}
        strategy = "rule_based"
        if core != self.REQUIRED_CORE and llm_classifier:
            strategy = "llm_assisted"
            headings = [b["heading"] for b in blocks]
            cls = llm_classifier.classify_headings(headings)
            mapping = llm_classifier.map_sections_to_categories(headings, cls)
            categories = [mapping.get(h) or categories[i] for i, h in enumerate(headings)]
            core = {c for c in categories if c in self.REQUIRED_CORE}

        if core != self.REQUIRED_CORE:
            strategy = "keyword_fallback"
            for i, b in enumerate(blocks):
                if categories[i] == "Other":
                    blob = f"{b['heading']} {b.get('content','')[:400]}"
                    categories[i] = self._rule_category(blob)
        for i, b in enumerate(blocks):
            b["category"] = categories[i]
        return blocks, strategy

    def _build_section_content(self, blocks: List[Dict]) -> Dict:
        out = {k: "" for k in self.TARGET_CATEGORIES}
        by_cat = defaultdict(list)
        for b in blocks:
            by_cat[b.get("category", "Other")].append(b)
        for cat, arr in by_cat.items():
            out[cat] = _format_blocks_once(arr)

        method_blocks = [b for b in blocks if b["category"] == "Methods"]
        method_hierarchy = []
        stack = []
        for mb in sorted(method_blocks, key=lambda x: x["line_start"]):
            node = {"heading": mb["heading"], "level": mb["heading_level"], "content": mb["content"], "children": []}
            while stack and stack[-1]["level"] >= node["level"]:
                stack.pop()
            if stack:
                stack[-1]["children"].append(node)
            else:
                method_hierarchy.append(node)
            stack.append(node)
        out["Methods_Hierarchy"] = method_hierarchy
        return out

    def parse_markdown_file(self, file_path: str) -> Tuple[Dict, List, List, List]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        blocks, tables, meta = self._extract_blocks(lines)
        grouped = defaultdict(list)
        for block in blocks:
            if block.get("content"):
                grouped[block["heading"]].append(block["content"])
        document_structure = {
            heading: "\n\n".join(parts).strip()
            for heading, parts in grouped.items()
        }
        document_structure["Metadata_Total_Pages"] = meta["declared_total_pages"]
        document_structure["Parsed_Page_Count"] = meta["parsed_page_markers"]
        return document_structure, [t["content"] for t in tables], [""] * len(tables), [""] * len(tables)

    def segment_document(self, file_path: str, llm_classifier=None) -> Tuple[Dict, Dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        blocks, tables, meta = self._extract_blocks(lines)
        blocks, strategy = self._classify_blocks(blocks, llm_classifier=llm_classifier)
        structured = self._build_section_content(blocks)
        structured["Tables"] = tables
        detected_sections = {
            category
            for category in self.TARGET_CATEGORIES
            if str(structured.get(category, "")).strip()
        }
        metadata = {
            "source_file": file_path,
            "strategy": strategy,
            "sections_detected": sorted(detected_sections),
            "required_sections_found": sorted(self.REQUIRED_CORE.intersection(detected_sections)),
            "required_sections_complete": self.REQUIRED_CORE.issubset(detected_sections),
            "page_info": meta,
            "block_count": len(blocks),
            "table_count": len(tables),
            "blocks": [{k: v for k, v in b.items() if k != "content"} for b in blocks],
        }
        return structured, metadata
    
    def segment_document_enhanced(self, 
                                   file_path: str, 
                                   llm_client=None,
                                   model_name: str = "deepseek-chat") -> Tuple[Dict, Dict]:
        """
        Enhanced document segmentation using extended rules, content features, and LLM.
        
        Args:
            file_path: Path to markdown file
            llm_client: LLM client instance (optional)
            model_name: Model name for LLM classification
            
        Returns:
            Tuple[Dict, Dict]: (structured content, metadata)
        """
        from .enhanced_segmentation import EnhancedSegmenter
        
        enhanced = EnhancedSegmenter()
        return enhanced.segment_document(file_path, llm_client, model_name)

    def validate_sections(self, document_structure: Dict) -> Dict:
        has_m = bool(str(document_structure.get("Methods", "")).strip())
        has_r = bool(str(document_structure.get("Results", "")).strip())
        return {
            "has_methods": has_m,
            "has_results": has_r,
            "missing_sections": [x for x, ok in [("Methods", has_m), ("Results", has_r)] if not ok],
            "warnings": [] if has_m and has_r else ["Missing required sections"],
        }

    def save_segmentation_results(self, document_structure: Dict, tables: List, tables_info: List, tables_annotation: List, output_dir: str, filename: str = "segmentation_results.json") -> str:
        os.makedirs(output_dir, exist_ok=True)
        p = os.path.join(output_dir, filename)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({"document_structure": document_structure, "tables": tables, "tables_info": tables_info, "tables_annotation": tables_annotation}, f, ensure_ascii=False, indent=2)
        return p

    def save_structured_outputs(self, structured: Dict, metadata: Dict, output_dir: str, base_name: str) -> Tuple[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        content_path = os.path.join(output_dir, f"{base_name}_structured_content.json")
        meta_path = os.path.join(output_dir, f"{base_name}_structured_meta.json")
        with open(content_path, "w", encoding="utf-8") as f:
            json.dump(structured, f, ensure_ascii=False, indent=2)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return content_path, meta_path
