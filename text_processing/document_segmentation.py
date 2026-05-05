"""Document segmentation module for ARneuro."""

import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from core.logger import get_logger

logger = get_logger(__name__)


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
        section_re = re.compile(r"^(#{1,6})\s+(.*)")
        total_pages_re = re.compile(r"\*\*Total Pages:\*\*\s*(\d+)", re.I)
        sep_re = re.compile(r"^={10,}$|^-{10,}$")
        html_start_re = re.compile(r"<table\b[^>]*>", re.I)
        html_end_re = re.compile(r"</table>", re.I)

        blocks, tables = [], []
        meta = {"declared_total_pages": None, "parsed_page_markers": 0}
        current = None
        i = 0
        while i < len(lines):
            raw = lines[i].rstrip("\n")
            s = raw.strip()
            if not s or sep_re.match(s):
                i += 1
                continue
            m = total_pages_re.search(s)
            if m:
                meta["declared_total_pages"] = int(m.group(1))
                i += 1
                continue
            if self._is_page_marker(s):
                meta["parsed_page_markers"] += 1
                i += 1
                continue
            if html_start_re.match(s):
                t_lines = []
                start = i
                while i < len(lines):
                    t_lines.append(lines[i].rstrip("\n"))
                    if html_end_re.search(lines[i]):
                        i += 1
                        break
                    i += 1
                tables.append({"id": f"table_{len(tables)+1}", "content": "\n".join(t_lines), "line_start": start + 1})
                if current:
                    current["content_lines"].append(f"[TABLE: table_{len(tables)}]")
                continue
            if s.startswith("|"):
                t_lines = []
                start = i
                while i < len(lines) and lines[i].strip().startswith("|"):
                    t_lines.append(lines[i].rstrip("\n"))
                    i += 1
                tables.append({"id": f"table_{len(tables)+1}", "content": "\n".join(t_lines), "line_start": start + 1})
                if current:
                    current["content_lines"].append(f"[TABLE: table_{len(tables)}]")
                continue

            head = section_re.match(s)
            if head:
                if current:
                    blocks.append(current)
                current = {
                    "heading": head.group(2).strip(),
                    "heading_level": len(head.group(1)),
                    "line_start": i + 1,
                    "content_lines": [],
                }
            else:
                if current is None:
                    current = {"heading": "Document_Root", "heading_level": 0, "line_start": 1, "content_lines": []}
                current["content_lines"].append(raw)
            i += 1
        if current:
            blocks.append(current)
        for b in blocks:
            b["content"] = "\n".join([x for x in b["content_lines"] if x.strip()]).strip()
            b.pop("content_lines", None)
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
            by_cat[b["category"]].append(b)
        for cat, arr in by_cat.items():
            ordered = sorted(arr, key=lambda x: x["line_start"])
            out[cat] = "\n\n".join([f"### {x['heading']}\n{x['content']}".strip() for x in ordered if x.get("content") or x.get("heading")])

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
        document_structure = {b["heading"]: b["content"] for b in blocks if b.get("content")}
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
        metadata = {
            "source_file": file_path,
            "strategy": strategy,
            "sections_detected": sorted({b["category"] for b in blocks}),
            "required_sections_found": sorted(self.REQUIRED_CORE.intersection({b["category"] for b in blocks})),
            "required_sections_complete": self.REQUIRED_CORE.issubset({b["category"] for b in blocks}),
            "page_info": meta,
            "block_count": len(blocks),
            "table_count": len(tables),
            "blocks": [{k: v for k, v in b.items() if k != "content"} for b in blocks],
        }
        return structured, metadata

    def validate_sections(self, document_structure: Dict) -> Dict:
        text = " ".join(document_structure.keys()).lower()
        has_m = "method" in text
        has_r = "result" in text
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
