"""Read and validate PubMed exports from Excel, CSV, or TSV files."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pandas as pd


CANONICAL_COLUMNS = ("PMID", "Title", "Abstract", "DOI", "PMCID", "Authors", "Journal", "Year")
COLUMN_ALIASES = {
    "PMID": ("pmid", "pubmed id", "pubmedid", "id"),
    "Title": ("title", "article title"),
    "Abstract": ("abstract", "abstract text"),
    "DOI": ("doi", "digital object identifier"),
    "PMCID": ("pmcid", "pmc id", "pmc"),
    "Authors": ("authors", "author", "author list"),
    "Journal": ("journal", "journal/book", "source"),
    "Year": ("year", "publication year", "pub date"),
}


def _normalise_column(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


def _normalise_pmid(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text if text.isdigit() else ""


@dataclass(frozen=True)
class InputArticle:
    pmid: str
    title: str = ""
    abstract: str = ""
    doi: str = ""
    pmcid: str = ""
    authors: str = ""
    journal: str = ""
    year: str = ""
    source_row: int = 0

    def to_row(self) -> Dict[str, str | int]:
        row = asdict(self)
        return {"PMID" if key == "pmid" else key: value for key, value in row.items()}


def read_pubmed_export(path: str | Path) -> Tuple[List[InputArticle], Dict[str, Any]]:
    """Return de-duplicated canonical records and a transparent validation report."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Input export not found: {source}")
    suffix = source.suffix.casefold()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(source, dtype=str)
    elif suffix == ".tsv":
        frame = pd.read_csv(source, sep="\t", dtype=str)
    else:
        frame = pd.read_csv(source, dtype=str, sep=None, engine="python")
    frame = frame.fillna("")
    source_columns = { _normalise_column(column): str(column) for column in frame.columns }
    mapped: Dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for candidate in aliases:
            if candidate in source_columns:
                mapped[canonical] = source_columns[candidate]
                break
    if "PMID" not in mapped:
        raise ValueError("PubMed export requires a PMID column (for example PMID or PubMed ID).")

    records: List[InputArticle] = []
    invalid_rows: List[int] = []
    duplicate_pmids: List[str] = []
    seen: set[str] = set()
    for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
        pmid = _normalise_pmid(row.get(mapped["PMID"], ""))
        if not pmid:
            invalid_rows.append(row_number)
            continue
        if pmid in seen:
            duplicate_pmids.append(pmid)
            continue
        seen.add(pmid)
        values = {canonical.casefold(): str(row.get(column, "") or "").strip() for canonical, column in mapped.items() if canonical != "PMID"}
        records.append(InputArticle(pmid=pmid, source_row=row_number, **values))
    report = {
        "source": str(source.resolve()),
        "source_rows": int(len(frame)),
        "accepted_records": len(records),
        "invalid_pmid_rows": invalid_rows,
        "duplicate_pmids": duplicate_pmids,
        "column_mapping": mapped,
    }
    return records, report


def validate_pubmed_export(path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    """Write canonical records and a report. This stage never contacts PubMed."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records, report = read_pubmed_export(path)
    canonical_path = output / "canonical_articles.csv"
    pd.DataFrame([record.to_row() for record in records]).to_csv(canonical_path, index=False, encoding="utf-8-sig")
    report["canonical_articles"] = str(canonical_path)
    (output / "input_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
