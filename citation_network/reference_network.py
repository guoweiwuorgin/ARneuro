"""Local-first, provenance-preserving citation network construction.

The builder keeps the citation graph separate from the text corpus SQLite file.
This permits fast rebuilds and protects the corpus while task annotations change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import networkx as nx
import requests
from rapidfuzz.fuzz import token_set_ratio


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
PMID_RE = re.compile(r"\b(?:PMID|PUBMED\s*(?:ID|NO\.?)?)\s*[:#]?\s*(\d{5,9})\b", re.IGNORECASE)
YEAR_RE = re.compile(r"\b((?:18|19|20)\d{2})\b")
NUMBERED_LINE_RE = re.compile(r"(?m)(?:^|\n)\s*(?:\[\s*)?(\d{1,4})(?:\s*\]|[.)])\s+")
NUMBERED_INLINE_RE = re.compile(r"(?<!\w)(?:\[\s*)?(\d{1,4})(?:\s*\]|\.)\s+(?=[A-Z])")
HEADING_RE = re.compile(r"(?im)^#{1,6}\s*references?\s*$")
WORD_RE = re.compile(r"[a-z0-9]+")
BAD_TITLE_RE = re.compile(r"^#{0,3}\s*paper_\d+\b", re.IGNORECASE)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


def normalize_doi(value: str) -> str:
    return re.sub(r"[.);,]+$", "", str(value or "").strip().lower())


def title_key(value: str) -> str:
    return " ".join(WORD_RE.findall(str(value or "").lower()))


def raw_hash(value: str) -> str:
    return hashlib.sha256(normalize_space(value).encode("utf-8")).hexdigest()


def reference_key_from_fields(doi: str, title: str, year: str, raw: str) -> str:
    if doi:
        return f"doi:{doi}"
    normalized_title = title_key(title)
    if normalized_title:
        return f"title:{normalized_title}|{year}"
    return f"raw:{raw_hash(raw)}"


def split_reference_entries(text: str) -> List[Tuple[int, str]]:
    """Split a References section while retaining the original entry text.

    Numbered entries on separate lines are preferred. A conservative inline fallback
    handles OCR output in which an entire reference list was collapsed to one line.
    """
    text = str(text or "").replace("\r", "\n")
    text = HEADING_RE.sub("", text).strip()
    matches = list(NUMBERED_LINE_RE.finditer(text))
    if len(matches) < 2:
        matches = list(NUMBERED_INLINE_RE.finditer(text))
    if len(matches) < 2:
        return [(1, normalize_space(text))] if normalize_space(text) else []

    entries: List[Tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        entry = normalize_space(text[match.end() : end])
        if len(entry) >= 12:
            try:
                number = int(match.group(1))
            except ValueError:
                number = index + 1
            entries.append((number, entry))
    return entries


def split_title_parts(before_year: str) -> List[str]:
    """Return plausible post-author title fragments from a conventional citation."""
    parts = re.split(r"\.\s+(?=[A-Z][A-Za-z]{2,})", before_year)
    parts = [normalize_space(item) for item in parts if len(normalize_space(item)) >= 4]
    return parts[1:] if len(parts) > 1 else []


def parse_reference(raw: str) -> Dict[str, str]:
    cleaned = normalize_space(raw)
    doi_match = DOI_RE.search(cleaned)
    pmid_match = PMID_RE.search(cleaned)
    year_match = YEAR_RE.search(cleaned)
    year = year_match.group(1) if year_match else ""
    before_year = cleaned[: year_match.start()] if year_match else cleaned
    # Many older clinical references use "Authors: Title. Journal Year;...".
    # This is more reliable than generic sentence splitting when the colon occurs
    # before the publication year and leaves a substantive phrase behind it.
    after_year = cleaned[year_match.end() :].lstrip(" )];,:.") if year_match else ""
    if after_year and len(after_year) >= 12 and cleaned[year_match.start() - 1 : year_match.start()] == "(":
        # Author-year references: "Authors (1991) Article title. Journal ...".
        title = after_year.split(". ", 1)[0]
    else:
        title = ""
    colon_match = re.search(r":\s+(.{12,})$", before_year)
    if not title and colon_match:
        after_colon = colon_match.group(1)
        title = after_colon.split(". ", 1)[0]
    elif not title:
        candidates = split_title_parts(before_year)
        title = max(candidates, key=lambda value: (len(value.split()), len(value)), default="")
    title = re.sub(r"\s*(?:\[?in press\]?|doi\s*:.*)$", "", title, flags=re.IGNORECASE).strip(" .;,")
    first_author = ""
    author_match = re.match(r"([A-Z][A-Za-z'\-]+)", cleaned)
    if author_match:
        first_author = author_match.group(1)
    doi = normalize_doi(doi_match.group(0)) if doi_match else ""
    pmid = pmid_match.group(1) if pmid_match else ""
    status = "identifier" if doi or pmid else ("title_year" if title and year else "partial")
    return {
        "raw_reference": cleaned,
        "doi": doi,
        "pmid": pmid,
        "year": year,
        "title": title,
        "title_key": title_key(title),
        "first_author": first_author,
        "parse_status": status,
    }


@dataclass
class CitationNetworkBuildConfig:
    corpus_database: Path
    output_dir: Path
    database_name: str = "citation_network.sqlite3"
    pubmed_cache_name: str = "pubmed_metadata_cache.json"
    bib_cache_name: str = "deepseek_bib_cache.jsonl"
    fetch_pubmed_metadata: bool = False
    ncbi_tool: str = "ARneuroCitationNetwork"
    ncbi_email: str = ""
    request_timeout_seconds: int = 45
    pubmed_batch_size: int = 200

    @property
    def database_path(self) -> Path:
        return self.output_dir / self.database_name

    @property
    def pubmed_cache_path(self) -> Path:
        return self.output_dir / self.pubmed_cache_name

    @property
    def bib_cache_path(self) -> Path:
        return self.output_dir / self.bib_cache_name


class CitationNetworkBuilder:
    def __init__(self, config: CitationNetworkBuildConfig) -> None:
        self.config = config

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(str(path))
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=DELETE")
        return connection

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE works (
                work_key TEXT PRIMARY KEY,
                is_corpus_work INTEGER NOT NULL DEFAULT 0,
                pmid TEXT NOT NULL DEFAULT '',
                doi TEXT NOT NULL DEFAULT '',
                title TEXT NOT NULL DEFAULT '',
                normalized_title TEXT NOT NULL DEFAULT '',
                publication_year TEXT NOT NULL DEFAULT '',
                first_author TEXT NOT NULL DEFAULT '',
                provenance_json TEXT NOT NULL DEFAULT ''
            );
            CREATE UNIQUE INDEX idx_works_pmid ON works(pmid) WHERE pmid <> '';
            CREATE INDEX idx_works_doi ON works(doi);
            CREATE INDEX idx_works_title ON works(normalized_title);

            CREATE TABLE reference_records (
                reference_id INTEGER PRIMARY KEY,
                citing_pmid TEXT NOT NULL,
                reference_order INTEGER NOT NULL,
                raw_reference TEXT NOT NULL,
                raw_hash TEXT NOT NULL,
                doi TEXT NOT NULL DEFAULT '',
                cited_pmid TEXT NOT NULL DEFAULT '',
                publication_year TEXT NOT NULL DEFAULT '',
                parsed_title TEXT NOT NULL DEFAULT '',
                normalized_title TEXT NOT NULL DEFAULT '',
                first_author TEXT NOT NULL DEFAULT '',
                parse_status TEXT NOT NULL,
                bibtex TEXT NOT NULL DEFAULT '',
                bib_source TEXT NOT NULL DEFAULT '',
                target_work_key TEXT NOT NULL DEFAULT '',
                target_corpus_pmid TEXT NOT NULL DEFAULT '',
                match_method TEXT NOT NULL DEFAULT '',
                match_confidence REAL NOT NULL DEFAULT 0,
                UNIQUE(citing_pmid, reference_order, raw_hash)
            );
            CREATE INDEX idx_reference_citing ON reference_records(citing_pmid);
            CREATE INDEX idx_reference_doi ON reference_records(doi);
            CREATE INDEX idx_reference_target ON reference_records(target_work_key);

            CREATE TABLE citation_edges (
                citing_pmid TEXT NOT NULL,
                target_work_key TEXT NOT NULL,
                target_corpus_pmid TEXT NOT NULL DEFAULT '',
                reference_count INTEGER NOT NULL DEFAULT 1,
                max_match_confidence REAL NOT NULL DEFAULT 0,
                PRIMARY KEY(citing_pmid, target_work_key)
            );
            CREATE INDEX idx_edges_internal_target ON citation_edges(target_corpus_pmid);

            CREATE TABLE citation_metrics (
                work_key TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                PRIMARY KEY(work_key, metric_name)
            );
            CREATE TABLE citation_communities (
                pmid TEXT PRIMARY KEY,
                community_id INTEGER NOT NULL
            );
            """
        )

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default

    @staticmethod
    def _load_bib_cache(path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        records: Dict[str, Dict[str, Any]] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            digest = str(row.get("raw_hash", ""))
            if digest:
                records[digest] = row
        return records

    def _fetch_pubmed_metadata(self, pmids: Sequence[str]) -> Dict[str, Dict[str, str]]:
        cache = self._load_json(self.config.pubmed_cache_path, {})
        cache = cache if isinstance(cache, dict) else {}
        missing = [pmid for pmid in pmids if pmid not in cache]
        if not self.config.fetch_pubmed_metadata or not missing:
            return {pmid: dict(cache.get(pmid, {})) for pmid in pmids}

        session = requests.Session()
        endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
        for start in range(0, len(missing), self.config.pubmed_batch_size):
            batch = missing[start : start + self.config.pubmed_batch_size]
            params = {
                "db": "pubmed",
                "retmode": "json",
                "id": ",".join(batch),
                "tool": self.config.ncbi_tool,
            }
            if self.config.ncbi_email:
                params["email"] = self.config.ncbi_email
            response = session.get(endpoint, params=params, timeout=self.config.request_timeout_seconds)
            response.raise_for_status()
            payload = response.json().get("result", {})
            for pmid in batch:
                item = payload.get(str(pmid), {})
                article_ids = item.get("articleids", []) if isinstance(item, dict) else []
                doi = ""
                for identifier in article_ids:
                    if str(identifier.get("idtype", "")).lower() == "doi":
                        doi = normalize_doi(identifier.get("value", ""))
                        break
                authors = item.get("authors", []) if isinstance(item, dict) else []
                first_author = str(authors[0].get("name", "")) if authors else str(item.get("sortfirstauthor", ""))
                pubdate = str(item.get("pubdate", item.get("epubdate", "")))
                year_match = YEAR_RE.search(pubdate)
                cache[str(pmid)] = {
                    "title": normalize_space(item.get("title", "")),
                    "year": year_match.group(1) if year_match else "",
                    "first_author": first_author,
                    "doi": doi,
                    "journal": normalize_space(item.get("fulljournalname", "")),
                    "source": "ncbi_esummary",
                }
            self.config.pubmed_cache_path.write_text(
                json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
            )
            time.sleep(0.35)
        return {pmid: dict(cache.get(pmid, {})) for pmid in pmids}

    @staticmethod
    def _iter_reference_texts(corpus: sqlite3.Connection) -> Iterator[Tuple[str, str]]:
        query = """
            SELECT pmid, content FROM sections
            WHERE lower(section_name) = 'references'
            UNION ALL
            SELECT pmid, content FROM sections
            WHERE lower(section_name) = 'other'
              AND lower(content) LIKE '%references%'
        """
        seen: set[Tuple[str, str]] = set()
        for row in corpus.execute(query):
            pmid, content = str(row["pmid"]), str(row["content"])
            digest = raw_hash(content)
            if (pmid, digest) not in seen:
                seen.add((pmid, digest))
                yield pmid, content

    @staticmethod
    def _match_reference(
        reference: Mapping[str, str],
        by_pmid: Mapping[str, Dict[str, str]],
        by_doi: Mapping[str, Dict[str, str]],
        by_title: Mapping[str, List[Dict[str, str]]],
        token_index: Mapping[str, set[str]],
    ) -> Tuple[Optional[Dict[str, str]], str, float]:
        cited_pmid = reference.get("pmid", "")
        if cited_pmid and cited_pmid in by_pmid:
            return by_pmid[cited_pmid], "reference_pmid", 1.0
        doi = reference.get("doi", "")
        if doi and doi in by_doi:
            return by_doi[doi], "doi", 1.0
        normalized_title = reference.get("title_key", "")
        if normalized_title and normalized_title in by_title:
            candidates = by_title[normalized_title]
            if len(candidates) == 1:
                return candidates[0], "title_exact", 0.99

        tokens = [token for token in normalized_title.split() if len(token) >= 5]
        scores: Counter[str] = Counter()
        for token in tokens:
            scores.update(token_index.get(token, set()))
        candidate_pmids = [pmid for pmid, overlap in scores.most_common(25) if overlap >= 2]
        best: Optional[Dict[str, str]] = None
        best_score = 0.0
        for pmid in candidate_pmids:
            candidate = by_pmid[pmid]
            score = token_set_ratio(normalized_title, candidate["normalized_title"])
            year = reference.get("year", "")
            year_agrees = not year or not candidate.get("year") or abs(int(year) - int(candidate["year"])) <= 1
            author = reference.get("first_author", "").casefold()
            author_agrees = not author or author == candidate.get("first_author", "").casefold()
            if score >= 96 and year_agrees and (author_agrees or score >= 99):
                if score > best_score:
                    best, best_score = candidate, float(score)
        if best:
            return best, "title_fuzzy", round(best_score / 100.0, 3)
        return None, "", 0.0

    @staticmethod
    def _write_metrics(connection: sqlite3.Connection) -> Dict[str, int]:
        rows = connection.execute(
            "SELECT citing_pmid, target_work_key, target_corpus_pmid FROM citation_edges"
        ).fetchall()
        total_incoming: Counter[str] = Counter(row["target_work_key"] for row in rows)
        internal = nx.DiGraph()
        corpus_pmids = [row[0] for row in connection.execute("SELECT pmid FROM works WHERE is_corpus_work=1")]
        internal.add_nodes_from(corpus_pmids)
        internal.add_edges_from(
            (row["citing_pmid"], row["target_corpus_pmid"])
            for row in rows
            if row["target_corpus_pmid"]
        )
        metric_rows: List[Tuple[str, str, float]] = []
        for work_key, count in total_incoming.items():
            metric_rows.append((work_key, "local_in_degree", float(count)))
        if internal.number_of_edges():
            pagerank = nx.pagerank(internal, alpha=0.85, max_iter=300)
            for pmid, value in pagerank.items():
                metric_rows.append((f"pmid:{pmid}", "internal_pagerank", float(value)))
            approximate_k = min(128, internal.number_of_nodes())
            if approximate_k >= 2:
                betweenness = nx.betweenness_centrality(internal, k=approximate_k, seed=42)
                for pmid, value in betweenness.items():
                    metric_rows.append((f"pmid:{pmid}", "internal_betweenness_approx", float(value)))
            undirected = internal.to_undirected()
            try:
                communities = nx.community.louvain_communities(undirected, seed=42)
            except Exception:
                communities = list(nx.community.greedy_modularity_communities(undirected))
            connection.executemany(
                "INSERT INTO citation_communities(pmid, community_id) VALUES (?, ?)",
                [(pmid, community_id) for community_id, group in enumerate(communities) for pmid in group],
            )
        connection.executemany(
            "INSERT INTO citation_metrics(work_key, metric_name, metric_value) VALUES (?, ?, ?)", metric_rows
        )
        return {"internal_nodes": internal.number_of_nodes(), "internal_edges": internal.number_of_edges()}

    def build(self) -> Dict[str, Any]:
        config = self.config
        if not config.corpus_database.exists():
            raise FileNotFoundError(f"Corpus database not found: {config.corpus_database}")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = config.database_path.with_name(config.database_path.stem + ".building.sqlite3")
        if temporary.exists():
            temporary.unlink()
        bib_cache = self._load_bib_cache(config.bib_cache_path)

        corpus = sqlite3.connect(config.corpus_database)
        corpus.row_factory = sqlite3.Row
        documents = corpus.execute("SELECT pmid, title FROM documents ORDER BY pmid").fetchall()
        pmids = [str(row["pmid"]) for row in documents]
        pubmed = self._fetch_pubmed_metadata(pmids)

        connection = self._connect(temporary)
        self._create_schema(connection)
        by_pmid: Dict[str, Dict[str, str]] = {}
        by_doi: Dict[str, Dict[str, str]] = {}
        by_title: Dict[str, List[Dict[str, str]]] = defaultdict(list)
        token_index: Dict[str, set[str]] = defaultdict(set)
        for row in documents:
            pmid = str(row["pmid"])
            metadata = pubmed.get(pmid, {})
            raw_title = normalize_space(metadata.get("title") or row["title"])
            title = "" if BAD_TITLE_RE.search(raw_title) else raw_title
            record = {
                "pmid": pmid,
                "doi": normalize_doi(metadata.get("doi", "")),
                "title": title,
                "normalized_title": title_key(title),
                "year": str(metadata.get("year", "")),
                "first_author": normalize_space(metadata.get("first_author", "")).split(" ")[0],
            }
            by_pmid[pmid] = record
            if record["doi"]:
                by_doi[record["doi"]] = record
            if record["normalized_title"]:
                by_title[record["normalized_title"]].append(record)
                for token in set(token for token in record["normalized_title"].split() if len(token) >= 5):
                    token_index[token].add(pmid)
            connection.execute(
                """INSERT INTO works(work_key,is_corpus_work,pmid,doi,title,normalized_title,publication_year,first_author,provenance_json)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (
                    f"pmid:{pmid}", 1, pmid, record["doi"], title, record["normalized_title"], record["year"],
                    record["first_author"], json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                ),
            )

        counters: Counter[str] = Counter()
        for citing_pmid, reference_text in self._iter_reference_texts(corpus):
            entries = split_reference_entries(reference_text)
            if len(entries) == 1 and len(entries[0][1]) > 1200:
                counters["suspect_unsplit_sections"] += 1
            for reference_order, raw in entries:
                parsed = parse_reference(raw)
                override = bib_cache.get(raw_hash(raw), {})
                if override:
                    parsed["doi"] = normalize_doi(override.get("doi", parsed["doi"]))
                    parsed["pmid"] = str(override.get("pmid", parsed["pmid"]))
                    parsed["year"] = str(override.get("year", parsed["year"]))
                    parsed["title"] = normalize_space(override.get("title", parsed["title"]))
                    parsed["title_key"] = title_key(parsed["title"])
                    parsed["first_author"] = normalize_space(override.get("first_author", parsed["first_author"]))
                    parsed["parse_status"] = "deepseek_bib"
                target, method, confidence = self._match_reference(parsed, by_pmid, by_doi, by_title, token_index)
                if target:
                    target_key = f"pmid:{target['pmid']}"
                    target_pmid = target["pmid"]
                    counters[f"match_{method}"] += 1
                else:
                    target_key = reference_key_from_fields(parsed["doi"], parsed["title"], parsed["year"], raw)
                    target_pmid = ""
                    connection.execute(
                        """INSERT OR IGNORE INTO works(work_key,is_corpus_work,doi,title,normalized_title,publication_year,first_author,provenance_json)
                           VALUES(?,?,?,?,?,?,?,?)""",
                        (target_key, 0, parsed["doi"], parsed["title"], parsed["title_key"], parsed["year"], parsed["first_author"], ""),
                    )
                    counters["unmatched_external"] += 1
                connection.execute(
                    """INSERT OR IGNORE INTO reference_records(
                       citing_pmid,reference_order,raw_reference,raw_hash,doi,cited_pmid,publication_year,parsed_title,
                       normalized_title,first_author,parse_status,bibtex,bib_source,target_work_key,target_corpus_pmid,
                       match_method,match_confidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        citing_pmid, reference_order, parsed["raw_reference"], raw_hash(raw), parsed["doi"], parsed["pmid"],
                        parsed["year"], parsed["title"], parsed["title_key"], parsed["first_author"], parsed["parse_status"],
                        str(override.get("bibtex", "")), "deepseek" if override else "", target_key, target_pmid, method, confidence,
                    ),
                )
                counters["reference_records"] += 1

        connection.execute(
            """INSERT INTO citation_edges(citing_pmid,target_work_key,target_corpus_pmid,reference_count,max_match_confidence)
               SELECT citing_pmid,target_work_key,target_corpus_pmid,COUNT(*),MAX(match_confidence)
               FROM reference_records GROUP BY citing_pmid,target_work_key,target_corpus_pmid"""
        )
        # Counters accumulated during parsing include harmless duplicate OCR entries;
        # report the persisted, deduplicated values as the official network totals.
        counters["reference_records"] = int(connection.execute("SELECT COUNT(*) FROM reference_records").fetchone()[0])
        counters["unmatched_external"] = int(
            connection.execute("SELECT COUNT(*) FROM reference_records WHERE target_corpus_pmid='' ").fetchone()[0]
        )
        counters["citation_edges"] = int(connection.execute("SELECT COUNT(*) FROM citation_edges").fetchone()[0])
        metric_counts = self._write_metrics(connection)
        metadata = {
            "built_at_utc": utc_now(),
            "corpus_database": str(config.corpus_database),
            "fetch_pubmed_metadata": config.fetch_pubmed_metadata,
            "corpus_documents": len(documents),
            **dict(counters),
            **metric_counts,
        }
        connection.executemany("INSERT INTO metadata(key,value) VALUES(?,?)", [(key, json.dumps(value)) for key, value in metadata.items()])
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        corpus.close()
        temporary.replace(config.database_path)
        self._export_top_works(metadata)
        (config.output_dir / "citation_network_summary.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metadata

    def _export_top_works(self, metadata: Mapping[str, Any]) -> None:
        connection = self._connect(self.config.database_path)
        rows = connection.execute(
            """SELECT w.work_key,w.pmid,w.doi,w.title,w.publication_year,w.first_author,
                      MAX(CASE WHEN m.metric_name='local_in_degree' THEN m.metric_value END) AS local_in_degree,
                      MAX(CASE WHEN m.metric_name='internal_pagerank' THEN m.metric_value END) AS internal_pagerank,
                      MAX(CASE WHEN m.metric_name='internal_betweenness_approx' THEN m.metric_value END) AS internal_betweenness_approx
               FROM works w LEFT JOIN citation_metrics m ON w.work_key=m.work_key
               GROUP BY w.work_key ORDER BY local_in_degree DESC, internal_pagerank DESC LIMIT 5000"""
        ).fetchall()
        output = self.config.output_dir / "top_cited_works.csv"
        with output.open("w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=rows[0].keys() if rows else ["work_key"], delimiter=";")
            writer.writeheader()
            writer.writerows([dict(row) for row in rows])
        connection.close()

    def unparsed_references(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        connection = self._connect(self.config.database_path)
        query = """SELECT MIN(reference_id) AS reference_id,MIN(citing_pmid) AS citing_pmid,
                          MIN(reference_order) AS reference_order,MIN(raw_reference) AS raw_reference,raw_hash
                   FROM reference_records
                   WHERE parse_status='partial' AND doi='' AND cited_pmid='' AND parsed_title=''
                     AND length(raw_reference) BETWEEN 80 AND 1200
                     AND raw_reference GLOB '*[0-9][0-9][0-9][0-9]*'
                     AND lower(raw_reference) NOT LIKE 'crossref%'
                     AND lower(raw_reference) NOT LIKE 'received %'
                     AND raw_reference NOT LIKE '%<!--%'
                   GROUP BY raw_hash
                   ORDER BY citing_pmid,reference_order"""
        if limit:
            query += f" LIMIT {int(limit)}"
        rows = [dict(row) for row in connection.execute(query)]
        connection.close()
        return rows


class DeepSeekBibFormatter:
    """Cache-only DeepSeek fallback for references that local parsing cannot identify."""

    def __init__(self, cache_path: Path, api_key: str, model_name: str = "deepseek-v4-flash") -> None:
        self.cache_path = cache_path
        self.api_key = api_key
        self.model_name = model_name

    @staticmethod
    def _messages(entries: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
        prompt = """
You are a bibliographic parser. Convert each raw reference into conservative BibTeX.
Use only facts present in the raw reference. Never invent authors, title, journal, year,
volume, pages, DOI, or PMID. If a field is absent, omit it. Return valid JSON only:
{"entries":[{"raw_hash":"...","bibtex":"@article{...}","title":"","year":"",
"doi":"","pmid":"","first_author":"","confidence":0.0}]}
For incomplete references use @misc with note containing the original reference. BibTeX
must be syntactically valid and include the original reference in note when confidence <0.8.
""".strip()
        return [{"role": "system", "content": prompt}, {"role": "user", "content": json.dumps({"entries": entries}, ensure_ascii=False)}]

    def format_entries(self, entries: Sequence[Mapping[str, str]], batch_size: int = 12) -> int:
        if not self.api_key:
            raise ValueError("DeepSeek API key is required for BibTeX fallback.")
        try:
            from ARneuro.core.llm_client import LLMClientManager
        except ImportError:
            from core.llm_client import LLMClientManager
        existing = CitationNetworkBuilder._load_bib_cache(self.cache_path)
        pending = [entry for entry in entries if entry["raw_hash"] not in existing]
        if not pending:
            return 0
        manager = LLMClientManager({"deepseek_api_key": self.api_key, "deepseek_model_name": self.model_name})
        client, model = manager.get_client("deepseek", model_name=self.model_name)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        written = 0
        with self.cache_path.open("a", encoding="utf-8") as file_obj:
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                response = client.chat.completions.create(
                    model=model or self.model_name,
                    messages=self._messages(batch),
                    temperature=0,
                    max_tokens=3500,
                    response_format={"type": "json_object"},
                    extra_body={"thinking": {"type": "disabled"}},
                )
                payload = json.loads(response.choices[0].message.content or "{}")
                by_hash = {str(item.get("raw_hash", "")): item for item in payload.get("entries", [])}
                for entry in batch:
                    result = by_hash.get(entry["raw_hash"], {})
                    record = {"raw_hash": entry["raw_hash"], "raw_reference": entry["raw_reference"], **result, "model": model or self.model_name}
                    file_obj.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                file_obj.flush()
                time.sleep(0.8)
        return written
