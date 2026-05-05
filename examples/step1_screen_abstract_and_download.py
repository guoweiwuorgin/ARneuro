"""步骤1：CSV -> 合法渠道获取Abstract -> LLM初筛 -> 下载纳入文献PDF。
支持断点续跑：每处理完一个PMID立即保存checkpoint，重启后自动跳过已完成条目。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
import sys

sys.path.append("/storage/work/wuguowei/reviewer/Code_tools/")
import os

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

from ARneuro.core.llm_client import LLMClientManager
from ARneuro.data_fetch.pubmed_fetcher import PubMedFetcher
from ARneuro.data_fetch.pdf_downloader import PDFDownloader
from ARneuro.feature_extraction.abstract_screening import ReviewCriteriaLoader, ScreeningCriteria


@dataclass
class Step1Decision:
    pmid: str
    title: str
    abstract: str
    include: bool
    confidence: float
    reason: str
    labels: Dict[str, Any]
    pdf_path: str = ""
    download_status: str = "skipped"


class LegalAbstractProvider:
    """仅通过合法公开渠道（PubMed API via metapub）获取abstract。"""

    def __init__(self, fetcher: PubMedFetcher):
        self.fetcher = fetcher

    def fetch_abstract(self, pmid: str) -> Optional[str]:
        records = self.fetcher.fetch_abstracts([pmid], delay=0.1)
        if not records:
            return None
        return records[0].abstract


class CriteriaLLMJudge:
    def __init__(
        self,
        llm_manager: LLMClientManager,
        client_type: str = "deepseek",
        model_name: str = None,
        api_key: str = None
    ):
        self.llm_manager = llm_manager
        self.client_type = client_type
        self.model_name = model_name
        self.api_key = api_key

    def evaluate(self, criteria: ScreeningCriteria, abstract: str, pmid: str) -> Dict[str, Any]:
        client, model = self.llm_manager.get_client(client_type=self.client_type, model_name=self.model_name, api_key=self.api_key)
        prompt = self._build_prompt(criteria, abstract, pmid)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a strict literature screening assistant."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1200,
        )
        return json.loads(response.choices[0].message.content)

    @staticmethod
    def _build_prompt(criteria: ScreeningCriteria, abstract: str, pmid: str) -> str:
        payload = {
            "pmid": pmid,
            "criteria": {
                "review_name": criteria.review_name,
                "research_field": criteria.research_field,
                "research_topic_statement": criteria.research_topic_statement,
                "methodology_include": criteria.methodology_include,
                "methodology_exclude": criteria.methodology_exclude,
                "participant_scope": criteria.participant_scope,
                "disease_and_case_report_policy": criteria.disease_and_case_report_policy,
                "publication_filters": criteria.publication_filters,
                "priority_signals": criteria.priority_signals,
                "exclusion_signals": criteria.exclusion_signals,
                "decision_policy": criteria.decision_policy,
            },
            "abstract": abstract,
            "output_schema": {
                "include": "bool",
                "confidence": "0~1 float",
                "reason": "short sentence",
                "labels": {
                    "topic_fit": "yes/no/uncertain",
                    "method_fit": "yes/no/uncertain",
                    "participant_fit": "yes/no/uncertain",
                    "risk_flags": ["list of risk terms"],
                },
            },
        }
        return (
            "Evaluate whether this abstract should be included. "
            "Return ONLY JSON, do not add markdown.\n\n"
            f"{json.dumps(payload, ensure_ascii=False)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="步骤1：abstract筛选并下载PDF（支持断点续跑）")
    parser.add_argument("--csv-file", default="./data/ARneuro_test.csv", help="输入CSV（至少包含PMID列）")
    parser.add_argument("--criteria-yaml", default="../config/review_screening_criteria_template.yaml", help="筛选标准YAML")
    parser.add_argument("--output-dir", default="./data/workflow/step1", help="步骤1输出目录")
    parser.add_argument("--llm-client-type", default="deepseek", choices=["deepseek", "gpt4", "glm", "huoshan"])
    parser.add_argument("--llm-model-name", default="deepseek-chat")
    parser.add_argument("--llm-apikey", default="sk-e97ed8cba5234d64b19b51d3696a10d0")
    parser.add_argument("--config-path", default=None, help="可选配置文件路径")
    parser.add_argument("--skip-download", action="store_true", help="只做筛选，不下载PDF")
    return parser.parse_args()


def load_checkpoint(checkpoint_file: Path) -> Tuple[List[Step1Decision], Set[str]]:
    """加载checkpoint，返回(decisions列表, 已处理的pmid集合)。"""
    if not checkpoint_file.exists():
        return [], set()
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    decisions = [Step1Decision(**item) for item in data]
    processed_pmids = {d.pmid for d in decisions}
    return decisions, processed_pmids


def save_checkpoint(decisions: List[Step1Decision], checkpoint_file: Path) -> None:
    """将当前decisions列表保存到checkpoint文件（覆盖写入）。"""
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump([asdict(d) for d in decisions], f, ensure_ascii=False, indent=2)


def generate_outputs(decisions: List[Step1Decision], output_dir: Path) -> Dict[str, str]:
    """根据最终的decisions列表生成所有输出文件。"""
    csv_out = output_dir / "screening_results.csv"
    json_out = output_dir / "screening_results.json"
    included_out = output_dir / "included_pmids.txt"

    # 写入CSV
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "PMID", "Title", "Include", "Confidence", "Reason", "DownloadStatus", "PDFPath", "Labels",
            ],
        )
        writer.writeheader()
        for d in decisions:
            writer.writerow(
                {
                    "PMID": d.pmid,
                    "Title": d.title,
                    "Include": d.include,
                    "Confidence": d.confidence,
                    "Reason": d.reason,
                    "DownloadStatus": d.download_status,
                    "PDFPath": d.pdf_path,
                    "Labels": json.dumps(d.labels, ensure_ascii=False),
                }
            )

    # 写入JSON
    json_out.write_text(
        json.dumps([asdict(d) for d in decisions], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 写入included PMIDs
    included_pmids = [d.pmid for d in decisions if d.include]
    included_out.write_text("\n".join(included_pmids), encoding="utf-8")

    return {
        "csv": str(csv_out),
        "json": str(json_out),
        "included_pmids": str(included_out),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = output_dir / "checkpoint.json"

    # 加载已有进度
    decisions, processed_pmids = load_checkpoint(checkpoint_file)
    print(f"Loaded {len(decisions)} previous decisions, {len(processed_pmids)} unique PMIDs processed.")

    # 初始化组件
    criteria = ReviewCriteriaLoader.load(Path(args.criteria_yaml))
    pubmed_fetcher = PubMedFetcher(config_path=args.config_path)
    downloader = PDFDownloader(config_path=args.config_path)
    llm_manager = LLMClientManager(config={})
    abstract_provider = LegalAbstractProvider(fetcher=pubmed_fetcher)
    judge = CriteriaLLMJudge(
        llm_manager=llm_manager,
        client_type=args.llm_client_type,
        model_name=args.llm_model_name,
        api_key=args.llm_apikey
    )

    # 读取输入CSV，获取所有PMID（保持顺序）
    input_pmids: List[Dict[str, str]] = []
    with open(args.csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = str(row.get("PMID", "")).strip()
            if not pmid:
                continue
            input_pmids.append({
                "pmid": pmid,
                "title": str(row.get("Title", "")).strip(),
                "abstract_from_csv": str(row.get("Abstract", "")).strip(),
            })

    # 过滤出未处理的PMID
    new_entries = [entry for entry in input_pmids if entry["pmid"] not in processed_pmids]
    print(f"Total PMIDs in CSV: {len(input_pmids)}, already processed: {len(processed_pmids)}, to process: {len(new_entries)}")

    # 处理新的PMID
    for entry in new_entries:
        pmid = entry["pmid"]
        title = entry["title"]
        abstract = entry["abstract_from_csv"]

        # 获取abstract（如果CSV没有则从PubMed获取）
        if not abstract:
            abstract = abstract_provider.fetch_abstract(pmid) or ""
            if not abstract:
                print(f"PMID {pmid}: abstract unavailable, skipping.")
                decision = Step1Decision(
                    pmid=pmid,
                    title=title,
                    abstract="",
                    include=False,
                    confidence=0.0,
                    reason="abstract_unavailable",
                    labels={"risk_flags": ["no_abstract"]},
                )
                decisions.append(decision)
                save_checkpoint(decisions, checkpoint_file)
                continue

        # LLM评估
        try:
            llm_out = judge.evaluate(criteria=criteria, abstract=abstract, pmid=pmid)
            decision = Step1Decision(
                pmid=pmid,
                title=title,
                abstract=abstract,
                include=bool(llm_out.get("include", False)),
                confidence=float(llm_out.get("confidence", 0.0)),
                reason=str(llm_out.get("reason", "")),
                labels=dict(llm_out.get("labels", {})),
            )
        except Exception as e:
            print(f"PMID {pmid}: LLM evaluation failed: {e}, marking as excluded.")
            decision = Step1Decision(
                pmid=pmid,
                title=title,
                abstract=abstract,
                include=False,
                confidence=0.0,
                reason=f"llm_error: {str(e)}",
                labels={"risk_flags": ["llm_failure"]},
            )

        # 下载PDF（如果满足条件）
        if decision.include and not args.skip_download:
            try:
                dl = downloader.download(pmid)
                decision.download_status = dl.status.value
                decision.pdf_path = str(dl.file_path) if dl.file_path else ""
            except Exception as e:
                print(f"PMID {pmid}: PDF download failed: {e}")
                decision.download_status = "failed"
                decision.pdf_path = ""

        decisions.append(decision)
        save_checkpoint(decisions, checkpoint_file)
        print(f"PMID {pmid}: processed, include={decision.include}, total decisions now: {len(decisions)}")

    # 所有处理完成后，生成最终输出文件
    output_files = generate_outputs(decisions, output_dir)
    summary = {
        "total": len(decisions),
        "included": len([d for d in decisions if d.include]),
        "excluded": len([d for d in decisions if not d.include]),
        "downloaded": len([d for d in decisions if d.pdf_path and d.download_status == "success"]),
        "output": output_files,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()