"""Shared helpers for the 100-paper GLM-4.5-Air Batch API test."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


STEP3_DIR = Path("D:/language_template/reviewer/current_data/step3_library")
ARNEURO_DIR = STEP3_DIR / "ARneuro"
TEST_DIR = ARNEURO_DIR / "examples" / "glm45_air_batch_test"

DEFAULT_SEGMENTED_DIR = (
    STEP3_DIR / "final_segmented_for_analysis" / "full_segmented"
)
DEFAULT_EXCLUDED_DETAILS_CSV = (
    STEP3_DIR
    / "final_segmented_for_analysis"
    / "reports"
    / "final_seg_details.csv"
)
DEFAULT_OUTPUT_DIR = TEST_DIR / "output"
DEFAULT_API_KEY = "random-placeholder-7fb7ca41ea0e861ae48cac68bcd5c19f"
DEFAULT_MODEL_NAME = "glm-4.5-air"

for import_path in [ARNEURO_DIR / "feature_extraction", ARNEURO_DIR, STEP3_DIR]:
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def load_python_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


method_module = load_python_module(
    "batch_test_method_info_extractor",
    ARNEURO_DIR / "feature_extraction" / "method_info_extractor.py",
)
glm_module = load_python_module(
    "batch_test_glm_method_info_extractor",
    ARNEURO_DIR / "feature_extraction" / "glm_method_info_extractor.py",
)

METHOD_INFO_COLUMNS = method_module.METHOD_INFO_COLUMNS
MethodSectionInfoExtractor = method_module.MethodSectionInfoExtractor
GLMMethodSectionInfoExtractor = glm_module.GLMMethodSectionInfoExtractor
extract_json_object = method_module.extract_json_object
pmid_from_path = method_module.pmid_from_path
write_csv = method_module.write_csv
write_json = method_module.write_json


def get_api_key(cli_value: str | None = None) -> str:
    api_key = cli_value or os.environ.get("BIGMODEL_API_KEY") or DEFAULT_API_KEY
    if not api_key:
        raise ValueError("BigModel API key is empty.")
    return api_key


def create_zai_client(api_key: str):
    try:
        from zai import ZhipuAiClient
    except ImportError as exc:
        raise RuntimeError(
            "zai-sdk is required. Install it with: pip install -U zai-sdk"
        ) from exc
    return ZhipuAiClient(api_key=api_key)


def load_excluded_pmids(details_csv: Path) -> set[str]:
    if not details_csv.exists():
        return set()
    with details_csv.open("r", encoding="utf-8-sig", newline="") as file_obj:
        return {
            str(row.get("pmid", "")).strip()
            for row in csv.DictReader(file_obj)
            if row.get("status") == "excluded"
            and str(row.get("pmid", "")).strip()
        }


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_obj:
        data = json.load(file_obj)
    return data if isinstance(data, dict) else {}


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as file_obj:
        for line_number, line in enumerate(file_obj, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}: {exc}"
                ) from exc
            if isinstance(data, dict):
                records.append(data)
    return records


def write_jsonl(records: list[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file_obj:
        for record in records:
            file_obj.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )


def sdk_object_to_dict(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    for method_name in ["model_dump", "to_dict", "dict"]:
        method = getattr(value, method_name, None)
        if callable(method):
            data = method()
            if isinstance(data, dict):
                return data
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return {"value": str(value)}


def object_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)
