"""Shared helpers for the offline, two-paper ARneuro Toy workflow."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ARNEURO_ROOT = ROOT.parent
STEP3_ROOT = ARNEURO_ROOT.parent
if str(STEP3_ROOT) not in sys.path:
    sys.path.insert(0, str(STEP3_ROOT))

from ARneuro.workflow.project import load_project


def project():
    return load_project(ROOT / "arneuro_project.yaml")


def copy_fixtures(source: Path, target: Path, pattern: str) -> int:
    target.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted(source.glob(pattern)):
        destination = target / path.name
        if not destination.exists():
            shutil.copy2(path, destination)
            copied += 1
    return copied


def write_report(name: str, value: Any) -> Path:
    report_path = project().stage_dir("logs") / f"{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return report_path
