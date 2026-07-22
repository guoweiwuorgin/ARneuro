"""Portable project configuration and resumable stage manifests for ARneuro v2.

The workflow layer deliberately knows nothing about language research.  It
creates a stable project layout, resolves paths relative to a YAML file and
records every stage in a small JSON manifest that can be inspected or resumed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


STAGE_ORDER = (
    "validate_input",
    "screen_records",
    "acquire_pdfs",
    "ocr_markdown",
    "segment_documents",
    "extract_study_info",
    "extract_tables_coordinates",
    "build_citation_network",
    "build_database",
    "export_training_data",
)

DEFAULT_LAYOUT = {
    "input": "00_input",
    "screening": "01_screening",
    "pdfs": "02_pdfs",
    "markdown": "03_markdown",
    "segmented": "04_segmented",
    "study_info": "05_study_info",
    "tables": "06_tables",
    "citations": "07_citations",
    "database": "08_database",
    "exports": "09_exports",
    "logs": "logs",
}

ENV_TOKEN_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        return ENV_TOKEN_RE.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


def _stable_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProjectConfig:
    """Validated, path-independent content of an ``arneuro_project.yaml`` file."""

    project_name: str
    input_table: str
    output_root: str = "output"
    providers: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    screening: Dict[str, Any] = field(default_factory=dict)
    extraction: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        expanded = _expand_environment(dict(data))
        project = expanded.get("project", {}) if isinstance(expanded.get("project"), dict) else {}
        input_data = expanded.get("input", {}) if isinstance(expanded.get("input"), dict) else {}
        output_data = expanded.get("output", {}) if isinstance(expanded.get("output"), dict) else {}
        input_table = str(input_data.get("pubmed_table", "")).strip()
        if not input_table:
            raise ValueError("Project config requires input.pubmed_table.")
        return cls(
            project_name=str(project.get("name", "arneuro_project")).strip() or "arneuro_project",
            input_table=input_table,
            output_root=str(output_data.get("root", "output")).strip() or "output",
            providers=dict(expanded.get("providers", {}) or {}),
            screening=dict(expanded.get("screening", {}) or {}),
            extraction=dict(expanded.get("extraction", {}) or {}),
            raw=dict(expanded),
        )


@dataclass
class ProjectContext:
    """Resolved project paths plus a durable, atomically written run manifest."""

    config_path: Path
    config: ProjectConfig

    @property
    def root(self) -> Path:
        return self.config_path.parent.resolve()

    @property
    def output_root(self) -> Path:
        return (self.root / self.config.output_root).resolve()

    @property
    def manifest_path(self) -> Path:
        return self.output_root / "run_manifest.json"

    @property
    def input_table_path(self) -> Path:
        return self.resolve_path(self.config.input_table)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (self.root / path).resolve()

    def stage_dir(self, name: str) -> Path:
        if name not in DEFAULT_LAYOUT:
            raise KeyError(f"Unknown ARneuro project directory: {name}")
        return self.output_root / DEFAULT_LAYOUT[name]

    def initialize(self) -> None:
        for name in DEFAULT_LAYOUT:
            self.stage_dir(name).mkdir(parents=True, exist_ok=True)
        if not self.manifest_path.exists():
            self._write_manifest(
                {
                    "schema_version": "2.0",
                    "project_name": self.config.project_name,
                    "created_at": utc_now(),
                    "config_sha256": _stable_hash(self.config.raw),
                    "stages": {},
                }
            )

    def read_manifest(self) -> Dict[str, Any]:
        self.initialize()
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid run manifest: {self.manifest_path}") from exc

    def record_stage(
        self,
        stage: str,
        state: str,
        *,
        inputs: Iterable[str | Path] = (),
        outputs: Iterable[str | Path] = (),
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if stage not in STAGE_ORDER:
            raise KeyError(f"Unknown ARneuro stage: {stage}")
        manifest = self.read_manifest()
        stages = manifest.setdefault("stages", {})
        stages[stage] = {
            "state": state,
            "updated_at": utc_now(),
            "inputs": [str(Path(value)) for value in inputs],
            "outputs": [str(Path(value)) for value in outputs],
            "details": dict(details or {}),
        }
        self._write_manifest(manifest)

    def _write_manifest(self, payload: Mapping[str, Any]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.manifest_path)


def load_project(config_path: str | Path) -> ProjectContext:
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Project config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Project config must be a YAML mapping.")
    return ProjectContext(config_path=path, config=ProjectConfig.from_mapping(data))
