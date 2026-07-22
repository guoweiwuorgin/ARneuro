#!/usr/bin/env python3
"""Create an ARneuro v2 project skeleton without copying source data or secrets."""

from __future__ import annotations

import argparse
from pathlib import Path


DIRECTORIES = (
    "input",
    "output/00_input",
    "output/01_screening",
    "output/02_pdfs",
    "output/03_markdown",
    "output/04_segmented",
    "output/05_study_info",
    "output/06_tables",
    "output/07_citations",
    "output/08_database",
    "output/09_exports",
    "output/logs",
)


def yaml_quote(value: str) -> str:
    return '"' + value.replace("\\", "/").replace('"', '\\"') + '"'


def config_text(name: str, input_table: str) -> str:
    return f"""project:
  name: {yaml_quote(name)}

input:
  pubmed_table: {yaml_quote(input_table)}

output:
  root: "output"

# Store secrets in environment variables, never in this file.
providers:
  study_extraction:
    api_key_env: "ARNEURO_LLM_API_KEY"
    model: "replace_with_your_model"

screening:
  include: []
  exclude: []

extraction:
  generic_study_schema: "v2"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_dir", type=Path, help="New or empty project directory")
    parser.add_argument("--name", default="arneuro_review", help="Project name stored in YAML")
    parser.add_argument("--input-table", default="input/pubmed_export.xlsx", help="Input path relative to project directory")
    parser.add_argument("--force", action="store_true", help="Replace an existing YAML file only")
    args = parser.parse_args()

    root = args.project_dir.expanduser().resolve()
    config_path = root / "arneuro_project.yaml"
    if config_path.exists() and not args.force:
        raise SystemExit(f"Refusing to replace existing config: {config_path}. Use --force to replace only the config file.")
    for relative in DIRECTORIES:
        (root / relative).mkdir(parents=True, exist_ok=True)
    config_path.write_text(config_text(args.name, args.input_table), encoding="utf-8")
    print(f"Created ARneuro project: {root}")
    print(f"Config: {config_path}")
    print("Next: place the PubMed export at the configured input path, then run:")
    print(f"  python -m ARneuro validate-input --project {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
