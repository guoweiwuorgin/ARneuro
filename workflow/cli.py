"""The compact command-line entry point for the ARneuro v2 project workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .project import DEFAULT_LAYOUT, STAGE_ORDER, load_project
from .stages import build_database, segment_markdown_directory, validate_input


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _context_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", required=True, type=Path, help="Path to arneuro_project.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="arneuro", description="ARneuro v2 generic review workflow")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "validate-input", "segment", "build-database", "show-plan"):
        child = commands.add_parser(name)
        _context_argument(child)
    commands.choices["segment"].add_argument("--use-llm", action="store_true", help="Permit FinalSeg's optional LLM fallback.")
    commands.choices["build-database"].add_argument("--no-rebuild", action="store_true")
    arguments = parser.parse_args(argv)
    context = load_project(arguments.project)
    if arguments.command == "init":
        context.initialize()
        _print({"project": context.config.project_name, "output_root": str(context.output_root), "directories": {key: str(context.stage_dir(key)) for key in DEFAULT_LAYOUT}})
    elif arguments.command == "validate-input":
        _print(validate_input(context))
    elif arguments.command == "segment":
        _print(segment_markdown_directory(context, use_llm=arguments.use_llm))
    elif arguments.command == "build-database":
        _print(build_database(context, rebuild=not arguments.no_rebuild))
    else:
        _print({"stages": list(STAGE_ORDER), "manifest": context.read_manifest(), "next_commands": ["validate-input", "segment", "build-database"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
