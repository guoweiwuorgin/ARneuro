"""Install Markdown fixtures in the OCR output contract; no OCR model is executed."""

from _toy import ROOT, copy_fixtures, project, write_report


def main() -> None:
    context = project()
    context.initialize()
    copied = copy_fixtures(ROOT / "fixtures" / "source_markdown", context.stage_dir("markdown"), "*.md")
    report = {"mode": "offline_markdown_fixture", "copied": copied, "note": "Replace this stage with OCR for imported PDFs in a real project."}
    write_report("04_ocr_to_markdown", report)
    context.record_stage("ocr_markdown", "completed", outputs=[context.stage_dir("markdown")], details=report)
    print(report)


if __name__ == "__main__":
    main()
