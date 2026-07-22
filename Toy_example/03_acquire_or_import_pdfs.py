"""Demonstrate the PDF contract without downloading copyrighted material."""

from _toy import project, write_report


def main() -> None:
    context = project()
    context.initialize()
    report = {"mode": "offline_fixture", "imported_pdfs": 0, "note": "Real projects import local lawful PDFs here or use an OA-aware acquisition adapter."}
    write_report("03_acquire_or_import_pdfs", report)
    context.record_stage("acquire_pdfs", "completed", outputs=[context.stage_dir("pdfs")], details=report)
    print(report)


if __name__ == "__main__":
    main()
