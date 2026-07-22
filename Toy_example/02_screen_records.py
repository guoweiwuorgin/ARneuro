"""Offline screening contract demo. Replace fixture decisions with an LLM screen in a real project."""

from _toy import project, write_report


def main() -> None:
    context = project()
    context.initialize()
    report = {"mode": "offline_fixture", "included_pmids": ["90000001", "90000002"], "excluded_pmids": [], "note": "No model call is made in the Toy example."}
    write_report("02_screen_records", report)
    context.record_stage("screen_records", "completed", inputs=[context.stage_dir("input") / "canonical_articles.csv"], outputs=[context.stage_dir("screening")], details=report)
    print(report)


if __name__ == "__main__":
    main()
