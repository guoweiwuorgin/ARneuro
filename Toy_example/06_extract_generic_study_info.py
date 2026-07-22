"""Copy reviewed fixture extraction records; real runs inject an API/local-model callable."""

from _toy import ROOT, copy_fixtures, project, write_report


def main() -> None:
    context = project()
    context.initialize()
    copied = copy_fixtures(ROOT / "fixtures" / "study_info", context.stage_dir("study_info"), "*.json")
    report = {"mode": "offline_reviewed_fixture", "copied": copied, "schema": "generic_study_schema_v2"}
    write_report("06_extract_generic_study_info", report)
    context.record_stage("extract_study_info", "completed", inputs=[context.stage_dir("segmented")], outputs=[context.stage_dir("study_info")], details=report)
    print(report)


if __name__ == "__main__":
    main()
