"""Declare the optional table/coordinate stage; the fixture papers have no activation table."""

from _toy import project, write_report


def main() -> None:
    context = project()
    context.initialize()
    report = {"mode": "offline_fixture", "activation_tables": 0, "coordinates": 0, "note": "The generic table parser is available for real segmented papers."}
    write_report("07_extract_tables_and_coordinates", report)
    context.record_stage("extract_tables_coordinates", "completed", outputs=[context.stage_dir("tables")], details=report)
    print(report)


if __name__ == "__main__":
    main()
