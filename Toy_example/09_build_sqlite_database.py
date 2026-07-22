from _toy import project, write_report
from ARneuro.workflow.stages import build_database


def main() -> None:
    report = build_database(project(), rebuild=True)
    write_report("09_build_sqlite_database", report)
    print(report)


if __name__ == "__main__":
    main()
