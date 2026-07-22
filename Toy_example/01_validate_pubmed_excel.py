from _toy import project, write_report
from ARneuro.workflow.stages import validate_input


def main() -> None:
    context = project()
    report = validate_input(context)
    write_report("01_validate_pubmed_excel", report)
    print(report)


if __name__ == "__main__":
    main()
