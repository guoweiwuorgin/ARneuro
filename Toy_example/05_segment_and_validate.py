from _toy import project, write_report
from ARneuro.workflow.stages import segment_markdown_directory


def main() -> None:
    report = segment_markdown_directory(project(), use_llm=False)
    write_report("05_segment_and_validate", report)
    print(report)


if __name__ == "__main__":
    main()
