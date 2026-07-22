from _toy import project, write_report
from ARneuro.database_ui.app import CorpusRepository
from ARneuro.database_ui.workspace import AnnotationWorkspace


def main() -> None:
    context = project()
    database = context.stage_dir("database") / "review_corpus.sqlite3"
    workspace = context.stage_dir("database") / "annotation_workspace.sqlite3"
    report = CorpusRepository(database, AnnotationWorkspace(workspace)).training_export(context.stage_dir("exports"))
    context.record_stage("export_training_data", "completed", inputs=[database], outputs=[context.stage_dir("exports")], details=report)
    write_report("11_export_training_data", report)
    print(report)


if __name__ == "__main__":
    main()
