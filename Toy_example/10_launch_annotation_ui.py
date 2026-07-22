"""Launch the existing local review UI against the Toy SQLite corpus."""

from _toy import project
from ARneuro.database_ui.app import create_app


def main() -> None:
    context = project()
    database = context.stage_dir("database") / "review_corpus.sqlite3"
    workspace = context.stage_dir("database") / "annotation_workspace.sqlite3"
    app = create_app(database, workspace)
    app.run(host="127.0.0.1", port=8766, debug=False)


if __name__ == "__main__":
    main()
