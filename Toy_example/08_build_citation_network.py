"""Citation-network stage contract. The real builder parses References from the SQLite corpus."""

from _toy import project, write_report


def main() -> None:
    context = project()
    context.initialize()
    report = {"mode": "deferred_until_database", "note": "Run CitationNetworkBuilder after 09_build_sqlite_database.py for a real corpus."}
    write_report("08_build_citation_network", report)
    context.record_stage("build_citation_network", "planned", outputs=[context.stage_dir("citations")], details=report)
    print(report)


if __name__ == "__main__":
    main()
