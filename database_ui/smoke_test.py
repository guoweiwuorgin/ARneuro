"""Small non-destructive API check for the local corpus workspace."""

from __future__ import annotations

import tempfile
import sqlite3
import json
from pathlib import Path

from .app import DEFAULT_DATABASE, create_app


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        app = create_app(workspace_path=Path(directory) / "workspace.sqlite3")
        client = app.test_client()
        stats = client.get("/api/stats")
        assert stats.status_code == 200, stats.data
        documents = client.get("/api/documents?page=1&page_size=10")
        assert documents.status_code == 200, documents.data
        items = documents.get_json()["items"]
        assert items, "Expected at least one indexed document."
        pmid = items[0]["pmid"]
        detail = client.get(f"/api/document/{pmid}")
        assert detail.status_code == 200, detail.data
        record = detail.get_json()
        annotation = client.post(
            f"/api/document/{pmid}/annotation",
            json={"training_status": "in_progress", "review_status": "draft", "reviewer": "smoke", "note": ""},
        )
        assert annotation.status_code == 200, annotation.data
        with sqlite3.connect(str(DEFAULT_DATABASE)) as connection:
            table_section = connection.execute(
                "SELECT pmid, content FROM sections WHERE lower(section_name)='tables' LIMIT 1"
            ).fetchone()
        print(
            json.dumps(
                {
                "pmid": pmid,
                "documents": stats.get_json()["documents"],
                "sections": len(record["sections"]),
                "tasks": len(record["tasks"]),
                "table_candidates": len(record["table_candidates"]),
                    "tables_section_sample": str(table_section[1])[:280] if table_section else "",
                },
                ensure_ascii=True,
            )
        )


if __name__ == "__main__":
    main()
