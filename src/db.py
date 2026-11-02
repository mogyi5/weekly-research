"""SQLite record of items already handled by earlier runs."""
import sqlite3
from contextlib import closing

from src import config


def _connect() -> sqlite3.Connection:
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(config.DB_PATH)


def init_db() -> None:
    with closing(_connect()) as conn, conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_items (
                item_id TEXT PRIMARY KEY,
                source TEXT,
                title TEXT,
                url TEXT,
                seen_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def is_item_seen(item_id: str) -> bool:
    with closing(_connect()) as conn:
        row = conn.execute(
            "SELECT 1 FROM seen_items WHERE item_id = ?", (item_id,)
        ).fetchone()
    return row is not None


def mark_items_seen(items: list[dict]) -> None:
    rows = [(i["id"], i["source"], i["title"], i["url"]) for i in items]
    with closing(_connect()) as conn, conn:
        conn.executemany(
            "INSERT OR IGNORE INTO seen_items (item_id, source, title, url) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
