import sqlite3
from typing import Dict, List
from contextlib import closing


class ChatHistoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _init_db(self):
        with closing(self._get_conn()) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON messages(chat_id)")
            conn.commit()

    def load_history(self, chat_id: int, limit: int = 20) -> List[Dict[str, str]]:
        with closing(self._get_conn()) as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?",
                (chat_id, limit)
            ).fetchall()
        return [{"role": r, "content": c} for r, c in reversed(rows)]

    def append(self, chat_id: int, role: str, content: str, max_rows: int = 200):
        with closing(self._get_conn()) as conn:
            conn.execute(
                "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
                (chat_id, role, content)
            )
            conn.execute("""
                DELETE FROM messages
                WHERE chat_id = ? AND id NOT IN (
                    SELECT id FROM messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?
                )
            """, (chat_id, chat_id, max_rows))
            conn.commit()

    def clear(self, chat_id: int):
        with closing(self._get_conn()) as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            conn.commit()
