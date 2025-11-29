import os
import sqlite3


DB_PATH = os.path.join(os.getcwd(), "app.db")




def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn




def init():
    conn = get_conn()
    conn.execute(
    """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            notes TEXT DEFAULT '',
            due_at TEXT,
            priority TEXT DEFAULT 'Medium',
            status TEXT DEFAULT 'todo',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()
    conn.close()