from datetime import datetime
from typing import List, Dict
from .db import get_conn, init as init_db




def init_db():
    init_db()




def add_task(title: str, notes: str, due_at: datetime, priority: str = "Medium"):
    conn = get_conn()
    conn.execute(
        "INSERT INTO tasks (title, notes, due_at, priority, status) VALUES (?, ?, ?, ?, 'todo')",
        (title, notes, due_at.isoformat(), priority),
    )
    conn.commit()
    conn.close()




def list_tasks() -> List[Dict]:
    conn = get_conn()
    rows = conn.execute("SELECT * FROM tasks ORDER BY status DESC, due_at ASC NULLS LAST, id DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]




def complete_task(task_id: int):
    conn = get_conn()
    conn.execute(
        "UPDATE tasks SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?",
        (task_id,),
    )
    conn.commit()
    conn.close()