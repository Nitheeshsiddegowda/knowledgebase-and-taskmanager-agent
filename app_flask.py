import os, sqlite3, re
from datetime import datetime
from typing import List, Dict, Tuple

from flask import Flask, render_template, request, redirect, url_for, flash
from pypdf import PdfReader
from groq import Groq
import traceback

from math import log

def build_corpus(rows):
    # rows: list of sqlite rows with fields: source, page, chunk
    docs = []
    for r in rows:
        toks = tokenize(r["chunk"])
        docs.append({"row": r, "tokens": toks, "len": len(toks)})
    return docs

def bm25_rank(query_tokens, docs, k1=1.5, b=0.75, topk=4):
    # Precompute DF and IDF
    N = len(docs) or 1
    df = {}
    for d in docs:
        for t in set(d["tokens"]):
            df[t] = df.get(t, 0) + 1
    idf = {t: log( (N - df_t + 0.5) / (df_t + 0.5) + 1 ) for t, df_t in df.items()}
    avgdl = (sum(d["len"] for d in docs) / N) if N else 1.0

    scores = []
    for d in docs:
        # term frequency map
        tf = {}
        for t in d["tokens"]:
            tf[t] = tf.get(t, 0) + 1
        s = 0.0
        for q in set(query_tokens):
            if q not in tf:
                continue
            q_idf = idf.get(q, 0.0)
            numer = tf[q] * (k1 + 1.0)
            denom = tf[q] + k1 * (1.0 - b + b * (d["len"] / avgdl))
            s += q_idf * (numer / denom)
        scores.append((s, d["row"]))
    scores.sort(key=lambda x: x[0], reverse=True)
    # Keep only the best chunk per (source,page)
    best_per_page = {}
    for s, r in scores:
        key = (r["source"], r["page"])
        if key not in best_per_page:
            best_per_page[key] = (s, r)
    ranked = sorted(best_per_page.values(), key=lambda x: x[0], reverse=True)[:topk]
    return [{"source": r["source"], "page": r["page"], "chunk": r["chunk"], "score": round(s,4)} for s, r in ranked]

def safe_int(val, default=4):
    try:
        return int(val)
    except Exception:
        return default


DB_PATH = "app.db"
# Tuning knobs (EXTREME SAFE)
PAGE_LIMIT = 5          # process first 2 pages per PDF while testing
CHUNK_SIZE = 1200       # bigger chunks => fewer rows
CHUNK_OVERLAP = 50      # less overlap => fewer rows
MIN_CHARS = 20         # skip tiny fragments
MAX_PAGE_CHARS = 120_000  # hard cap per page to avoid huge memory
BATCH_INSERT = 50       # insert to DB every N chunks


app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", "dev-secret")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- DB ----------
def db():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c

def init_db():
    c = db()
    c.execute("""
      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source TEXT,
        page INTEGER,
        chunk TEXT
      );
    """)
    c.execute("""
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
    """)
    c.commit(); c.close()

# ---------- Utilities ----------
_word_re = re.compile(r"[A-Za-z0-9']+")

def tokenize(text: str) -> List[str]:
    return [w.lower() for w in _word_re.findall(text or "")]

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Generator that yields chunks without storing all in memory.
    """
    if not text:
        return
    # Normalize whitespace early
    text = " ".join(text.split())
    # Hard cap to avoid huge pages (some PDFs have page-level text > 1–5 MB)
    if len(text) > MAX_PAGE_CHARS:
        text = text[:MAX_PAGE_CHARS]

    n = len(text)
    i = 0
    while i < n:
        j = min(i + size, n)
        piece = text[i:j]
        if len(piece) >= MIN_CHARS:
            yield piece
        if j == n:
            break
        # advance window with overlap protection
        nxt = j - overlap
        i = nxt if nxt > i else j


# ---------- Indexing (very light) ----------
def index_pdf(file_storage, page_limit: int | None = PAGE_LIMIT) -> int:
    """
    Stream page -> chunk generator -> small batch inserts.
    Keeps memory usage very low.
    """
    name = file_storage.filename
    reader = PdfReader(file_storage.stream)

    total_added = 0
    batch = []

    def flush_batch():
        nonlocal batch, total_added
        if not batch:
            return
        c = db()
        c.executemany("INSERT INTO chunks (source, page, chunk) VALUES (?, ?, ?)", batch)
        c.commit(); c.close()
        total_added += len(batch)
        batch = []

    total_pages = len(reader.pages)
    max_pages = min(total_pages, page_limit) if page_limit else total_pages
    print(f"[INDEX] {name}: {total_pages} pages (processing first {max_pages})")

    try:
        for pi, page in enumerate(reader.pages, start=1):
            if page_limit and pi > page_limit:
                break

            # Extract text (might be very large if PDF is odd)
            txt = page.extract_text() or ""
            # Stream chunks; don't accumulate more than BATCH_INSERT at once
            page_added = 0
            for ch in chunk_text(txt):
                batch.append((name, pi, ch))
                if len(batch) >= BATCH_INSERT:
                    flush_batch()
                page_added += 1

            print(f"[INDEX] {name}: page {pi}/{max_pages} -> {page_added} chunks")
        # flush remaining
        flush_batch()

    except MemoryError:
        # Flush anything pending, then bail out gracefully
        try:
            flush_batch()
        except Exception:
            pass
        print("[INDEX][ERROR] MemoryError while indexing; partial data was saved.")
        # Return what we did save so far
        return total_added

    print(f"[INDEX] {name}: wrote {total_added} chunks total")
    return total_added


# ---------- Retrieval (keyword score; no ML) ----------
def score_chunk(query_tokens: List[str], chunk_text: str) -> int:
    if not query_tokens or not chunk_text:
        return 0
    ctoks = set(tokenize(chunk_text))
    # Simple overlap score (can be improved later)
    return sum(1 for t in query_tokens if t in ctoks)

def retrieve(query: str, k: int = 4) -> List[Dict]:
    q_tokens = tokenize(query)
    c = db()
    rows = c.execute("SELECT source, page, chunk FROM chunks").fetchall()
    c.close()

    # If DB empty
    if not rows:
        return []

    # Try BM25 first if we have a query
    results = []
    if q_tokens:
        docs = build_corpus(rows)
        results = bm25_rank(q_tokens, docs, k1=1.5, b=0.75, topk=k)

    # Backfill if matches are too weak or empty:
    if len(results) < k:
        # one longest chunk per (source,page)
        best_by_len = {}
        for r in rows:
            key = (r["source"], r["page"])
            cur = best_by_len.get(key)
            if (cur is None) or (len(r["chunk"]) > len(cur["chunk"])):
                best_by_len[key] = r
        already = {(x["source"], x["page"]) for x in results}
        fillers = [
            {"source": v["source"], "page": v["page"], "chunk": v["chunk"], "score": 0.0}
            for k2, v in sorted(best_by_len.items(), key=lambda kv: len(kv[1]["chunk"]), reverse=True)
            if k2 not in already
        ]
        results.extend(fillers[: max(0, k - len(results))])

    return results

# --- Bullet formatting helper ---
# --- Bullet formatting helper ---
def format_bullets(text: str) -> str:
    """
    Convert lines starting with '*' into <li> HTML bullet points.
    Supports multi-line answers gracefully.
    """
    lines = text.split("\n")
    items = []
    ul_open = False

    for ln in lines:
        striped = ln.strip()
        if striped.startswith("* "):
            if not ul_open:
                items.append("<ul>")
                ul_open = True
            items.append(f"<li>{striped[2:].strip()}</li>")
        else:
            if ul_open:
                items.append("</ul>")
                ul_open = False
            items.append(striped)

    if ul_open:
        items.append("</ul>")

    return "<br>".join(items)


# ---------- Answer (Groq) ----------
import time, traceback

def answer(query: str, k: int = 4, model: str = "llama-3.1-8b-instant"):
    ctx = retrieve(query, k)
    if not ctx:
        c = db()
        r = c.execute("SELECT source, page, chunk FROM chunks LIMIT 4").fetchall()
        c.close()
        ctx = [{"source": x["source"], "page": x["page"], "chunk": x["chunk"], "score": 0} for x in r]

    context_txt = "\n\n".join([f"[Source: {c['source']} p{c['page']}]\n{c['chunk']}" for c in ctx]) or "(no relevant context found)"
    msgs = [
  {"role":"system","content":"Using ONLY the context, give a concise answer. If the question is broad, give a 3–5 bullet summary. Always include short citations like [source pX]. If context is empty, say it may be a scanned PDF or empty page."},
  {"role":"user","content": f"Question: {query}\n\nContext:\n{context_txt}"}
]


    last_err = None
    for attempt in range(3):  # retry 0s, 2s, 4s
        try:
            resp = client.chat.completions.create(model=model, messages=msgs, temperature=0.2)
            ans = resp.choices[0].message.content
            ans = format_bullets(ans)
            return ans, ctx

        except Exception as e:
            last_err = e
            # if model issue, fallback once
            if "model" in str(e).lower():
                try:
                    resp = client.chat.completions.create(model="llama-3.1-8b-instant", messages=msgs, temperature=0.2)
                    ans = resp.choices[0].message.content
                    ans = format_bullets(ans)
                    return ans, ctx

                except Exception as e2:
                    last_err = e2
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Connection to Groq failed: {last_err}")

# ---------- Tasks ----------
def add_task(title: str, notes: str, due_iso: str, priority: str):
    c = db()
    c.execute("INSERT INTO tasks(title,notes,due_at,priority,status) VALUES (?,?,?,?, 'todo')",
              (title, notes, due_iso, priority))
    c.commit(); c.close()

def list_tasks():
    c = db()
    rows = c.execute("SELECT * FROM tasks ORDER BY status DESC, due_at ASC, id DESC").fetchall()
    c.close()
    return rows

def mark_done(task_id: int):
    c = db()
    c.execute("UPDATE tasks SET status='done', updated_at=CURRENT_TIMESTAMP WHERE id=?", (task_id,))
    c.commit(); c.close()

def delete_task(task_id: int):
    c = db()
    c.execute("DELETE FROM tasks WHERE id=?", (task_id,))
    c.commit(); c.close()


def clear_kb():
    c = db()
    c.execute("DELETE FROM chunks")
    c.commit(); c.close()

# --- KB preview (see what’s indexed) ---
def list_chunks(limit=200):
    c = db()
    rows = c.execute(
        "SELECT source, page, LENGTH(chunk) AS n, SUBSTR(chunk,1,200) AS preview "
        "FROM chunks ORDER BY source, page LIMIT ?",
        (limit,)
    ).fetchall()
    c.close()
    return rows


@app.route("/kb")
def kb():
    rows = list_chunks(200)
    return render_template("kb.html", rows=rows)

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/index", methods=["POST"])
def do_index():
    try:
        files = request.files.getlist("pdfs")
        if not files or files[0].filename == "":
            flash("Please choose at least one text-based PDF.")
            return redirect(url_for("home"))

        total = 0
        for f in files:
            try:
                total += index_pdf(f, page_limit=PAGE_LIMIT)
            except MemoryError:
                flash("Memory limit reached while indexing. Try smaller PDFs or keep PAGE_LIMIT low.")
                break
            except Exception as e:
                print("[INDEX][ERROR]", traceback.format_exc())
                flash(f"Index error: {e}")
                break

        flash(f"Indexed {total} chunks.")
        return redirect(url_for("home"))
    except Exception as e:
        print("[INDEX][FATAL]", traceback.format_exc())
        flash(f"Fatal index error: {e}")
        return redirect(url_for("home"))

@app.route("/ask", methods=["GET","POST"])
def ask():
    ans, sources = None, []
    if request.method == "POST":
        if not os.getenv("GROQ_API_KEY"):
            flash("GROQ_API_KEY is not set. Set it and try again.")
            return render_template("ask.html", answer=None, sources=[])

        q = (request.form.get("query") or "").strip()
        k = safe_int(request.form.get("topk"), 4)
        model = (request.form.get("model") or "llama-3.1-8b-instant").strip()
        if q:
            try:
                ans, sources = answer(q, k, model)
            except Exception as e:
                print("[ASK][ERROR]", traceback.format_exc())
                flash(f"Ask failed: {e}")
                ans, sources = None, []
    return render_template("ask.html", answer=ans, sources=sources)

@app.route("/tasks", methods=["GET","POST"])
def tasks():
    if request.method == "POST":
        title = request.form.get("title","").strip()
        notes = request.form.get("notes","")
        due = request.form.get("due_at","")
        priority = request.form.get("priority","Medium")
        if title:
            due_iso = (datetime.fromisoformat(due).isoformat() if due else None)
            add_task(title, notes, due_iso, priority)
            flash("Task added.")
        return redirect(url_for("tasks"))
    items = list_tasks()
    return render_template("tasks.html", items=items)

@app.route("/tasks/done/<int:task_id>")
def tasks_done(task_id):
    mark_done(task_id)
    return redirect(url_for("tasks"))

@app.route("/tasks/delete/<int:task_id>", methods=["POST"])
def tasks_delete(task_id):
    try:
        delete_task(task_id)
        flash("Task deleted.")
    except Exception as e:
        flash(f"Delete failed: {e}")
    return redirect(url_for("tasks"))


@app.errorhandler(500)
def handle_500(e):
    print("[500]", traceback.format_exc())
    msg = getattr(e, "description", str(e))
    return render_template("500.html", msg=msg), 500

@app.route("/clear", methods=["POST"])
def clear():
    try:
        clear_kb()
        flash("Knowledge Base cleared.")
    except Exception as e:
        flash(f"Clear failed: {e}")
    return redirect(url_for("home"))

if __name__ == "__main__":
    if not os.getenv("GROQ_API_KEY"):
        print("[WARN] Set GROQ_API_KEY before running. Example:")
        print('       $env:GROQ_API_KEY="gsk_your_key_here"   (PowerShell)')
    init_db()
    print("[ROUTES]", app.url_map)
    app.run(debug=False)
