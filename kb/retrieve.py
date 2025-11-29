import os
from typing import List, Tuple
from openai import OpenAI


SYSTEM_PROMPT = (
    "You are a helpful assistant that answers using ONLY the provided context. "
    "If the answer isn't in the context, say you don't know. Include short, inline citations as [source pX]."
)




def _format_context(chunks: List[str], metas: List[dict]) -> str:
    lines = []
    for c, m in zip(chunks, metas):
        src = m.get("source", "?")
        page = m.get("page", "?")
        lines.append(f"[Source: {src} p{page}]\n{c}")
    return "\n\n".join(lines)




def answer_with_citations(collection, query: str, model: str = "gpt-4o-mini", top_k: int = 4) -> Tuple[str, List[dict]]:
    res = collection.query(query_texts=[query], n_results=top_k)
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]


    context = _format_context(docs, metas)
    client = OpenAI()
    msg = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
    ]
    cmp = client.chat.completions.create(model=model, messages=msg, temperature=0.2)
    answer = cmp.choices[0].message.content


    # Build a compact source list for UI
    uniq = []
    for m in metas:
        key = (m.get("source"), m.get("page"))
        if key not in [(u.get("source"), u.get("page")) for u in uniq]:
            uniq.append(m)


    return answer, uniq