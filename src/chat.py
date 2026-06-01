import os
from functools import cache

from groq import Groq

MODEL = "llama-3.3-70b-versatile"


@cache
def _get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def ask(question: str, context_chunks: list[str]) -> str | None:
    context = "\n\n---\n\n".join(context_chunks)
    prompt = (
        "You are a helpful assistant. Answer the question based solely "
        "on the information below.\n"
        "If the answer is not found in the information, say so clearly.\n\n"
        f"INFORMATION:\n{context}\n\nQUESTION: {question}"
    )

    response = _get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content
