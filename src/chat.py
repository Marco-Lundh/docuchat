import os

from groq import Groq

MODEL = "llama-3.3-70b-versatile"

_client: Groq | None = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=os.environ["GROQ_API_KEY"])
    return _client


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
