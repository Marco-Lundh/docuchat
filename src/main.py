import io
import logging
import sys
from pathlib import Path

import truststore
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from chat import ask
from ingest import INDEX_PATH, build_index, reset_index
from retriever import retrieve
from tts import speak

console = Console()


def _setup() -> None:
    if (
        isinstance(sys.stdout, io.TextIOWrapper)
        and sys.stdout.encoding != "utf-8"
    ):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    truststore.inject_into_ssl()
    load_dotenv()


USAGE = (
    "Usage:\n"
    "  uv run src/main.py <pdf> [more.pdf ...]     Index and chat\n"
    "  uv run src/main.py <pdf> --speak            Read aloud in Swedish\n"
    "  uv run src/main.py <pdf> --speak --lang en  Read aloud in English\n"
    "  uv run src/main.py --reset                  Reset index\n"
    "  uv run src/main.py --reset <pdf> [...]      Reset and rebuild"
)

_LANG_LABEL = {"sv": "Swedish", "en": "English"}


def chat_loop(
    pdf_paths: list[str], speak_aloud: bool = False, lang: str = "sv"
) -> None:
    if not INDEX_PATH.exists():
        build_index(pdf_paths)
    else:
        console.print(
            "[dim]Using existing index. Pass --reset to rebuild.[/dim]"
        )

    if speak_aloud:
        label = _LANG_LABEL.get(lang, lang)
        speak_hint = f" [dim](reading aloud in {label})[/dim]"
    else:
        speak_hint = ""
    console.print(
        f"\n[bold green]docuchat[/bold green]{speak_hint} — "
        "type your question or [bold]quit[/bold] to exit.\n"
    )

    while True:
        question = Prompt.ask("[bold cyan]You[/bold cyan]")
        if question.strip().lower() in {"quit", "exit", "q"}:
            console.print("[dim]Exiting.[/dim]")
            break

        chunks = retrieve(question)
        answer = ask(question, chunks)
        if answer:
            console.print(f"\n[bold yellow]Answer[/bold yellow]: {answer}\n")
            if speak_aloud:
                speak(answer, lang=lang)
        else:
            console.print("[red]No answer returned.[/red]")


def main() -> None:
    _setup()
    args = sys.argv[1:]

    if "--help" in args or "-h" in args:
        console.print(USAGE)
        return

    do_reset = "--reset" in args
    speak_aloud = "--speak" in args

    lang = "sv"
    if "--lang" in args:
        idx = args.index("--lang")
        if idx + 1 < len(args):
            lang = args[idx + 1]
        else:
            console.print("[red]--lang requires a value: sv or en[/red]")
            sys.exit(1)
    if lang not in {"sv", "en"}:
        console.print(
            f"[red]Unsupported language: {lang!r}. Use 'sv' or 'en'.[/red]"
        )
        sys.exit(1)

    pdf_paths: list[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a == "--lang":
            skip_next = True
            continue
        if a.startswith("--"):
            continue
        pdf_paths.append(a)

    if do_reset and not pdf_paths:
        reset_index()
        console.print("[green]Index reset.[/green]")
        return

    if not pdf_paths and not INDEX_PATH.exists():
        console.print(USAGE)
        sys.exit(1)

    missing = [p for p in pdf_paths if not Path(p).exists()]
    if missing:
        for p in missing:
            console.print(f"[red]File not found:[/red] {p}")
        sys.exit(1)

    if do_reset:
        reset_index()
        console.print("[dim]Index reset, rebuilding...[/dim]")

    chat_loop(pdf_paths, speak_aloud=speak_aloud, lang=lang)


if __name__ == "__main__":
    main()
