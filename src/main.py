import io
import sys
from pathlib import Path

import truststore
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from chat import ask
from ingest import INDEX_PATH, build_index, reset_index
from retriever import retrieve

console = Console()


def _setup() -> None:
    if (
        isinstance(sys.stdout, io.TextIOWrapper)
        and sys.stdout.encoding != "utf-8"
    ):
        sys.stdout.reconfigure(encoding="utf-8")
    truststore.inject_into_ssl()
    load_dotenv()


USAGE = (
    "Usage:\n"
    "  uv run src/main.py <file.pdf> [more.pdf ...]  Index and chat\n"
    "  uv run src/main.py --reset                    Reset index\n"
    "  uv run src/main.py --reset <file.pdf> [...]   Reset and rebuild"
)


def chat_loop(pdf_paths: list[str]) -> None:

    if not INDEX_PATH.exists():
        build_index(pdf_paths)
    else:
        console.print(
            "[dim]Using existing index. Pass --reset to rebuild.[/dim]"
        )

    console.print(
        "\n[bold green]docuchat[/bold green] — type your question "
        "or [bold]quit[/bold] to exit.\n"
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
        else:
            console.print("[red]No answer returned.[/red]")


def main() -> None:
    _setup()
    args = sys.argv[1:]

    do_reset = "--reset" in args
    pdf_paths = [a for a in args if not a.startswith("--")]

    if do_reset and not pdf_paths:
        reset_index()
        console.print("[green]Index reset.[/green]")
        return

    if not pdf_paths:
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

    chat_loop(pdf_paths)


if __name__ == "__main__":
    main()
