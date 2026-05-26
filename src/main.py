import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import truststore
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

truststore.inject_into_ssl()
load_dotenv()

console = Console()

USAGE = (
    "Usage:\n"
    "  uv run src/main.py <file.pdf> [more.pdf ...]  Index and chat\n"
    "  uv run src/main.py --reset                    Reset index\n"
    "  uv run src/main.py --reset <file.pdf> [...]   Reset and rebuild"
)


def chat_loop(pdf_paths: list[str]) -> None:
    from ingest import INDEX_PATH, build_index
    from retriever import retrieve
    from chat import ask

    if not INDEX_PATH.exists():
        build_index(pdf_paths)
    else:
        console.print(
            "[dim]Using existing index. "
            "Pass --reset to rebuild.[/dim]"
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
        console.print(f"\n[bold yellow]Answer[/bold yellow]: {answer}\n")


def main() -> None:
    args = sys.argv[1:]

    do_reset = "--reset" in args
    pdf_paths = [a for a in args if not a.startswith("--")]

    if do_reset and not pdf_paths:
        from ingest import reset_index
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
        from ingest import reset_index
        reset_index()
        console.print("[dim]Index reset, rebuilding...[/dim]")

    chat_loop(pdf_paths)


if __name__ == "__main__":
    main()
