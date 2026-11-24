"""Rich interactive CLI for SmartAgent."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

from smartagent.agent import SmartAgent

console = Console()


def main() -> None:
    agent = SmartAgent()
    console.print(
        Panel.fit(
            f"SmartAgent ready (mode: [bold]{agent.mode}[/bold]). "
            "Type 'exit' to quit or 'reset' to clear memory.",
            title="SmartAgent CLI",
        )
    )
    while True:
        try:
            prompt = console.input("[bold cyan]You>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() in {"exit", "quit"}:
            break
        if prompt.lower() == "reset":
            agent.reset_memory()
            console.print("[yellow]Conversation memory cleared.[/yellow]")
            continue

        result = agent.ask(prompt)
        for trace in result.tool_calls:
            console.print(
                f"[dim]Tool `{trace.tool_name}` ({trace.arguments}) -> ok={trace.ok}, attempts={trace.attempts}[/dim]"
            )
        console.print(f"[bold green]SmartAgent>[/bold green] {result.answer}\n")


if __name__ == "__main__":
    main()
