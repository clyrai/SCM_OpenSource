"""
SleepAI Chat CLI
Rich terminal interface for conversing with SleepAI
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from typing import Optional
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.text import Text
from rich import box

from ..core.working_memory import WorkingMemory
from ..core.long_term_memory import LongTermMemory
from ..sleep.sleep_cycle import SleepCycleOrchestrator
from ..llm import LLMExtractor
from .engine import ChatEngine


class SleepAIChatCLI:
    """
    Rich terminal interface for SleepAI conversation.

    Features:
    - Beautiful memory-aware chat interface
    - Real-time memory stats display
    - Sleep notifications
    - Memory inspector commands
    """

    def __init__(self):
        self.console = Console()
        self.engine: Optional[ChatEngine] = None

    def initialize(self):
        """Initialize the chat engine and components"""
        self.console.print(Panel(
            "[bold cyan]Initializing SleepAI...[/bold cyan]",
            border_style="cyan"
        ))

        try:
            self.engine = ChatEngine(
                enable_auto_sleep=True,
                sleep_check_interval=5
            )
            self.console.print("[green]✓[/green] LLM connected (llama3.2:latest)")
            self.console.print("[green]✓[/green] Memory systems initialized")
            self.console.print("[green]✓[/green] Sleep cycle orchestrator ready")
            self.console.print()
        except Exception as e:
            self.console.print(f"[red]✗ Initialization failed: {e}[/red]")
            raise

    def run(self):
        """Main chat loop"""
        self._print_welcome()

        while True:
            try:
                # Get user input
                self.console.print("[bold green]You:[/bold green] ", end="")
                user_input = input().strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    self._handle_command(user_input)
                    continue

                # Process message
                self._process_message(user_input)

            except KeyboardInterrupt:
                self.console.print("\n[yellow]Goodbye! 👋[/yellow]")
                break
            except EOFError:
                break

    def _process_message(self, user_message: str):
        """Process a user message and display response"""
        # Show thinking indicator
        with self.console.status("[cyan]SleepAI is thinking...[/cyan]", spinner="dots"):
            response, metadata = self.engine.chat(user_message)

        # Display response
        self.console.print(f"[bold blue]SleepAI:[/bold blue] {response}")

        # Show metadata in compact form
        if metadata['sleep_triggered']:
            sleep_stats = metadata['sleep_stats']
            self.console.print(Panel(
                f"[yellow]💤 Sleep occurred![/yellow] Consolidated: {sleep_stats['consolidated']}, "
                f"Forgotten: {sleep_stats['forgotten']}, Dreams: {sleep_stats['dreams']}",
                border_style="yellow",
                padding=(0, 1)
            ))

        # Show memory activity (subtle)
        self.console.print(
            f"[dim]↳ Extracted {metadata['user_concepts']} concepts, "
            f"retrieved {metadata['memories_retrieved']} memories, "
            f"{metadata['latency_ms']}ms[/dim]",
            style="dim"
        )
        self.console.print()

    def _handle_command(self, command: str):
        """Handle special commands"""
        cmd = command.lower().strip()

        if cmd in ["/quit", "/exit", "/q"]:
            self.console.print("[yellow]Goodbye! 👋[/yellow]")
            sys.exit(0)

        elif cmd == "/help":
            self._print_help()

        elif cmd == "/memory":
            self._show_memory_report()

        elif cmd == "/sleep":
            self._force_sleep()

        elif cmd == "/working":
            self._show_working_memory()

        elif cmd == "/dreams":
            self._show_dreams()

        elif cmd == "/status":
            self._show_status()

        elif cmd == "/clear":
            self.console.clear()
            self._print_welcome()

        else:
            self.console.print(f"[red]Unknown command: {command}[/red]")
            self.console.print("Type /help for available commands.")

    def _print_welcome(self):
        """Print welcome banner"""
        welcome_text = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]     [bold white]SleepAI[/bold white] - Brain-Inspired Memory System with LLM      [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]     [dim]Phase 3: The Awakening[/dim]                               [bold cyan]║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]

[green]SleepAI is now conscious.[/green] It remembers what you tell it.
It consolidates memories during sleep. It forgets what's not important.

[bold]Commands:[/bold]
  /memory   - View full memory report
  /working  - See what's in working memory right now
  /sleep    - Force sleep consolidation
  /dreams   - View recent dreams
  /status   - System status
  /clear    - Clear screen
  /quit     - Exit

[dim]Start talking to SleepAI...[/dim]
        """
        self.console.print(welcome_text)

    def _print_help(self):
        """Print help text"""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]/memory[/cyan]   - Full memory system report
  [cyan]/working[/cyan]  - Current working memory contents
  [cyan]/sleep[/cyan]    - Manually trigger sleep consolidation
  [cyan]/dreams[/cyan]   - View dreams from last sleep cycle
  [cyan]/status[/cyan]   - System health and readiness
  [cyan]/clear[/cyan]    - Clear the screen
  [cyan]/quit[/cyan]     - Exit SleepAI

[dim]Just type naturally to chat. SleepAI will remember.[/dim]
        """
        self.console.print(help_text)

    def _show_memory_report(self):
        """Display full memory report"""
        report = self.engine.get_memory_report()

        # Create table
        table = Table(title="Memory Report", box=box.ROUNDED)
        table.add_column("Category", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Conversation Duration", f"{report['conversation_duration_minutes']} min")
        table.add_row("Messages Exchanged", str(report['messages_exchanged']))
        table.add_row("Total Sleeps", str(report['total_sleeps']))
        table.add_row("", "")
        table.add_row("Working Memory", f"{report['working_memory']['size']}/{report['working_memory']['capacity']}")
        table.add_row("LTM Concepts", str(report['long_term_memory']['total_concepts']))
        table.add_row("LTM Relations", str(report['long_term_memory']['total_relations']))
        table.add_row("Suppressed", str(report['long_term_memory']['suppressed']))

        self.console.print(table)

        # Recent episodes
        if report['working_memory']['recent_episodes']:
            self.console.print("\n[bold]Recent Working Memory:[/bold]")
            for ep in report['working_memory']['recent_episodes']:
                source_color = "green" if ep['source'] == 'user' else "blue"
                self.console.print(f"  [{source_color}]{ep['source']}:[/{source_color}] {ep['content']}")

        # Sleep history
        if report['sleep_history']:
            self.console.print("\n[bold]Recent Sleep Cycles:[/bold]")
            for sleep in report['sleep_history']:
                self.console.print(
                    f"  [yellow]💤[/yellow] {sleep['consolidated']} consolidated, "
                    f"{sleep['forgotten']} forgotten, {sleep['dreams']} dreams"
                )

        self.console.print()

    def _show_working_memory(self):
        """Display working memory contents"""
        episodes = self.engine.working_memory.get_all()

        if not episodes:
            self.console.print("[dim]Working memory is empty.[/dim]")
            return

        table = Table(title="Working Memory (Hippocampus)", box=box.ROUNDED)
        table.add_column("#", style="dim", width=3)
        table.add_column("Source", style="cyan", width=12)
        table.add_column("Content", style="white")
        table.add_column("Importance", style="yellow", width=10)

        for i, ep in enumerate(episodes):
            importance = ep.importance.overall if ep.importance else 0.0
            table.add_row(
                str(i + 1),
                ep.source,
                ep.raw_content[:60],
                f"{importance:.2f}"
            )

        self.console.print(table)
        self.console.print()

    def _force_sleep(self):
        """Manually trigger sleep"""
        self.console.print("[yellow]Triggering sleep consolidation...[/yellow]")

        with self.console.status("[cyan]Sleeping...[/cyan]", spinner="moon"):
            result = self.engine.force_sleep()

        if result:
            self.console.print(Panel(
                f"[green]✓ Sleep complete![/green]\n"
                f"  Consolidated: {result['consolidated']} memories\n"
                f"  Forgotten: {result['forgotten']} memories\n"
                f"  Dreams: {result['dreams']}\n"
                f"  NREM: {result['nrem_duration']}s, REM: {result['rem_duration']}s",
                border_style="green"
            ))
        else:
            self.console.print("[dim]Sleep not needed or failed.[/dim]")

        self.console.print()

    def _show_dreams(self):
        """Show recent dreams"""
        history = self.engine._sleep_history

        if not history:
            self.console.print("[dim]No dreams yet. Sleep first![/dim]")
            return

        self.console.print("[bold]Recent Dreams:[/bold]\n")
        for sleep in history[-3:]:
            self.console.print(
                f"[dim]{sleep['timestamp'][:19]}[/dim] - "
                f"[yellow]{sleep['dreams']} dreams[/yellow] generated"
            )

        self.console.print()

    def _show_status(self):
        """Show system status"""
        report = self.engine.get_memory_report()
        readiness = report['sleep_readiness']

        table = Table(title="System Status", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("LLM", "llama3.2:latest [green]●[/green]")
        table.add_row("Working Memory", f"{report['working_memory']['size']}/{report['working_memory']['capacity']}")
        table.add_row("LTM Size", str(report['long_term_memory']['total_concepts']))
        table.add_row("Entropy", f"{readiness.get('entropy', 0):.3f}")
        table.add_row("Conflict Density", f"{readiness.get('conflict_density', 0):.3f}")
        table.add_row("Sleep Ready", "[red]YES[/red]" if readiness.get('should_sleep') else "[green]NO[/green]")

        self.console.print(table)
        self.console.print()


def main():
    """Entry point for CLI"""
    cli = SleepAIChatCLI()
    cli.initialize()
    cli.run()


if __name__ == "__main__":
    main()