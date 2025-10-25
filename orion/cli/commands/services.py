"""Service management commands (Neo4j, Ollama)."""

from __future__ import annotations

import argparse
import subprocess

from rich import box
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def handle_neo4j(args: argparse.Namespace) -> None:
    """Handle Neo4j service management commands."""
    action = getattr(args, "neo4j_action", None)
    if action is None:
        console.print("[red]No Neo4j action provided. Use 'orion services neo4j --help'.[/red]")
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            if action == "start":
                task = progress.add_task("[cyan]Checking Neo4j container...", total=None)

                check_result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", "name=orion-neo4j"],
                    capture_output=True,
                    text=True,
                )

                if "orion-neo4j" in check_result.stdout:
                    status_result = subprocess.run(
                        ["docker", "ps", "--filter", "name=orion-neo4j"],
                        capture_output=True,
                        text=True,
                    )

                    if "orion-neo4j" in status_result.stdout:
                        progress.update(task, description="[yellow]Neo4j already running")
                        progress.stop()
                        console.print("[green]✓ Neo4j container already running[/green]")
                    else:
                        progress.update(task, description="[cyan]Starting Neo4j...")
                        subprocess.run(
                            ["docker", "start", "orion-neo4j"], check=True, capture_output=True, text=True
                        )
                        progress.stop()
                        console.print("[green]✓ Neo4j container started successfully[/green]")
                        console.print("[dim]   Access at: http://localhost:7474[/dim]")
                else:
                    progress.stop()
                    console.print("[red]✗ Neo4j container not found[/red]")
                    console.print("[yellow]   Run 'orion init' first to create the container[/yellow]")

            elif action == "stop":
                task = progress.add_task("[cyan]Stopping Neo4j...", total=None)
                subprocess.run(["docker", "stop", "orion-neo4j"], check=True, capture_output=True, text=True)
                progress.stop()
                console.print("[green]✓ Neo4j container stopped[/green]")

            elif action == "status":
                task = progress.add_task("[cyan]Checking status...", total=None)
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=orion-neo4j"], capture_output=True, text=True
                )
                progress.stop()

                status_table = Table(box=box.ROUNDED, show_header=False)
                status_table.add_column("Item", style="cyan", width=20)
                status_table.add_column("Value", style="green", width=50)

                if "orion-neo4j" in result.stdout:
                    status_table.add_row("Status", "[green]● Running[/green]")
                    status_table.add_row("Browser UI", "http://localhost:7474")
                    status_table.add_row("Neo4j URI", "neo4j://localhost:7687")
                    console.print(status_table)
                else:
                    status_table.add_row("Status", "[red]○ Stopped[/red]")
                    console.print(status_table)
                    console.print("[yellow]Run 'orion services neo4j start' to start[/yellow]")

            elif action == "restart":
                task = progress.add_task("[cyan]Restarting Neo4j...", total=None)
                subprocess.run(
                    ["docker", "restart", "orion-neo4j"], check=True, capture_output=True, text=True
                )
                progress.stop()
                console.print("[green]✓ Neo4j container restarted[/green]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Docker command failed: {e.stderr}[/red]")
    except FileNotFoundError:
        console.print("[red]✗ Docker not found. Please install Docker first.[/red]")


def handle_ollama(args: argparse.Namespace) -> None:
    """Handle Ollama service management commands."""
    action = getattr(args, "ollama_action", None)
    if action is None:
        console.print("[red]No Ollama action provided. Use 'orion services ollama --help'.[/red]")
        return

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            if action == "start":
                task = progress.add_task("[cyan]Checking Ollama...", total=None)
                check_result = subprocess.run(
                    ["docker", "ps", "-a", "--filter", "name=orion-ollama"],
                    capture_output=True,
                    text=True,
                )

                if "orion-ollama" in check_result.stdout:
                    status_result = subprocess.run(
                        ["docker", "ps", "--filter", "name=orion-ollama"],
                        capture_output=True,
                        text=True,
                    )

                    if "orion-ollama" in status_result.stdout:
                        progress.update(task, description="[yellow]Ollama already running")
                        progress.stop()
                        console.print("[green]✓ Ollama container already running[/green]")
                    else:
                        progress.update(task, description="[cyan]Starting Ollama...")
                        subprocess.run(
                            ["docker", "start", "orion-ollama"], check=True, capture_output=True, text=True
                        )
                        progress.stop()
                        console.print("[green]✓ Ollama container started successfully[/green]")
                        console.print("[dim]   API available at: http://localhost:11434[/dim]")
                else:
                    progress.stop()
                    console.print("[red]✗ Ollama container not found[/red]")
                    console.print("[yellow]   Run 'orion init' first to set up Ollama[/yellow]")

            elif action == "stop":
                task = progress.add_task("[cyan]Stopping Ollama...", total=None)
                subprocess.run(
                    ["docker", "stop", "orion-ollama"], check=True, capture_output=True, text=True
                )
                progress.stop()
                console.print("[green]✓ Ollama container stopped[/green]")

            elif action == "status":
                task = progress.add_task("[cyan]Checking status...", total=None)
                result = subprocess.run(
                    ["docker", "ps", "--filter", "name=orion-ollama"], capture_output=True, text=True
                )
                progress.stop()

                status_table = Table(box=box.ROUNDED, show_header=False)
                status_table.add_column("Item", style="cyan", width=20)
                status_table.add_column("Value", style="green", width=50)

                if "orion-ollama" in result.stdout:
                    status_table.add_row("Status", "[green]● Running[/green]")
                    status_table.add_row("API Endpoint", "http://localhost:11434")
                    status_table.add_row("Model", "gemma3:4b")
                    console.print(status_table)
                else:
                    status_table.add_row("Status", "[red]○ Stopped[/red]")
                    console.print(status_table)
                    console.print("[yellow]Run 'orion services ollama start' to start[/yellow]")

            elif action == "restart":
                task = progress.add_task("[cyan]Restarting Ollama...", total=None)
                subprocess.run(
                    ["docker", "restart", "orion-ollama"], check=True, capture_output=True, text=True
                )
                progress.stop()
                console.print("[green]✓ Ollama container restarted[/green]")

    except subprocess.CalledProcessError as e:
        console.print(f"[red]✗ Docker command failed: {e.stderr}[/red]")
    except FileNotFoundError:
        console.print("[red]✗ Docker not found. Please install Docker first.[/red]")
