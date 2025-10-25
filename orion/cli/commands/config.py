"""Config command - manage Orion configuration."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from ...settings import OrionSettings, SettingsError

console = Console()


def handle_config(args: argparse.Namespace, settings: OrionSettings) -> int:
    """Handle the config command - inspect and modify configuration."""
    from ...settings import SettingsError, OrionSettings

    subcommand = getattr(args, "config_command", None)
    if subcommand is None:
        console.print("[red]No configuration subcommand provided. Use 'orion config --help'.[/red]")
        return 1

    if subcommand == "show":
        table = Table(title="Orion Configuration", box=box.ROUNDED)
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for key, value in settings.iter_display_items():
            table.add_row(key, value)
        console.print(table)
        return 0

    if subcommand == "set":
        try:
            settings.set_value(args.key, args.value)
            settings.save()
        except SettingsError as exc:
            console.print(f"[red]{exc}[/red]")
            return 1
        console.print(f"[green]Updated {args.key}.[/green]")
        return 0

    if subcommand == "reset":
        defaults = OrionSettings()
        defaults.save()
        console.print("[green]Configuration reset to defaults.[/green]")
        return 0

    if subcommand == "path":
        console.print(str(OrionSettings.config_path()))
        return 0
    
    if subcommand == "credentials":
        try:
            password = settings.get_neo4j_password()
            console.print("\n[bold cyan]Neo4j Credentials[/bold cyan]")
            console.print(f"[bold]Browser URL:[/bold] http://localhost:7474")
            console.print(f"[bold]Username:[/bold] {settings.neo4j_user}")
            console.print(f"[bold]Password:[/bold] {password}")
            console.print("\n[dim]Use these credentials to access Neo4j Browser[/dim]\n")
            return 0
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1

    console.print(f"[red]Unknown configuration subcommand '{subcommand}'.[/red]")
    return 1
