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
            console.print(f"[bold]Connection URI:[/bold] {settings.neo4j_uri}")
            console.print(f"[bold]Username:[/bold] {settings.neo4j_user}")
            console.print(f"[bold]Password:[/bold] {password}")
            console.print(f"\n[dim]Encoded in config:[/dim] {settings.neo4j_password_encoded}")
            console.print("\n[dim]Use these credentials to access Neo4j Browser[/dim]")
            
            # Test the connection
            console.print("\n[cyan]Testing connection...[/cyan]")
            try:
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(
                    settings.neo4j_uri,
                    auth=(settings.neo4j_user, password),
                    connection_timeout=5
                )
                driver.verify_connectivity()
                driver.close()
                console.print("[green]✓ Connection successful![/green]\n")
            except Exception as e:
                console.print(f"[red]✗ Connection failed: {e}[/red]")
                console.print("\n[yellow]Troubleshooting tips:[/yellow]")
                console.print("  1. Check if Neo4j is running: [bold]orion services neo4j status[/bold]")
                console.print("  2. Reset password: [bold]orion config reset-password[/bold]")
                console.print("  3. Restart Neo4j: [bold]orion services neo4j restart[/bold]\n")
            return 0
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 1
    
    if subcommand == "reset-password":
        from ...settings import generate_secure_password
        from rich.prompt import Confirm
        import getpass
        import subprocess
        
        console.print("\n[bold cyan]Reset Neo4j Password[/bold cyan]\n")
        console.print("[yellow]⚠ This will update the password in your config and restart the Neo4j container.[/yellow]\n")
        
        choice = Confirm.ask("Would you like to generate a random password?", default=True)
        
        if choice:
            new_password = generate_secure_password(16)
            console.print(f"\n[green]✓ Generated password:[/green] {new_password}")
        else:
            console.print("\n[cyan]Enter your new Neo4j password (min 8 characters):[/cyan]")
            while True:
                new_password = getpass.getpass("Password: ")
                if len(new_password) < 8:
                    console.print("[yellow]⚠ Password must be at least 8 characters[/yellow]")
                    continue
                password_confirm = getpass.getpass("Confirm: ")
                if new_password != password_confirm:
                    console.print("[yellow]⚠ Passwords don't match, try again[/yellow]")
                    continue
                break
        
        # Update config
        settings.set_neo4j_password(new_password)
        settings.save()
        console.print(f"\n[green]✓ Password saved to {settings.config_path()}[/green]")
        
        # Check if Neo4j container exists
        try:
            result = subprocess.run(
                ["docker", "ps", "-a", "--filter", "name=orion-neo4j", "--format", "{{.Names}}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            if "orion-neo4j" in result.stdout:
                console.print("\n[yellow]Updating Neo4j container with new password...[/yellow]")
                
                # Stop and remove old container
                subprocess.run(["docker", "stop", "orion-neo4j"], capture_output=True, check=False)
                subprocess.run(["docker", "rm", "orion-neo4j"], capture_output=True, check=False)
                
                # Create new container with new password
                subprocess.run(
                    [
                        "docker", "run", "-d", "--name", "orion-neo4j",
                        "-p", "7474:7474", "-p", "7687:7687",
                        "-e", f"NEO4J_AUTH=neo4j/{new_password}",
                        "neo4j:5"
                    ],
                    capture_output=True,
                    check=True
                )
                console.print("[green]✓ Neo4j container restarted with new password[/green]")
            else:
                console.print("\n[dim]No Neo4j container found - password updated in config only[/dim]")
                console.print("[dim]Run 'orion init' to create the container[/dim]")
        except Exception as e:
            console.print(f"\n[yellow]⚠ Password updated in config, but container update failed: {e}[/yellow]")
            console.print("[dim]You may need to manually restart the Neo4j container[/dim]")
        
        console.print("\n[bold green]✅ Password reset complete![/bold green]\n")
        return 0

    console.print(f"[red]Unknown configuration subcommand '{subcommand}'.[/red]")
    return 1
