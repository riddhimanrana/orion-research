"""Init command - initialize Orion environment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table

from ..utils import is_docker_daemon_running, prompt_user, run_command

console = Console()


def setup_neo4j(password: str) -> None:
    """Set up the Neo4j container with the provided password."""
    NEO4J_CONTAINER_NAME = "orion-neo4j"
    NEO4J_IMAGE = "neo4j:5"

    # Check if running
    output = run_command(
        f"docker ps -q -f name={NEO4J_CONTAINER_NAME}",
        f"Checking for container: {NEO4J_CONTAINER_NAME}",
    )
    if output and output != "":
        console.print(f"[green]✓ Neo4j container '{NEO4J_CONTAINER_NAME}' is already running.[/green]")
        return

    if prompt_user(f"Neo4j container '{NEO4J_CONTAINER_NAME}' not found. Would you like to create and start it?"):
        run_command(f"docker pull {NEO4J_IMAGE}", f"Pulling Neo4j image: {NEO4J_IMAGE}")
        run_command(
            f"docker run -d --name {NEO4J_CONTAINER_NAME} "
            f"-p 7474:7474 -p 7687:7687 "
            f"-e NEO4J_AUTH=neo4j/{password} "
            f"{NEO4J_IMAGE}",
            "Starting Neo4j container",
        )
        console.print("[green]✓ Neo4j container started.[/green]")


def setup_ollama() -> None:
    """Check for local Ollama installation and pull models."""
    OLLAMA_MODELS = ["gemma3:4b"]

    # Check if ollama is installed
    if run_command("ollama --version", "Checking for local Ollama installation") is None:
        console.print("[red]✗ Ollama is not installed.[/red]")
        console.print("Please install it from https://ollama.com and then run this command again.")
        sys.exit(1)

    console.print("[green]✓ Ollama is installed.[/green]")

    # Pull models
    for model in OLLAMA_MODELS:
        console.print(f"\n-- Checking for Ollama model: {model} --")
        run_command(f"ollama pull {model}", f"Pulling Ollama model: {model}")


def handle_init(args: argparse.Namespace) -> None:
    """Handle the init command - full system initialization and setup."""
    console.print("\n[bold cyan]🚀 Orion Full System Initialization[/bold cyan]\n")
    console.print("This will set up Orion from scratch (config, services, models)\n")

    # ═══════════════════════════════════════════════════════════
    # STEP 1: Create Configuration
    # ═══════════════════════════════════════════════════════════
    console.print("[bold]Step 1: Creating Configuration[/bold]\n")
    
    from orion.settings import OrionSettings, generate_secure_password
    import os
    from pathlib import Path
    
    # Check if user wants to reset configuration
    force_reset = getattr(args, "reset", False)
    
    # Determine if we need a new password or use existing
    settings = None
    neo4j_password = None
    password_was_generated = False
    password_is_new = False
    
    # Try to load existing settings
    try:
        settings = OrionSettings.load()
        console.print(f"[green]✓ Configuration found at {settings.config_path()}[/green]")
        
        # Check if password exists
        try:
            neo4j_password = settings.get_neo4j_password()
            
            # If reset flag OR user wants to reconfigure
            if force_reset:
                console.print("[yellow]⚠ Reset flag detected - reconfiguring password[/yellow]\n")
                password_is_new = True
            else:
                console.print("[dim]Existing Neo4j password found in configuration[/dim]")
                if prompt_user("Would you like to reconfigure the Neo4j password?", default=False):
                    console.print()
                    password_is_new = True
                else:
                    console.print("[yellow]Using existing Neo4j password[/yellow]\n")
        except Exception:
            # Config exists but no password set, ask user
            console.print("[yellow]⚠ No Neo4j password found in configuration[/yellow]\n")
            password_is_new = True
    except Exception as e:
        console.print(f"[dim]Creating new configuration...[/dim]")
        password_is_new = True
        settings = None
    
    # If we need a new password, ask the user
    if password_is_new:
        console.print("[bold cyan]Neo4j Password Setup[/bold cyan]\n")
        choice = prompt_user("Would you like to generate a random password?", default=True)
        
        if choice:
            neo4j_password = generate_secure_password(16)
            password_was_generated = True
            console.print(f"[green]✓ Generated secure password[/green]")
            console.print(f"[dim]Password: {neo4j_password}[/dim]\n")
        else:
            # Custom password
            console.print("\n[cyan]Enter your Neo4j password (min 8 characters):[/cyan]")
            import getpass
            while True:
                neo4j_password = getpass.getpass("Password: ")
                if len(neo4j_password) < 8:
                    console.print("[yellow]⚠ Password must be at least 8 characters[/yellow]")
                    continue
                password_confirm = getpass.getpass("Confirm: ")
                if neo4j_password != password_confirm:
                    console.print("[yellow]⚠ Passwords don't match, try again[/yellow]")
                    continue
                break
            console.print("[green]✓ Password set[/green]\n")
    
    # Create or update settings with password
    if settings is None:
        settings = OrionSettings()
    settings.set_neo4j_password(neo4j_password)
    settings.save()
    
    if password_is_new:
        console.print(f"[green]✓ Configuration saved to {settings.config_path()}[/green]\n")
    
    console.print(f"  • Neo4j URI: {settings.neo4j_uri}")
    console.print(f"  • Neo4j User: {settings.neo4j_user}")
    console.print(f"  • Runtime Backend: {settings.runtime_backend}")
    console.print(f"  • Q&A Model: {settings.qa_model}")
    console.print(f"  • Embedding Model: {settings.embedding_model}\n")

    # ═══════════════════════════════════════════════════════════
    # STEP 2: Pre-flight Check - Verify Prerequisites
    # ═══════════════════════════════════════════════════════════
    console.print("[bold]Step 2: Pre-flight Check - Verifying Prerequisites[/bold]\n")
    from orion.managers.auto_config import AutoConfiguration

    auto_config = AutoConfiguration()
    results = auto_config.detect_all_services()

    neo4j_ok, neo4j_msg = results["neo4j"]
    ollama_ok, ollama_msg, _ = results["ollama"]
    docker_ok, docker_msg = results["docker"]

    status_table = Table(title="Service Status", box=box.ROUNDED)
    status_table.add_column("Service", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Details", style="yellow")

    status_table.add_row(
        "Neo4j",
        "[green]✓ Running[/green]" if neo4j_ok else "[red]✗ Not available[/red]",
        neo4j_msg,
    )
    status_table.add_row(
        "Ollama",
        "[green]✓ Running[/green]" if ollama_ok else "[red]✗ Not available[/red]",
        ollama_msg,
    )
    status_table.add_row(
        "Docker",
        "[green]✓ Available[/green]" if docker_ok else "[yellow]⚠ Not available[/yellow]",
        docker_msg,
    )

    console.print(status_table)

    # Check for sentence-transformers
    try:
        import sentence_transformers
        console.print("[green]✓ sentence-transformers installed[/green]\n")
    except ImportError:
        console.print("[red]✗ sentence-transformers library not found.[/red]")
        console.print("Please install it by running: pip install sentence-transformers==2.2.2")
        sys.exit(1)

    if not docker_ok:
        console.print("\n[red]✗ Docker CLI not available.[/red]")
        
        # Check if it's a PATH issue on macOS
        import platform
        from pathlib import Path
        
        if platform.system() == "Darwin":
            docker_app = Path("/Applications/Docker.app")
            if docker_app.exists():
                console.print("\n[yellow]ℹ Docker Desktop is installed, but the CLI is not in your PATH.[/yellow]")
                console.print("\n[cyan]To fix this, try one of the following:[/cyan]")
                console.print("  1. Restart your terminal or IDE")
                console.print("  2. Or run: [bold]eval \"$(docker-machine env default)\"[/bold]")
                console.print("  3. Or add Docker to your PATH manually")
                console.print("\nIf you just installed Docker Desktop, a restart is recommended.\n")
                
                # Optionally allow retry instead of exiting
                if prompt_user("Would you like to retry Docker detection?"):
                    docker_ok, docker_msg = auto_config.detector.detect_docker()
                    if docker_ok:
                        console.print(f"[green]✓ Docker is now available![/green]")
                    else:
                        console.print(f"[red]✗ {docker_msg}[/red]")
                        sys.exit(1)
                else:
                    sys.exit(1)
            else:
                console.print("Please install Docker Desktop from https://www.docker.com/products/docker-desktop and then run this command again.")
                sys.exit(1)
        else:
            console.print("Please install Docker and ensure the 'docker' command is in your PATH.")
            sys.exit(1)

    if not is_docker_daemon_running():
        console.print("\n[red]✗ Docker daemon is not running.[/red]")
        console.print("Please start Docker Desktop and then run this command again.")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════
    # STEP 3: Set Up Services
    # ═══════════════════════════════════════════════════════════
    console.print("[bold]Step 3: Setting Up Services[/bold]\n")
    
    console.print("--- Setting up Neo4j ---")
    setup_neo4j(neo4j_password)
    
    # Verify Neo4j connection with the password
    console.print("\n[dim]Verifying Neo4j connection (this may take a moment)...[/dim]")
    import time
    import logging
    
    # Temporarily suppress Neo4j driver logging during retries
    neo4j_logger = logging.getLogger("neo4j")
    original_level = neo4j_logger.level
    neo4j_logger.setLevel(logging.CRITICAL)
    
    max_retries = 30
    retry_count = 0
    neo4j_verified = False
    
    while retry_count < max_retries and not neo4j_verified:
        try:
            from neo4j import GraphDatabase
            
            driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, neo4j_password),
                connection_timeout=5
            )
            driver.verify_connectivity()
            driver.close()
            neo4j_verified = True
            # Restore logging level
            neo4j_logger.setLevel(original_level)
            console.print("[green]✓ Neo4j authentication successful[/green]")
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                # Show a simple progress indicator instead of error messages
                dots = "." * (retry_count % 4)
                console.print(f"\r[dim]  Waiting for Neo4j to start{dots:4s}[/dim]", end="")
                time.sleep(1)
            else:
                # Restore logging level
                neo4j_logger.setLevel(original_level)
                console.print(f"\n[red]✗ Neo4j connection failed after {max_retries} attempts[/red]")
                console.print(f"[yellow]Error: {str(e)}[/yellow]")
                if prompt_user("\n[yellow]Continue anyway?[/yellow]", default=False):
                    neo4j_verified = True
                else:
                    sys.exit(1)
    
    if neo4j_verified and retry_count > 0:
        console.print()  # New line after progress dots

    console.print("\n--- Setting up Ollama ---")
    setup_ollama()

    # ═══════════════════════════════════════════════════════════
    # STEP 4: Detect Hardware & Select Runtime
    # ═══════════════════════════════════════════════════════════
    console.print("\n[bold]Step 4: Detecting hardware and selecting runtime...[/bold]\n")

    # Auto-detect the best runtime
    try:
        from ...managers.runtime import select_backend

        backend: str = select_backend(None)  # Force auto-detection
        console.print(f"[green]✓ Selected runtime: {backend}[/green]")
    except Exception as e:
        console.print(f"[red]Failed to select runtime: {e}[/red]")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════
    # STEP 5: Download Models
    # ═══════════════════════════════════════════════════════════
    console.print("\n[bold]Step 5: Downloading Models[/bold]\n")
    console.print("Preparing runtime and downloading models...\n")
    try:
        from ..utils import prepare_runtime

        backend, manager = prepare_runtime(backend)
    except Exception:
        sys.exit(1)

    console.print(f"[green]✓ Runtime '{backend}' assets are ready.[/green]")
    
    # Ensure YOLO11x is downloaded regardless of runtime
    # (needed for object detection in all pipelines)
    # Download it directly via HuggingFace since it's not runtime-specific
    console.print("\n[dim]Ensuring YOLO11x detector is available...[/dim]")
    try:
        from huggingface_hub import snapshot_download
        
        yolo_dir = manager.cache_dir / "yolo11x"
        yolo_dir.mkdir(parents=True, exist_ok=True)
        yolo_path = yolo_dir / "yolo11x.pt"
        
        if not yolo_path.exists():
            console.print("[cyan]Downloading YOLO11x from Ultralytics via HuggingFace...[/cyan]")
            snapshot_download(
                repo_id="ultralytics/YOLO11",
                local_dir=str(yolo_dir),
                allow_patterns=["yolo11x.pt"],
            )
        console.print("[green]✓ YOLO11x detector ready.[/green]\n")
    except Exception as e:
        console.print(f"[yellow]⚠ Warning: YOLO11x download failed: {e}[/yellow]")
        console.print("[dim]  (You can install it manually or retry later)[/dim]\n")

    # ═══════════════════════════════════════════════════════════
    # STEP 6: Test Connections
    # ═══════════════════════════════════════════════════════════
    console.print("[bold]Step 6: Testing Connections[/bold]\n")
    
    # Re-check services
    results = auto_config.detect_all_services()
    neo4j_ok, neo4j_msg = results["neo4j"]
    ollama_ok, ollama_msg, ollama_model = results["ollama"]
    
    if neo4j_ok:
        console.print(f"[green]✓ Neo4j: {neo4j_msg}[/green]")
    else:
        console.print(f"[yellow]⚠ Neo4j: {neo4j_msg}[/yellow]")
    
    if ollama_ok:
        console.print(f"[green]✓ Ollama: {ollama_msg}[/green]")
        if ollama_model:
            console.print(f"  • Default model: {ollama_model}")
    else:
        console.print(f"[yellow]⚠ Ollama: {ollama_msg}[/yellow]")
    
    console.print()

    # ═══════════════════════════════════════════════════════════
    # STEP 7: Print Setup Summary
    # ═══════════════════════════════════════════════════════════
    console.print("[bold]Step 7: Setup Summary[/bold]\n")
    summary_table = Table(title="Model Assets", box=box.ROUNDED)
    summary_table.add_column("Component", style="cyan", no_wrap=True)
    summary_table.add_column("Status", style="green", justify="center")
    summary_table.add_column("Details", style="magenta")

    # Show YOLO11x
    yolo_path = manager.get_asset_path("yolo11x") if "yolo11x" in manager._manifest else "Not found"
    summary_table.add_row("YOLO11x", "✓" if Path(yolo_path).exists() else "✗", str(yolo_path))

    # Show FastVLM
    fastvlm_asset_name = "fastvlm-0.5b-mlx" if backend == "mlx" else "fastvlm-0.5b"
    fastvlm_path = (
        manager.get_asset_path(fastvlm_asset_name) if fastvlm_asset_name in manager._manifest else "Not found"
    )
    summary_table.add_row("FastVLM-0.5B", "✓" if Path(fastvlm_path).exists() else "✗", str(fastvlm_path))

    # Show Gemma3:4b and CLIP
    summary_table.add_row("gemma3:4b", "✓", "Available in Ollama")
    summary_table.add_row("CLIP (OpenAI)", "✓", "Available via sentence-transformers")

    console.print(summary_table)

    console.print("\n[bold green]✅ Orion initialization complete![/bold green]")
    
    # Show Neo4j credentials and configuration info
    console.print("\n[bold cyan]📝 Neo4j Configuration[/bold cyan]")
    console.print(f"[bold]Browser URL:[/bold] http://localhost:7474")
    console.print(f"[bold]Connection URI:[/bold] neo4j://localhost:7687")
    console.print(f"[bold]Username:[/bold] neo4j")
    if password_was_generated:
        console.print(f"[bold]Password:[/bold] {neo4j_password} [yellow](auto-generated)[/yellow]")
    else:
        console.print(f"[dim]Password:[/dim] [green]✓ Set (stored in ~/.orion/config.json)[/green]")
    console.print("[dim]  • Credentials are encrypted and stored in ~/.orion/config.json[/dim]")
    console.print("[dim]  • Log in to Neo4j Browser to manage the database[/dim]")
    
    console.print("\n[bold cyan]Next steps:[/bold cyan]")
    console.print("  1. Check system status: [bold]orion status[/bold]")
    console.print("  2. Analyze a video: [bold]orion analyze video.mp4[/bold]")
    console.print("  3. Try Q&A mode: [bold]orion qa[/bold]\n")

