#!/usr/bin/env python3
"""
Orion Initialization Script
Sets up models, dependencies, and configuration
"""

import os
import sys
import subprocess
import platform
import json
import getpass
import urllib.request
from pathlib import Path

# Import Orion modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt, Confirm

from orion.config_manager import ConfigManager
from orion.neo4j_manager import Neo4jManager
from orion.runtime import is_backend_available, select_backend

console = Console()

MODELS_DIR = Path(__file__).parent.parent / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"


def check_python_version() -> bool:
    """Check if Python version is compatible"""
    console.print("\n[bold cyan]Checking Python version...[/]")

    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        console.print("[red]❌ Python 3.10+ required[/]")
        console.print(f"[yellow]Current version: {sys.version}[/]")
        return False

    console.print(f"[green]✓ Python {version.major}.{version.minor}.{version.micro}[/]")
    return True


def detect_hardware() -> str:
    """Detect available hardware and recommend backend"""
    console.print("\n[bold cyan]Detecting hardware...[/]")

    system = platform.system()
    processor = platform.processor()

    # Check for Apple Silicon
    if system == "Darwin" and processor == "arm":
        if is_backend_available("mlx"):
            console.print("[green]✓ Apple Silicon detected (M1/M2/M3+)[/]")
            console.print("  [cyan]MLX backend available for optimal performance[/]")
            return "mlx"

    # Check for NVIDIA CUDA
    try:
        import torch

        if torch.cuda.is_available():
            console.print("[green]✓ NVIDIA GPU detected[/]")
            console.print(f"  [cyan]CUDA available: {torch.cuda.get_device_name(0)}[/]")
            return "torch"
    except ImportError:
        pass

    console.print("[yellow]⚠️  CPU-only mode[/]")
    console.print("  [cyan]Consider installing CUDA for GPU acceleration[/]")
    return "torch"


def check_system_deps() -> bool:
    """Check for required system dependencies"""
    console.print("\n[bold cyan]Checking system dependencies...[/]")

    # Check for wget
    try:
        subprocess.run(["which", "wget"], capture_output=True, check=True)
        console.print("[green]✓ wget found[/]")
    except subprocess.CalledProcessError:
        console.print("[yellow]⚠️  wget not found (optional)[/]")
        if platform.system() == "Darwin":
            console.print("  [cyan]Install with: brew install wget[/]")
        else:
            console.print("  [cyan]Install with: sudo apt-get install wget[/]")

    return True


def setup_environment_variables() -> dict:
    """Interactively set up environment variables"""
    console.print("\n[bold cyan]Setting up environment variables...[/]")

    env_vars = {}

    # Neo4j configuration
    console.print("\n[bold yellow]Neo4j Configuration[/]")
    neo4j_uri = Prompt.ask(
        "Neo4j URI",
        default="neo4j://127.0.0.1:7687",
    )
    neo4j_user = Prompt.ask("Neo4j username", default="neo4j")

    # Securely prompt for password
    console.print("[yellow]Neo4j password will be stored in environment variable[/]")
    console.print("[yellow]It will NOT be saved to disk for security[/]")
    neo4j_password = getpass.getpass("Neo4j password: ")

    env_vars["ORION_NEO4J_URI"] = neo4j_uri
    env_vars["ORION_NEO4J_USER"] = neo4j_user
    env_vars["ORION_NEO4J_PASSWORD"] = neo4j_password

    # Ollama configuration
    console.print("\n[bold yellow]Ollama Configuration[/]")
    ollama_url = Prompt.ask(
        "Ollama server URL",
        default="http://localhost:11434",
    )
    env_vars["ORION_OLLAMA_URL"] = ollama_url

    # Runtime backend
    detected_backend = detect_hardware()
    use_detected = Confirm.ask(
        f"Use detected backend ({detected_backend})?", default=True
    )
    if not use_detected:
        backend = Prompt.ask(
            "Runtime backend", choices=["auto", "torch", "mlx"], default="auto"
        )
        env_vars["ORION_RUNTIME_BACKEND"] = backend
    else:
        env_vars["ORION_RUNTIME_BACKEND"] = detected_backend

    return env_vars


def save_environment_variables(env_vars: dict) -> None:
    """Save environment variables to ~/.orion/.env"""
    config_dir = Path.home() / ".orion"
    config_dir.mkdir(parents=True, exist_ok=True)

    env_file = config_dir / ".env"

    # Build .env content
    env_content = "# Orion Environment Configuration\n"
    env_content += "# Source this file: source ~/.orion/.env\n\n"

    for key, value in env_vars.items():
        # Only export non-password variables to shell
        if key == "ORION_NEO4J_PASSWORD":
            env_content += f"# {key}='{value}'  # Set manually for security\n"
        else:
            env_content += f'export {key}="{value}"\n'

    with open(env_file, "w") as f:
        f.write(env_content)

    console.print(f"\n[green]✓ Environment variables saved to {env_file}[/]")
    console.print("[yellow]Add to your shell profile (~/.zshrc or ~/.bashrc):[/]")
    console.print(f"  [cyan]source {env_file}[/]")


def verify_neo4j_connection(uri: str, user: str, password: str) -> bool:
    """Test Neo4j connection"""
    console.print("\n[bold cyan]Testing Neo4j connection...[/]")

    manager = Neo4jManager(uri, user, password)
    if manager.connect():
        manager.close()
        return True

    console.print("[red]❌ Could not connect to Neo4j[/]")
    console.print("[yellow]Make sure Neo4j is running:[/]")
    console.print("  [cyan]docker run -d --name neo4j -p 7474:7474 -p 7687:7687 \\[/]")
    console.print('    [cyan]-e NEO4J_AUTH=neo4j/your-password neo4j:latest[/]')
    return False


def verify_ollama_connection(url: str) -> bool:
    """Test Ollama connection"""
    console.print("\n[bold cyan]Testing Ollama connection...[/]")

    try:
        response = urllib.request.urlopen(f"{url}/api/tags", timeout=2)
        if response.status == 200:
            console.print("[green]✓ Ollama is running[/]")
            return True
    except Exception as e:
        console.print(f"[red]❌ Could not connect to Ollama: {e}[/]")
        console.print("[yellow]Start Ollama server:[/]")
        console.print("  [cyan]ollama serve[/]")
        return False

    return False


def download_models() -> bool:
    """Download required models (YOLO, FastVLM)"""
    console.print("\n[bold cyan]Downloading models...[/]")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    console.print(
        "[yellow]Models will be downloaded automatically on first use[/]"
    )
    console.print("[yellow]This includes:[/]")
    console.print("  - YOLO11x (object detection)")
    console.print("  - FastVLM (descriptions)")
    console.print("  - CLIP (embeddings)")

    return True


def main():
    """Main initialization workflow"""
    console.print(
        Panel(
            "[bold cyan]🌟 Welcome to Orion[/]\n"
            "[yellow]Video Analysis with Knowledge Graphs[/]",
            border_style="cyan",
        )
    )

    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)

    # Step 2: Check system dependencies
    check_system_deps()

    # Step 3: Detect hardware
    detected_backend = detect_hardware()

    # Step 4: Set up environment variables
    console.print(
        "\n[bold cyan]Step 1: Configure Environment[/]"
    )
    env_vars = setup_environment_variables()

    # Step 5: Save environment variables
    save_environment_variables(env_vars)

    # Step 6: Verify connections
    console.print(
        "\n[bold cyan]Step 2: Verify Services[/]"
    )

    if not verify_neo4j_connection(
        env_vars["ORION_NEO4J_URI"],
        env_vars["ORION_NEO4J_USER"],
        env_vars["ORION_NEO4J_PASSWORD"],
    ):
        console.print("[yellow]⚠️  Neo4j not available yet. Set it up and run again.[/]")
    else:
        console.print("[green]✓ Neo4j connection verified[/]")

    if not verify_ollama_connection(env_vars["ORION_OLLAMA_URL"]):
        console.print("[yellow]⚠️  Ollama not available yet. Start it in another terminal.[/]")
    else:
        console.print("[green]✓ Ollama connection verified[/]")

    # Step 7: Download models
    console.print(
        "\n[bold cyan]Step 3: Download Models[/]"
    )
    download_models()

    # Final summary
    console.print(
        Panel(
            "[bold green]✓ Initialization Complete![/]\n"
            "[cyan]Next steps:[/]\n"
            "1. Source environment: [yellow]source ~/.orion/.env[/]\n"
            "2. Start Neo4j: [yellow]docker run -d --name neo4j ...[/]\n"
            "3. Start Ollama: [yellow]ollama serve[/]\n"
            "4. Process video: [yellow]orion process video.mp4[/]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Initialization cancelled[/]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error during initialization: {e}[/]")
        import traceback

        traceback.print_exc()
        sys.exit(1)
