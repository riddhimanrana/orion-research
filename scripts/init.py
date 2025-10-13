#!/usr/bin/env python3
"""
Orion Initialization Script
Sets up models, dependencies, and environment
"""

import os
import sys
import subprocess
import platform
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import requests

console = Console()

MODELS_DIR = Path(__file__).parent.parent / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"

# Model configurations
MODELS = {
    "yolo11m": {
        "path": WEIGHTS_DIR / "yolo11m.pt",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt",
        "size": "20 MB",
        "description": "Object detection model"
    },
    "fastvlm-0.5b": {
        "path": CHECKPOINTS_DIR / "llava-fastvithd_0.5b_stage3",
        "hf_repo": "riddhimanrana/fastvlm-0.5b-captions",
        "size": "600 MB",
        "description": "Vision-language model for scene descriptions"
    }
}

OLLAMA_MODELS = {
    "gemma3:4b": {
        "size": "3.3 GB",
        "description": "Question answering and event composition"
    },
    "embeddinggemma": {
        "size": "622 MB",
        "description": "Text embeddings for semantic search"
    }
}


def check_python_version():
    """Check if Python version is compatible"""
    console.print("\n[bold cyan]Checking Python version...[/]")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        console.print("[red]❌ Python 3.10+ required[/]")
        console.print(f"[yellow]Current version: {sys.version}[/]")
        return False
    
    console.print(f"[green]✓ Python {version.major}.{version.minor}.{version.micro}[/]")
    return True


def check_ollama():
    """Check if Ollama is installed and running"""
    console.print("\n[bold cyan]Checking Ollama...[/]")
    
    # Check if installed
    try:
        result = subprocess.run(['which', 'ollama'], capture_output=True, text=True)
        if result.returncode != 0:
            console.print("[yellow]⚠️  Ollama not found[/]")
            console.print("\nInstall Ollama:")
            
            if platform.system() == "Darwin":
                console.print("  [cyan]brew install ollama[/]")
            else:
                console.print("  [cyan]curl -fsSL https://ollama.com/install.sh | sh[/]")
            
            return False
    except Exception:
        console.print("[red]❌ Could not check Ollama installation[/]")
        return False
    
    # Check if running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            console.print("[green]✓ Ollama is installed and running[/]")
            return True
    except:
        console.print("[yellow]⚠️  Ollama installed but not running[/]")
        console.print("\nStart Ollama:")
        console.print("  [cyan]ollama serve[/]")
        return False
    
    return True


def check_neo4j():
    """Check if Neo4j is accessible"""
    console.print("\n[bold cyan]Checking Neo4j (optional)...[/]")
    
    try:
        from neo4j import GraphDatabase
        
        # Try to connect
        driver = GraphDatabase.driver(
            "neo4j://127.0.0.1:7687",
            auth=("neo4j", "orion123"),
            max_connection_lifetime=10
        )
        
        with driver.session() as session:
            session.run("RETURN 1")
        
        driver.close()
        console.print("[green]✓ Neo4j connected (neo4j://127.0.0.1:7687)[/]")
        return True
        
    except Exception as e:
        console.print("[yellow]⚠️  Neo4j not running (optional)[/]")
        console.print("\nTo use knowledge graph features, start Neo4j:")
        console.print("  [cyan]docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/orion123 neo4j[/]")
        return False


def download_file(url: str, dest: Path, description: str):
    """Download file with progress bar"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task(f"Downloading {description}...", total=total_size)
        
        with open(dest, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                progress.update(task, advance=len(chunk))
    
    console.print(f"[green]✓ Downloaded {description}[/]")


def download_from_hf(repo_id: str, dest: Path, description: str):
    """Download model from Hugging Face"""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"\n[cyan]Downloading {description} from Hugging Face...[/]")
    
    try:
        from huggingface_hub import snapshot_download
        
        snapshot_download(
            repo_id=repo_id,
            local_dir=dest,
            local_dir_use_symlinks=False
        )
        
        console.print(f"[green]✓ Downloaded {description}[/]")
        return True
        
    except Exception as e:
        console.print(f"[red]❌ Failed to download {description}: {e}[/]")
        return False


def setup_models():
    """Download and setup required models"""
    console.print("\n[bold cyan]Setting up models...[/]")
    
    # YOLO11m
    yolo_config = MODELS["yolo11m"]
    if not yolo_config["path"].exists():
        console.print(f"\n[yellow]Downloading YOLO11m ({yolo_config['size']})...[/]")
        download_file(yolo_config["url"], yolo_config["path"], "YOLO11m")
    else:
        console.print(f"[green]✓ YOLO11m already exists[/]")
    
    # FastVLM
    fastvlm_config = MODELS["fastvlm-0.5b"]
    if not fastvlm_config["path"].exists():
        console.print(f"\n[yellow]Downloading FastVLM-0.5B ({fastvlm_config['size']})...[/]")
        download_from_hf(
            fastvlm_config["hf_repo"],
            fastvlm_config["path"],
            "FastVLM-0.5B"
        )
    else:
        console.print(f"[green]✓ FastVLM-0.5B already exists[/]")


def setup_ollama_models():
    """Pull Ollama models"""
    console.print("\n[bold cyan]Setting up Ollama models...[/]")
    
    try:
        # Check which models are already installed
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        installed = result.stdout
        
        for model_name, config in OLLAMA_MODELS.items():
            if model_name in installed:
                console.print(f"[green]✓ {model_name} already installed[/]")
            else:
                console.print(f"\n[yellow]Pulling {model_name} ({config['size']})...[/]")
                console.print(f"[dim]{config['description']}[/]")
                
                subprocess.run(['ollama', 'pull', model_name])
                console.print(f"[green]✓ Installed {model_name}[/]")
    
    except Exception as e:
        console.print(f"[red]❌ Failed to setup Ollama models: {e}[/]")


def create_directories():
    """Create necessary directories"""
    console.print("\n[bold cyan]Creating directory structure...[/]")
    
    dirs = [
        WEIGHTS_DIR,
        CHECKPOINTS_DIR,
        Path("data/examples"),
        Path("data/testing"),
        Path("production"),
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    console.print("[green]✓ Directories created[/]")


def print_summary():
    """Print setup summary"""
    table = Table(title="Setup Summary", show_header=True, header_style="bold cyan")
    table.add_column("Component", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Details")
    
    # Check models
    yolo_exists = MODELS["yolo11m"]["path"].exists()
    fastvlm_exists = MODELS["fastvlm-0.5b"]["path"].exists()
    
    table.add_row(
        "YOLO11m",
        "[green]✓[/]" if yolo_exists else "[red]✗[/]",
        str(MODELS["yolo11m"]["path"]) if yolo_exists else "Not found"
    )
    
    table.add_row(
        "FastVLM-0.5B",
        "[green]✓[/]" if fastvlm_exists else "[red]✗[/]",
        str(MODELS["fastvlm-0.5b"]["path"]) if fastvlm_exists else "Not found"
    )
    
    # Check Ollama
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        installed = result.stdout
        
        for model_name in OLLAMA_MODELS.keys():
            is_installed = model_name in installed
            table.add_row(
                model_name,
                "[green]✓[/]" if is_installed else "[red]✗[/]",
                "Installed" if is_installed else "Not installed"
            )
    except:
        table.add_row("Ollama Models", "[red]✗[/]", "Ollama not available")
    
    console.print("\n")
    console.print(table)


def main():
    """Main initialization routine"""
    console.print(Panel.fit(
        "[bold cyan]Orion Initialization[/]\n"
        "[dim]Setting up models, dependencies, and environment[/]",
        border_style="cyan"
    ))
    
    # Step 1: Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    check_ollama()
    check_neo4j()
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Setup models
    setup_models()
    
    # Step 4: Setup Ollama models
    ollama_available = check_ollama()
    if ollama_available:
        setup_ollama_models()
    else:
        console.print("\n[yellow]⚠️  Skipping Ollama models (Ollama not available)[/]")
    
    # Step 5: Summary
    console.print("\n")
    print_summary()
    
    # Step 6: Next steps
    console.print("\n[bold green]✓ Setup complete![/]\n")
    console.print("Next steps:")
    console.print("  1. [cyan]./orion analyze data/examples/video1.mp4[/]")
    console.print("  2. [cyan]./orion analyze video.mp4 -i[/] (interactive Q&A)")
    console.print("  3. [cyan]./orion --help[/] (see all commands)\n")


if __name__ == "__main__":
    main()
