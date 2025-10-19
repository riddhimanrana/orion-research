"""Model asset management for Orion runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import requests
from huggingface_hub import snapshot_download
from rich.console import Console
from rich.progress import Progress

console = Console()

_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class ModelAsset:
    name: str
    target: str
    runtimes: List[str]
    primary_file: Optional[str] = None
    url: Optional[str] = None
    sha256: Optional[str] = None
    huggingface_repo: Optional[str] = None
    huggingface_revision: Optional[str] = None
    allow_patterns: Optional[Sequence[str]] = None
    description: Optional[str] = None


class ModelManager:
    """Coordinate model downloads, caching, and environment settings for Orion."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir or self._default_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._configure_environment()
        self._manifest = self._load_manifest()

    @staticmethod
    def _default_cache_dir() -> Path:
        project_root = Path(__file__).resolve().parents[3]
        return project_root / "models"

    @staticmethod
    def _is_apple_silicon() -> bool:
        """Check if running on Apple Silicon (M1/M2/M3/etc)."""
        return platform.system() == "Darwin" and platform.processor() == "arm"

    def _configure_environment(self) -> None:
        torch_home = self.cache_dir / "_torch"
        hf_home = self.cache_dir / "_huggingface"
        ultralytics_home = self.cache_dir / "_ultralytics"

        torch_home.mkdir(parents=True, exist_ok=True)
        hf_home.mkdir(parents=True, exist_ok=True)
        ultralytics_home.mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("TORCH_HOME", str(torch_home))
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_home / "hub"))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "transformers"))
        os.environ.setdefault("ULTRALYTICS_CACHE_DIR", str(ultralytics_home))
        os.environ.setdefault("YOLO_DIR", str(ultralytics_home))

    def _load_manifest(self) -> Dict[str, ModelAsset]:
        manifest_path = Path(__file__).with_name(_MANIFEST_FILENAME)
        with manifest_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)

        assets: Dict[str, ModelAsset] = {}
        for entry in raw.get("assets", []):
            asset = ModelAsset(
                name=entry["name"],
                target=entry.get("target", entry["name"]),
                runtimes=entry.get("runtimes", []),
                primary_file=entry.get("primary_file"),
                url=entry.get("url"),
                sha256=entry.get("sha256"),
                huggingface_repo=entry.get("huggingface_repo"),
                huggingface_revision=entry.get("huggingface_revision"),
                allow_patterns=entry.get("allow_patterns"),
                description=entry.get("description"),
            )
            assets[asset.name] = asset
        return assets

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def model_subdir(self, name: str) -> Path:
        path = self.cache_dir / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _asset_dir(self, asset: ModelAsset) -> Path:
        return self.model_subdir(asset.target)

    def asset_path(self, asset: ModelAsset) -> Path:
        directory = self._asset_dir(asset)
        if asset.primary_file:
            return directory / asset.primary_file
        return directory

    def get_asset_path(self, name: str) -> Path:
        asset = self._manifest[name]
        return self.asset_path(asset)

    def get_asset_dir(self, name: str) -> Path:
        asset = self._manifest[name]
        return self._asset_dir(asset)

    # ------------------------------------------------------------------
    # Manifest queries
    # ------------------------------------------------------------------
    def list_runtime_assets(self, runtime: str) -> List[ModelAsset]:
        return [asset for asset in self._manifest.values() if runtime in asset.runtimes]

    def assets_ready(self, runtime: str) -> bool:
        for asset in self.list_runtime_assets(runtime):
            if not self._asset_exists(asset):
                return False
        return True

    def ensure_runtime_assets(self, runtime: str, *, download: bool = True) -> None:
        assets = self.list_runtime_assets(runtime)
        if not assets:
            console.print(
                f"[yellow]No assets registered for runtime '{runtime}'.[/yellow]"
            )
            return

        for asset in assets:
            self._prepare_asset(asset, download=download)

    def ensure_asset(self, asset_name: str, *, download: bool = True) -> Path:
        if asset_name not in self._manifest:
            raise KeyError(f"Asset '{asset_name}' is not defined in the manifest.")
        asset = self._manifest[asset_name]
        return self._prepare_asset(asset, download=download)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _asset_exists(self, asset: ModelAsset) -> bool:
        target = self._asset_dir(asset)
        if asset.primary_file:
            return (target / asset.primary_file).exists()
        return any(target.iterdir())

    def _prepare_asset(self, asset: ModelAsset, *, download: bool) -> Path:
        target_dir = self._asset_dir(asset)

        if self._asset_exists(asset):
            return target_dir

        if not download:
            raise FileNotFoundError(
                f"Model '{asset.name}' is missing from {target_dir} and downloads are disabled."
            )

        # Special case: MLX backend - download pre-converted Apple model with wget
        if "mlx" in asset.runtimes:
            if not self._is_apple_silicon():
                console.print("[yellow]MLX backend requires Apple Silicon (M1/M2/M3).[/yellow]")
                return target_dir
            
            if not asset.url:
                console.print(f"[red]Missing Apple CDN URL for MLX asset '{asset.name}'[/red]")
                raise ValueError(f"No download URL configured for MLX asset: {asset.name}")
            
            console.print(f"[cyan]🍎 Apple Silicon detected - downloading pre-converted fp16 model for MLX[/cyan]")
            console.print(f"[yellow]Downloading from Apple CDN: {asset.url}[/yellow]")
            
            # Download using wget (Apple Silicon standard workflow)
            zip_filename = Path(asset.url).name
            target_dir.mkdir(parents=True, exist_ok=True)
            zip_path = target_dir.parent / zip_filename
            
            try:
                # wget to models directory
                console.print(f"[dim]Running: wget -O {zip_path} {asset.url}[/dim]")
                subprocess.run(
                    ["wget", "-O", str(zip_path), asset.url],
                    check=True,
                    cwd=str(self.cache_dir),
                    capture_output=True
                )
                console.print(f"[green]✓ Downloaded {zip_filename}[/green]")
                
                # unzip -qq (quiet mode) to a temp location
                temp_extract = target_dir.parent / f"{target_dir.name}_temp"
                console.print(f"[dim]Running: unzip -qq {zip_path} -d {temp_extract}[/dim]")
                subprocess.run(
                    ["unzip", "-qq", str(zip_path), "-d", str(temp_extract)],
                    check=True
                )
                console.print(f"[green]✓ Extracted archive[/green]")
                
                # Move files from subdirectory to target (Apple zip has nested dir)
                import shutil
                extracted_subdir = temp_extract / "llava-fastvithd_0.5b_stage3_llm.fp16"
                if extracted_subdir.exists():
                    # Rename the subdirectory to the target name
                    extracted_subdir.rename(target_dir)
                    # Clean up temp directory and __MACOSX
                    shutil.rmtree(temp_extract, ignore_errors=True)
                    console.print(f"[green]✓ Moved to {target_dir}[/green]")
                else:
                    # Fallback: rename temp to target
                    temp_extract.rename(target_dir)
                    console.print(f"[green]✓ Extracted to {target_dir}[/green]")
                
                # rm zip file
                zip_path.unlink()
                console.print(f"[green]✓ Cleaned up {zip_filename}[/green]")
                
                # Convert to MLX format using mlx-vlm convert
                console.print(f"[yellow]Converting model to MLX format...[/yellow]")
                console.print(f"[dim]This may take a few minutes...[/dim]")
                mlx_converted_dir = target_dir.parent / f"{target_dir.name}_mlx"
                try:
                    subprocess.run(
                        [
                            "python", "-m", "mlx_vlm.convert",
                            "--hf-path", str(target_dir),
                            "--mlx-path", str(mlx_converted_dir),
                            "--only-llm"
                        ],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    # Move converted files back to target_dir
                    import shutil
                    shutil.rmtree(target_dir)
                    mlx_converted_dir.rename(target_dir)
                    console.print(f"[green]✓ Converted to MLX format[/green]")
                except subprocess.CalledProcessError as conv_exc:
                    console.print(f"[yellow]Warning: MLX conversion failed: {conv_exc}[/yellow]")
                    console.print(f"[yellow]Model will use PyTorch format (slower)[/yellow]")
                    if mlx_converted_dir.exists():
                        import shutil
                        shutil.rmtree(mlx_converted_dir, ignore_errors=True)
                
                console.print(f"[bold green]✓ MLX fp16 model ready at {target_dir}[/bold green]")
                
            except subprocess.CalledProcessError as exc:
                console.print(f"[red]Failed to download/extract MLX model: {exc}[/red]")
                if zip_path.exists():
                    zip_path.unlink()
                raise RuntimeError(f"MLX model download failed: {exc}")
            except FileNotFoundError as exc:
                console.print(f"[red]wget or unzip not found. Please install: brew install wget[/red]")
                raise RuntimeError("Missing required tools for MLX download") from exc
            
            return target_dir
        elif asset.huggingface_repo:
            console.print(
                f"[cyan]Syncing {asset.name} from Hugging Face ({asset.huggingface_repo})...[/cyan]"
            )
            snapshot_download(
                repo_id=asset.huggingface_repo,
                revision=asset.huggingface_revision,
                local_dir=str(target_dir),
                allow_patterns=(
                    list(asset.allow_patterns) if asset.allow_patterns else None
                ),
            )
        elif asset.url:
            destination = target_dir / (asset.primary_file or Path(asset.url).name)
            self._download_url(asset.url, destination)
        else:
            console.print(
                f"[yellow]Asset '{asset.name}' has no download source configured."
                " Place the required files manually if needed."
            )

        if asset.primary_file and asset.sha256:
            primary_path = target_dir / asset.primary_file
            if not self._verify_checksum(primary_path, asset.sha256):
                raise RuntimeError(
                    f"Checksum validation failed for '{asset.name}' after download."
                )

        return target_dir

    def _download_url(self, url: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Progress() as progress:
            task_id = progress.add_task(
                f"download {destination.name}", start=False, total=None
            )
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            progress.start_task(task_id)
            with destination.open("wb") as file_handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        file_handle.write(chunk)
                        progress.update(task_id, advance=len(chunk))
            progress.stop_task(task_id)

    @staticmethod
    def _verify_checksum(path: Path, expected_sha256: str) -> bool:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest().lower() == expected_sha256.lower()

    def describe_runtime_assets(self, runtime: str) -> Iterable[str]:
        for asset in self.list_runtime_assets(runtime):
            yield f"{asset.name} -> {self.asset_path(asset)}"
