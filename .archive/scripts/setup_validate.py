#!/usr/bin/env python3
"""
Setup Validation Script for Orion Research
===========================================

Validates local or remote environment for Orion pipeline.
Checks dependencies, model weights, GPU/MPS availability, and Memgraph connectivity.

Usage:
    # Validate local installation
    python scripts/setup_validate.py --local

    # Validate Lambda AI remote (requires SSH)
    python scripts/setup_validate.py --remote --host ubuntu@<LAMBDA-IP>

    # Fix environment automatically
    python scripts/setup_validate.py --auto-fix
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class SetupValidator:
    """Validates Orion environment setup."""

    def __init__(self, remote_host: str = None, auto_fix: bool = False):
        self.remote_host = remote_host
        self.auto_fix = auto_fix
        self.results: Dict[str, Dict] = {}

    def run_cmd(self, cmd: str, check: bool = False) -> Tuple[str, int]:
        """
        Execute command locally or remotely.
        
        Args:
            cmd: Shell command
            check: Raise error if command fails
            
        Returns:
            (stdout, returncode)
        """
        if self.remote_host:
            full_cmd = f"ssh -o ConnectTimeout=5 {self.remote_host} '{cmd}'"
        else:
            full_cmd = cmd

        try:
            result = subprocess.run(
                full_cmd, shell=True, capture_output=True, text=True, timeout=30
            )
            return result.stdout.strip(), result.returncode
        except subprocess.TimeoutExpired:
            if check:
                raise RuntimeError(f"Command timeout: {cmd}")
            return "", 1
        except Exception as e:
            if check:
                raise RuntimeError(f"Command failed: {e}")
            return "", 1

    def check_python(self) -> bool:
        """Check Python version (>=3.10 required)."""
        logger.info("Checking Python version...")
        try:
            version_str, _ = self.run_cmd("python3 --version")
            # Parse "Python 3.X.Y"
            major, minor = map(int, version_str.split()[1].split(".")[:2])
            if (major, minor) >= (3, 10):
                logger.info(f"  ✓ Python {major}.{minor} (>= 3.10)")
                self.results["python"] = {"status": "ok", "version": f"{major}.{minor}"}
                return True
            else:
                logger.warning(f"  ✗ Python {major}.{minor} < 3.10 required")
                self.results["python"] = {"status": "fail", "version": f"{major}.{minor}"}
                return False
        except Exception as e:
            logger.error(f"  ✗ Error checking Python: {e}")
            self.results["python"] = {"status": "error", "message": str(e)}
            return False

    def check_pytorch(self) -> bool:
        """Check PyTorch installation and CUDA/MPS availability."""
        logger.info("Checking PyTorch and device support...")
        try:
            output, code = self.run_cmd(
                "python3 -c \"import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'MPS: {torch.backends.mps.is_available()}')\" 2>&1"
            )
            if code == 0:
                lines = output.split('\n')
                logger.info(f"  ✓ {lines[0]}")
                
                has_cuda = "CUDA: True" in output
                has_mps = "MPS: True" in output
                has_device = has_cuda or has_mps
                
                if has_cuda:
                    logger.info("  ✓ CUDA available (GPU acceleration enabled)")
                elif has_mps:
                    logger.info("  ✓ Apple Silicon MPS available (GPU acceleration enabled)")
                else:
                    logger.warning("  ⚠ No GPU detected (CPU mode, slower)")
                
                self.results["pytorch"] = {
                    "status": "ok",
                    "version": lines[0].split()[-1],
                    "cuda": has_cuda,
                    "mps": has_mps
                }
                return True
            else:
                logger.error(f"  ✗ PyTorch not found or import error:\n{output}")
                self.results["pytorch"] = {"status": "fail", "message": output}
                return False
        except Exception as e:
            logger.error(f"  ✗ Error checking PyTorch: {e}")
            self.results["pytorch"] = {"status": "error", "message": str(e)}
            return False

    def check_orion_imports(self) -> bool:
        """Check critical Orion module imports."""
        logger.info("Checking Orion module imports...")
        
        modules = [
            ("orion.perception.engine", "Perception Engine"),
            ("orion.perception.embedder", "CLIP Embeddings"),
            ("orion.slam", "SLAM/Visual Odometry"),
            ("orion.perception.depth", "Depth Estimation"),
            ("orion.graph.builder", "Scene Graph Builder"),
        ]
        
        all_ok = True
        for module_name, label in modules:
            try:
                output, code = self.run_cmd(f"python3 -c \"import {module_name}; print('OK')\"")
                if code == 0:
                    logger.info(f"  ✓ {label}")
                    if module_name not in self.results:
                        self.results[module_name] = {"status": "ok"}
                else:
                    logger.warning(f"  ⚠ {label} - import issue (may be optional)")
                    self.results[module_name] = {"status": "warn", "message": output}
            except Exception as e:
                logger.warning(f"  ⚠ {label} - error: {e}")
                self.results[module_name] = {"status": "error", "message": str(e)}

        return all_ok

    def check_model_weights(self) -> bool:
        """Check for cached model weights."""
        logger.info("Checking model weights...")
        
        weights_to_check = [
            (MODELS_DIR / "_torch" / "yolo11m.pt", "YOLO11m"),
            (MODELS_DIR / "_torch" / "yolo11s.pt", "YOLO11s"),
            (MODELS_DIR / "dinov3-vitl16", "DINOv3"),
            (MODELS_DIR / "_huggingface", "HuggingFace Cache"),
        ]
        
        all_ok = True
        for path, label in weights_to_check:
            if self.remote_host:
                # Check remote
                output, code = self.run_cmd(f"test -e {path} && echo 'exists' || echo 'missing'")
                exists = "exists" in output
            else:
                exists = path.exists()
            
            status = "✓" if exists else "⚠"
            logger.info(f"  {status} {label}: {path}")
            
            if not exists:
                all_ok = False
        
        self.results["model_weights"] = {
            "status": "ok" if all_ok else "warn",
            "message": "Some weights may need to be downloaded on first run"
        }
        return all_ok

    def check_memgraph(self) -> bool:
        """Check Memgraph connectivity (optional)."""
        logger.info("Checking Memgraph backend (optional)...")
        
        try:
            output, code = self.run_cmd(
                "python3 -c \"from neo4j import GraphDatabase; print('OK')\" 2>&1"
            )
            if code == 0:
                logger.info("  ✓ neo4j client available")
                
                # Try to connect to localhost (optional, may not be running)
                output, code = self.run_cmd(
                    "python3 -c \"from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://localhost:7687'); print('Connected')\" 2>&1"
                )
                if code == 0:
                    logger.info("  ✓ Memgraph running at localhost:7687")
                    self.results["memgraph"] = {"status": "ok", "running": True}
                else:
                    logger.info("  ⓘ Memgraph client available but not running (start with: docker-compose up)")
                    self.results["memgraph"] = {"status": "ok", "running": False}
                return True
            else:
                logger.info("  ⓘ neo4j client not installed (optional, use pip install -e .[memgraph])")
                self.results["memgraph"] = {"status": "optional", "message": "Not installed"}
                return True  # Optional, not a failure
        except Exception as e:
            logger.warning(f"  ⚠ Memgraph check error: {e}")
            self.results["memgraph"] = {"status": "error", "message": str(e)}
            return True  # Don't fail if Memgraph is optional

    def check_git(self) -> bool:
        """Check Git is available."""
        logger.info("Checking Git...")
        
        try:
            output, code = self.run_cmd("git --version")
            if code == 0:
                logger.info(f"  ✓ {output}")
                self.results["git"] = {"status": "ok", "version": output}
                return True
            else:
                logger.warning("  ✗ Git not found")
                self.results["git"] = {"status": "fail"}
                return False
        except Exception as e:
            logger.error(f"  ✗ Git check failed: {e}")
            self.results["git"] = {"status": "error", "message": str(e)}
            return False

    def validate_all(self) -> bool:
        """Run all validation checks."""
        logger.info("=" * 80)
        logger.info(f"ORION SETUP VALIDATION ({self._get_environment()})")
        logger.info("=" * 80)
        
        checks = [
            ("Python Version", self.check_python),
            ("PyTorch & GPU", self.check_pytorch),
            ("Orion Imports", self.check_orion_imports),
            ("Model Weights", self.check_model_weights),
            ("Git", self.check_git),
            ("Memgraph (Optional)", self.check_memgraph),
        ]
        
        results = []
        for name, check_func in checks:
            try:
                result = check_func()
                results.append(result)
            except Exception as e:
                logger.error(f"{name} check failed: {e}")
                results.append(False)
        
        # Summary
        logger.info("=" * 80)
        self._print_summary()
        logger.info("=" * 80)
        
        return all(results)

    def _get_environment(self) -> str:
        """Get environment name."""
        if self.remote_host:
            return f"Remote ({self.remote_host})"
        else:
            return "Local"

    def _print_summary(self) -> bool:
        """Print validation summary."""
        critical = ["python", "pytorch"]
        
        all_critical_ok = all(
            self.results.get(k, {}).get("status") == "ok" for k in critical
        )
        
        if all_critical_ok:
            logger.info("✓ ENVIRONMENT READY")
            logger.info("\nNext steps:")
            if self.remote_host:
                logger.info(f"  1. SSH into instance: ssh -i <key> {self.remote_host}")
                logger.info(f"  2. Clone repo: git clone https://github.com/riddhimanrana/orion-research.git")
                logger.info(f"  3. Install: pip install -e .[all]")
                logger.info(f"  4. Run pipeline: python -m orion.cli.run_showcase --episode test_demo")
            else:
                logger.info("  1. Install optional deps: pip install -e .[all]")
                logger.info("  2. Run test: python -m orion.cli.run_showcase --episode test_demo")
            return True
        else:
            logger.error("✗ ENVIRONMENT HAS ISSUES")
            logger.info("\nFailed checks:")
            for key, result in self.results.items():
                if result.get("status") != "ok":
                    logger.info(f"  - {key}: {result}")
            return False

    def save_report(self, path: str = "results/setup_validation.json"):
        """Save validation report to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nValidation report saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate Orion setup (local or remote)",
        epilog="Examples:\n"
               "  python scripts/setup_validate.py  # Validate local\n"
               "  python scripts/setup_validate.py --remote --host ubuntu@1.2.3.4  # Validate Lambda AI"
    )
    parser.add_argument("--remote", action="store_true", help="Validate remote (Lambda AI) instance")
    parser.add_argument("--host", default=None, help="Remote host SSH address (e.g., ubuntu@1.2.3.4)")
    parser.add_argument("--auto-fix", action="store_true", help="Auto-fix issues (experimental)")
    parser.add_argument("--json", action="store_true", help="Save report to JSON")

    args = parser.parse_args()

    if args.remote and not args.host:
        parser.error("--remote requires --host (e.g., --host ubuntu@1.2.3.4)")

    validator = SetupValidator(remote_host=args.host, auto_fix=args.auto_fix)
    success = validator.validate_all()

    if args.json:
        validator.save_report()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
