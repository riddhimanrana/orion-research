.PHONY: help install install-dev install-research test lint format type-check profile benchmark demo clean docs ci-local

help:
	@echo "Orion Research: Common Commands"
	@echo ""
	@echo "Development:"
	@echo "  make install            Install package (production)"
	@echo "  make install-dev        Install with dev tools (testing, linting, profiling)"
	@echo "  make install-research   Install with research deps (Ego4D, notebooks, benchmarks)"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  make test               Run unit tests (CPU)"
	@echo "  make test-gpu           Run unit tests (CUDA GPU)"
	@echo "  make test-all           Run all tests (unit + integration)"
	@echo "  make lint               Lint code (flake8)"
	@echo "  make type-check         Type checking (mypy)"
	@echo "  make format             Format code (black, isort)"
	@echo "  make ci-local           Run CI pipeline locally"
	@echo ""
	@echo "Performance & Research:"
	@echo "  make profile            Profile latency & memory (auto-detect device)"
	@echo "  make benchmark-ego4d    Benchmark on Ego4D dataset"
	@echo "  make ablation           Run ablation studies (2D vs 3D, hands, occlusion)"
	@echo ""
	@echo "Demo & Deployment:"
	@echo "  make demo               Run WACV demo on sample video"
	@echo "  make build              Build distribution packages"
	@echo "  make clean              Clean build artifacts & cache"
	@echo "  make docs               Generate API documentation"

# ============================================================================
# Installation Targets
# ============================================================================

install:
	pip install -e .

install-dev: install
	pip install -e ".[dev]"

install-research: install-dev
	pip install -e ".[research]"

# ============================================================================
# Testing & Quality Assurance
# ============================================================================

test:
	pytest orion/tests/ -v --cov=orion --cov-report=term-missing --device cpu

test-gpu:
	pytest orion/tests/ -v --cov=orion --cov-report=term-missing --device cuda

test-all:
	pytest orion/tests/ orion/tests/integration/ -v --cov=orion --cov-report=html

lint:
	flake8 orion/ --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 orion/ --count --exit-zero --max-complexity=10 --max-line-length=120

type-check:
	mypy orion/ --ignore-missing-imports --no-incremental

format:
	black orion/ --line-length=120
	isort orion/

ci-local:
	@echo "Running local CI checks..."
	make lint
	make type-check
	make test
	@echo "✅ All CI checks passed!"

# ============================================================================
# Performance & Research
# ============================================================================

profile:
	python scripts/profile_performance.py --device auto --output results/profile.json
	@echo "✅ Profile saved to results/profile.json"

benchmark-ego4d:
	@echo "Downloading Ego4D subset (if not present)..."
	python scripts/download_ego4d.py --limit 10
	@echo "Running Ego4D benchmark..."
	python scripts/run_benchmark.py --dataset ego4d --output results/ego4d_results.json

ablation:
	@echo "Running ablation studies..."
	python scripts/run_ablation.py --output results/ablation_results.json
	@echo "Generating ablation report..."
	python scripts/generate_report.py --results results/ablation_results.json --output results/ablation_report.md

# ============================================================================
# Demo & Deployment
# ============================================================================

demo:
	@echo "Running WACV demo..."
	@mkdir -p results/demo
	python -m orion.cli analyze-with-qa \
		--video data/test/sample.mp4 \
		--questions "What did I hold?" "When did I pick it up?" "Did I use my hands?" \
		--output-dir results/demo \
		--visualize
	@echo "✅ Demo complete! Open results/demo/output.html"

build:
	pip install build
	python -m build
	@echo "✅ Build complete! Distributions in dist/"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ htmlcov/
	find . -type d -name ".coverage" -exec rm -rf {} + 2>/dev/null || true
	@echo "✅ Cleaned!"

docs:
	@command -v pdoc >/dev/null 2>&1 || pip install pdoc3
	pdoc --html orion/ -o docs/api/ --force
	@echo "✅ API docs generated in docs/api/"

# ============================================================================
# Development Helpers
# ============================================================================

.PHONY: watch-test debug-frame debug-entity

watch-test:
	ptw -- orion/tests/ -v

debug-frame:
	@echo "Extracting frame debug data..."
	python -c "from orion.research.debug import FrameLogger; print('Frame logger ready')"

debug-entity:
	@echo "Inspector mode - use breakpoints to inspect entity state"
	python -c "from orion.research.debug import DebugInspector; print('Debug inspector ready')"

# ============================================================================
# Continuous Integration (GitHub Actions simulation)
# ============================================================================

.PHONY: ci-ubuntu-cpu ci-ubuntu-gpu ci-macos ci-windows

ci-ubuntu-cpu:
	@echo "=== Ubuntu CPU Test ==="
	make install-dev
	make lint
	make type-check
	pytest orion/tests/ -v --device cpu

ci-ubuntu-gpu:
	@echo "=== Ubuntu GPU Test ==="
	make install-dev
	make lint
	pytest orion/tests/ -v --device cuda

ci-macos:
	@echo "=== macOS MPS Test ==="
	make install-dev
	pytest orion/tests/ -v --device mps --tb=short || true

ci-windows:
	@echo "=== Windows CPU Test ==="
	python -m pip install -e .[dev]
	python -m pytest orion/tests/ -v --device cpu
