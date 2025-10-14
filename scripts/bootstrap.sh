#!/usr/bin/env bash
# Bootstrap Orion: clone (if needed), create a virtualenv, install, and download models.

set -euo pipefail

REPO_URL=${ORION_REPO_URL:-https://github.com/riddhimanrana/orion-research.git}
REPO_BRANCH=${ORION_REPO_BRANCH:-main}
TARGET_DIR=${ORION_TARGET_DIR:-orion-research}
PYTHON_BIN=${ORION_PYTHON:-python3}
VENV_DIR=${ORION_VENV:-.orion-venv}

log() {
    printf '[orion-bootstrap] %s\n' "$1"
}

if ! command -v git >/dev/null 2>&1; then
    printf 'Error: git is required for the bootstrap script.\n' >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    printf 'Error: %s not found. Set ORION_PYTHON to an installed interpreter.\n' "$PYTHON_BIN" >&2
    exit 1
fi

if [ ! -d "$TARGET_DIR/.git" ]; then
    log "Cloning $REPO_URL (branch $REPO_BRANCH) into $TARGET_DIR"
    git clone --branch "$REPO_BRANCH" --depth 1 "$REPO_URL" "$TARGET_DIR"
else
    log "Repository already present at $TARGET_DIR; pulling latest changes"
    git -C "$TARGET_DIR" fetch --depth 1 origin "$REPO_BRANCH"
    git -C "$TARGET_DIR" checkout "$REPO_BRANCH"
    git -C "$TARGET_DIR" pull --ff-only origin "$REPO_BRANCH"
fi

cd "$TARGET_DIR"

if [ ! -d "$VENV_DIR" ]; then
    log "Creating virtual environment ($VENV_DIR)"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"

log "Upgrading pip"
pip install --upgrade pip >/dev/null

log "Installing Orion (editable)"
pip install -e .

log "Preparing model assets (python -m orion.cli init)"
python -m orion.cli init

ACTIVATE_PATH="$(pwd)/$VENV_DIR/bin/activate"

cat <<EOF

Orion bootstrap complete.

To activate the environment later:
  source "$ACTIVATE_PATH"

To run the CLI manually:
  python -m orion.cli --help
or
  orion --help  (if the entry point is on PATH)

Remember to start Ollama in another terminal:
  ollama serve
EOF
