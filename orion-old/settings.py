"""User-facing configuration management for Orion."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, ClassVar, Dict, Iterable, Tuple

CURRENT_VERSION = 1


class SettingsError(RuntimeError):
    """Raised when configuration operations fail."""


@dataclass
class OrionSettings:
    """Persisted Orion configuration loaded from ``config.json``."""

    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""  # Must be set via ORION_NEO4J_PASSWORD env var
    runtime_backend: str = "auto"
    qa_model: str = "gemma3:4b"
    embedding_backend: str = "auto"
    embedding_model: str = "openai/clip-vit-base-patch32"
    config_version: int = CURRENT_VERSION

    _KEY_ALIASES: ClassVar[Dict[str, str]] = {
        "neo4j.uri": "neo4j_uri",
        "neo4j.user": "neo4j_user",
        "neo4j.password": "neo4j_password",
        "runtime.backend": "runtime_backend",
        "runtime": "runtime_backend",
        "qa.model": "qa_model",
        "embedding.backend": "embedding_backend",
        "embedding.model": "embedding_model",
    }

    _SECRET_KEYS: ClassVar[Tuple[str, ...]] = ("neo4j.password",)
    _VALID_RUNTIMES: ClassVar[Tuple[str, ...]] = ("auto", "torch")
    _VALID_EMBEDDING_BACKENDS: ClassVar[Tuple[str, ...]] = (
        "auto",
        "ollama",
        "sentence-transformer",
    )

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    @classmethod
    def config_path(cls) -> Path:
        """Return the location of the user configuration file."""
        env_path = os.getenv("ORION_CONFIG_PATH")
        if env_path:
            path = Path(env_path).expanduser()
        else:
            base_dir = Path(
                os.getenv("ORION_CONFIG_DIR", Path.home() / ".orion")
            ).expanduser()
            path = base_dir / "config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def load(cls) -> "OrionSettings":
        """Load settings from disk, creating defaults if necessary."""
        path = cls.config_path()
        if not path.exists():
            settings = cls()
            settings.save(path)
            return settings

        try:
            raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (
            json.JSONDecodeError
        ) as exc:  # pragma: no cover - invalid config edge case
            raise SettingsError(
                f"Configuration file at {path} is not valid JSON: {exc}"
            ) from exc

        known_fields = {field.name for field in fields(cls)}
        data: Dict[str, Any] = {}
        for name in known_fields:
            if name in raw:
                data[name] = raw[name]

        settings = cls(**data)
        settings.config_version = CURRENT_VERSION
        try:
            settings.validate()
        except SettingsError:
            # If validation fails, do not persist; just surface the error.
            raise

        # Ensure new defaults are written back so the file stays current.
        settings.save(path)
        return settings

    def save(self, path: Path | None = None) -> None:
        """Persist the current settings to disk."""
        self.validate()
        target = path or self.config_path()
        payload = self.to_dict()
        target.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def iter_display_items(self) -> Iterable[Tuple[str, str]]:
        """Yield human-friendly key/value pairs for CLI display."""
        ordering = (
            "neo4j.uri",
            "neo4j.user",
            "neo4j.password",
            "runtime.backend",
            "qa.model",
            "embedding.backend",
            "embedding.model",
        )
        flat = self._flatten()
        for key in ordering:
            value = flat.get(key)
            if value is None:
                continue
            if key in self._SECRET_KEYS:
                yield key, self._mask_secret(value)
            else:
                yield key, value

    def _flatten(self) -> Dict[str, str]:
        return {
            "neo4j.uri": self.neo4j_uri,
            "neo4j.user": self.neo4j_user,
            "neo4j.password": self.neo4j_password,
            "runtime.backend": self.runtime_backend,
            "qa.model": self.qa_model,
            "embedding.backend": self.embedding_backend,
            "embedding.model": self.embedding_model,
        }

    @staticmethod
    def _mask_secret(value: str) -> str:
        if not value:
            return ""
        if len(value) <= 4:
            return "*" * len(value)
        return f"{value[:2]}…{value[-2:]}"

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def set_value(self, key: str, raw_value: str) -> None:
        """Update a configuration value using CLI-friendly keys."""
        field_name = self._resolve_field_name(key)
        value = raw_value.strip()

        if field_name == "runtime_backend":
            normalized = value.lower()
            if normalized == "mlx":
                normalized = "torch"
            if normalized not in self._VALID_RUNTIMES:
                raise SettingsError(
                    f"Runtime backend must be one of {self._VALID_RUNTIMES}."
                )
            value = normalized
        elif field_name == "embedding_backend":
            normalized = value.lower()
            if normalized not in self._VALID_EMBEDDING_BACKENDS:
                raise SettingsError(
                    f"Embedding backend must be one of {self._VALID_EMBEDDING_BACKENDS}."
                )
            value = normalized
        elif field_name == "neo4j_uri":
            value = value or self.neo4j_uri
        elif field_name in {
            "neo4j_user",
            "neo4j_password",
            "qa_model",
            "embedding_model",
        }:
            if not value:
                raise SettingsError(f"Configuration '{key}' cannot be empty.")
        else:  # pragma: no cover - future-proof branch
            raise SettingsError(f"'{key}' cannot be modified via the CLI.")

        setattr(self, field_name, value)

    def _resolve_field_name(self, key: str) -> str:
        normalized = key.strip().lower().replace("_", ".")
        alias = self._KEY_ALIASES.get(normalized)
        if alias:
            return alias
        normalized = normalized.replace("-", ".")
        alias = self._KEY_ALIASES.get(normalized)
        if alias:
            return alias
        raise SettingsError(f"Unknown configuration key '{key}'.")

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> None:
        if self.runtime_backend not in self._VALID_RUNTIMES:
            raise SettingsError(
                f"Runtime backend must be one of {self._VALID_RUNTIMES}."
            )
        if self.embedding_backend not in self._VALID_EMBEDDING_BACKENDS:
            raise SettingsError(
                f"Embedding backend must be one of {self._VALID_EMBEDDING_BACKENDS}."
            )
        if not self.neo4j_uri:
            raise SettingsError("Neo4j URI cannot be empty.")
        if not self.neo4j_user:
            raise SettingsError("Neo4j user cannot be empty.")

        # Check for password from environment variable or stored value
        password = self._get_neo4j_password()
        if not password:
            raise SettingsError(
                "Neo4j password must be set via ORION_NEO4J_PASSWORD environment variable "
                "or stored in config file."
            )

        if not self.qa_model:
            raise SettingsError("QA model cannot be empty.")
        if not self.embedding_model:
            raise SettingsError("Embedding model cannot be empty.")

    def _get_neo4j_password(self) -> str:
        """Get Neo4j password from environment variable or stored value."""
        # Priority: environment variable > stored value
        env_password = os.getenv("ORION_NEO4J_PASSWORD")
        if env_password:
            return env_password
        return self.neo4j_password

    def get_neo4j_password(self) -> str:
        """Get the actual Neo4j password for use in connections."""
        password = self._get_neo4j_password()
        if not password:
            raise SettingsError(
                "Neo4j password not configured. Set ORION_NEO4J_PASSWORD environment variable."
            )
        return password
