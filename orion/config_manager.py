"""
Centralized Configuration Manager for Orion
============================================

Manages loading, validation, and secure access to all Orion configurations
from environment variables and persisted config files.

Features:
- Unified config loading from environment and config files
- Secure credential handling via environment variables
- Validation and sensible defaults
- Support for overrides via CLI args or environment

Author: Orion Research Team
Date: October 2025
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .config import OrionConfig, Neo4jConfig, OllamaConfig, RuntimeConfig

logger = logging.getLogger(__name__)


class ConfigManager:
    """Unified configuration manager for Orion."""

    _instance: Optional["ConfigManager"] = None
    _config: Optional[OrionConfig] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_config(cls) -> OrionConfig:
        """Get the current Orion configuration."""
        if cls._config is None:
            cls._config = cls._load_from_environment()
        return cls._config

    @classmethod
    def set_config(cls, config: OrionConfig) -> None:
        """Set the Orion configuration."""
        cls._config = config

    @classmethod
    def _load_from_environment(cls) -> OrionConfig:
        """
        Load configuration from environment variables and config file.

        Environment variable precedence:
        - ORION_NEO4J_URI
        - ORION_NEO4J_USER
        - ORION_NEO4J_PASSWORD (via password_env_var)
        - ORION_OLLAMA_URL
        - ORION_OLLAMA_QA_MODEL
        - ORION_RUNTIME_BACKEND
        - ORION_CONFIG_PATH (for persisted config file)
        """
        config = OrionConfig()

        # Load Neo4j config from environment
        if uri := os.getenv("ORION_NEO4J_URI"):
            config.neo4j.uri = uri
        if user := os.getenv("ORION_NEO4J_USER"):
            config.neo4j.user = user

        # Set password environment variable name (for secure access)
        if pwd_var := os.getenv("ORION_NEO4J_PASSWORD_VAR"):
            config.neo4j.password_env_var = pwd_var

        # Load Ollama config from environment
        if url := os.getenv("ORION_OLLAMA_URL"):
            config.ollama.base_url = url
        if qa_model := os.getenv("ORION_OLLAMA_QA_MODEL"):
            config.ollama.qa_model = qa_model

        # Load runtime config from environment
        if backend := os.getenv("ORION_RUNTIME_BACKEND"):
            config.runtime.backend = backend  # type: ignore
        if device := os.getenv("ORION_DEVICE"):
            config.runtime.device = device  # type: ignore

        return config

    @classmethod
    def save_config(cls, config: Optional[OrionConfig] = None) -> Path:
        """
        Save configuration to disk.

        Note: Passwords and secrets are NOT saved. They must be set
        via environment variables for security.
        """
        if config is None:
            config = cls.get_config()

        config_path = cls.get_config_path()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Build dict, excluding password fields
        config_dict = {
            "neo4j": {
                "uri": config.neo4j.uri,
                "user": config.neo4j.user,
                "password_env_var": config.neo4j.password_env_var,
            },
            "ollama": {
                "base_url": config.ollama.base_url,
                "qa_model": config.ollama.qa_model,
                "embedding_model": config.ollama.embedding_model,
            },
            "runtime": {
                "backend": config.runtime.backend,
                "device": config.runtime.device,
            },
        }

        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)

        logger.info(f"Configuration saved to {config_path}")
        return config_path

    @classmethod
    def load_config_file(cls) -> Optional[OrionConfig]:
        """
        Load configuration from persisted config file.

        Returns None if file doesn't exist. Raises ValueError if file is invalid.
        """
        config_path = cls.get_config_path()

        if not config_path.exists():
            return None

        try:
            with open(config_path) as f:
                data = json.load(f)

            config = OrionConfig()

            # Load Neo4j config
            if neo4j_data := data.get("neo4j"):
                if "uri" in neo4j_data:
                    config.neo4j.uri = neo4j_data["uri"]
                if "user" in neo4j_data:
                    config.neo4j.user = neo4j_data["user"]
                if "password_env_var" in neo4j_data:
                    config.neo4j.password_env_var = neo4j_data["password_env_var"]

            # Load Ollama config
            if ollama_data := data.get("ollama"):
                if "base_url" in ollama_data:
                    config.ollama.base_url = ollama_data["base_url"]
                if "qa_model" in ollama_data:
                    config.ollama.qa_model = ollama_data["qa_model"]
                if "embedding_model" in ollama_data:
                    config.ollama.embedding_model = ollama_data["embedding_model"]

            # Load runtime config
            if runtime_data := data.get("runtime"):
                if "backend" in runtime_data:
                    config.runtime.backend = runtime_data["backend"]  # type: ignore
                if "device" in runtime_data:
                    config.runtime.device = runtime_data["device"]  # type: ignore

            logger.info(f"Configuration loaded from {config_path}")
            return config

        except (json.JSONDecodeError, KeyError) as e:
            raise ValueError(f"Invalid config file {config_path}: {e}")

    @classmethod
    def get_config_path(cls) -> Path:
        """Get the configuration file path."""
        env_path = os.getenv("ORION_CONFIG_PATH")
        if env_path:
            return Path(env_path).expanduser()

        config_dir = Path(
            os.getenv("ORION_CONFIG_DIR", Path.home() / ".orion")
        ).expanduser()
        return config_dir / "config.json"

    @classmethod
    def validate_neo4j_connection(cls, config: Optional[OrionConfig] = None) -> bool:
        """
        Validate Neo4j connection without connecting.

        Returns True if credentials are set and parseable, False otherwise.
        """
        if config is None:
            config = cls.get_config()

        # Check URI format
        if not config.neo4j.uri:
            logger.warning("Neo4j URI not configured")
            return False

        # Check credentials exist
        if not config.neo4j.user:
            logger.warning("Neo4j user not configured")
            return False

        # Check password environment variable is set
        try:
            _ = config.neo4j.password
        except ValueError as e:
            logger.warning(f"Neo4j password not set: {e}")
            return False

        return True

    @classmethod
    def validate_ollama_connection(cls, config: Optional[OrionConfig] = None) -> bool:
        """
        Validate Ollama connection settings are configured.

        Returns True if settings look valid, False otherwise.
        """
        if config is None:
            config = cls.get_config()

        if not config.ollama.base_url:
            logger.warning("Ollama base URL not configured")
            return False

        if not config.ollama.qa_model:
            logger.warning("Ollama Q&A model not configured")
            return False

        return True
