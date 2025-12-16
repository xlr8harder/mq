from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .errors import ConfigError, UserError

CONFIG_VERSION = 1
CONVERSATION_VERSION = 1


def mq_home() -> Path:
    override = os.getenv("MQ_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".mq"


def ensure_home() -> Path:
    home = mq_home()
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    return home


def config_path() -> Path:
    return ensure_home() / "config.json"


def last_conversation_path() -> Path:
    return ensure_home() / "last_conversation.json"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {path}: {e}") from e


def _write_json_atomic(path: Path, data: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=path.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
        json.dump(data, tmp, indent=2, ensure_ascii=False)
        tmp.write("\n")
        tmp.flush()
        os.fsync(tmp.fileno())
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def load_config() -> dict[str, Any]:
    path = config_path()
    try:
        data = _read_json(path)
    except FileNotFoundError:
        return {"version": CONFIG_VERSION, "models": {}}
    if not isinstance(data, dict):
        raise ConfigError(f"Invalid config format in {path} (expected object)")
    version = data.get("version", CONFIG_VERSION)
    models = data.get("models", {})
    if not isinstance(version, int):
        raise ConfigError(f"Invalid config version in {path}")
    if not isinstance(models, dict):
        raise ConfigError(f"Invalid models map in {path}")
    return {"version": version, "models": models}


def save_config(config: dict[str, Any]) -> None:
    _write_json_atomic(config_path(), config)


def upsert_model(shortname: str, provider: str, model: str, sysprompt: str | None) -> None:
    if not shortname.strip():
        raise UserError("Shortname must be non-empty")
    if not provider.strip():
        raise UserError("Provider must be non-empty")
    if not model.strip():
        raise UserError("Model must be non-empty")

    config = load_config()
    models = dict(config.get("models", {}))
    models[shortname] = {"provider": provider, "model": model, "sysprompt": sysprompt}
    config["version"] = CONFIG_VERSION
    config["models"] = models
    save_config(config)


def get_model(shortname: str) -> dict[str, Any]:
    config = load_config()
    models = config.get("models") or {}
    if shortname not in models:
        raise UserError(f"Unknown model shortname: {shortname!r}")
    entry = models[shortname]
    if not isinstance(entry, dict):
        raise ConfigError(f"Invalid model entry for {shortname!r}")
    provider = entry.get("provider")
    model = entry.get("model")
    sysprompt = entry.get("sysprompt")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ConfigError(f"Invalid model entry for {shortname!r} (missing provider/model)")
    if sysprompt is not None and not isinstance(sysprompt, str):
        raise ConfigError(f"Invalid sysprompt for {shortname!r} (expected string)")
    return {"provider": provider, "model": model, "sysprompt": sysprompt}


def list_models() -> list[tuple[str, dict[str, Any]]]:
    config = load_config()
    models = config.get("models") or {}
    if not isinstance(models, dict):
        raise ConfigError("Invalid config models map")
    items: list[tuple[str, dict[str, Any]]] = []
    for shortname in sorted(models.keys()):
        entry = get_model(shortname)
        items.append((shortname, entry))
    return items


def remove_model(shortname: str) -> None:
    config = load_config()
    models = dict(config.get("models", {}))
    if shortname not in models:
        raise UserError(f"Unknown model shortname: {shortname!r}")
    models.pop(shortname, None)
    config["version"] = CONFIG_VERSION
    config["models"] = models
    save_config(config)


def load_last_conversation() -> dict[str, Any]:
    path = last_conversation_path()
    try:
        data = _read_json(path)
    except FileNotFoundError as e:
        raise UserError("No previous conversation found") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Invalid conversation format in {path}")
    return data


def save_last_conversation(conversation: dict[str, Any]) -> None:
    _write_json_atomic(last_conversation_path(), conversation)
