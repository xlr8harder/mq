from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import ConfigError, UserError

CONFIG_VERSION = 1
SESSION_VERSION = 1


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


def sessions_dir() -> Path:
    path = ensure_home() / "sessions"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def latest_session_link_path() -> Path:
    return sessions_dir() / "latest"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _session_filename(session_id: str) -> str:
    return f"{session_id}.json"


def session_path(session_id: str) -> Path:
    return sessions_dir() / _session_filename(session_id)


def _set_latest_session(session_id: str) -> None:
    link = latest_session_link_path()
    target_name = _session_filename(session_id)
    try:
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(target_name, link)
    except OSError:
        # Fallback if symlinks are not available.
        link.write_text(session_id + "\n", encoding="utf-8")


def _resolve_latest_session_id() -> str:
    link = latest_session_link_path()
    if link.is_symlink():
        try:
            target = os.readlink(link)
        except OSError:
            target = ""
        name = Path(target).name
        if name.endswith(".json"):
            return name[: -len(".json")]
    if link.exists():
        try:
            session_id = link.read_text(encoding="utf-8").strip()
            if session_id:
                return session_id
        except OSError:
            pass

    # No pointer: pick the newest session file.
    candidates = sorted(
        (p for p in sessions_dir().glob("*.json") if p.name != "latest.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise UserError("No previous conversation found")
    return candidates[0].stem


def create_session(*, model_shortname: str, provider: str, model: str, sysprompt: str | None, messages: list[dict]) -> str:
    session_id = uuid.uuid4().hex
    created_at = _now_iso()
    data = {
        "version": SESSION_VERSION,
        "id": session_id,
        "created_at": created_at,
        "updated_at": created_at,
        "model_shortname": model_shortname,
        "provider": provider,
        "model": model,
        "sysprompt": sysprompt,
        "messages": messages,
    }
    _write_json_atomic(session_path(session_id), data)
    _set_latest_session(session_id)
    return session_id


def load_session(session_id: str) -> dict[str, Any]:
    path = session_path(session_id)
    try:
        data = _read_json(path)
    except FileNotFoundError as e:
        raise UserError(f"Unknown session id: {session_id!r}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"Invalid session format in {path}")
    return data


def save_session(session: dict[str, Any]) -> None:
    session_id = session.get("id")
    if not isinstance(session_id, str) or not session_id:
        raise ConfigError("Invalid session (missing id)")
    session["updated_at"] = _now_iso()
    _write_json_atomic(session_path(session_id), session)
    _set_latest_session(session_id)


def load_latest_session() -> dict[str, Any]:
    return load_session(_resolve_latest_session_id())


def list_sessions() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sessions_dir().glob("*.json"):
        if path.name == "latest.json":
            continue
        try:
            data = _read_json(path)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        session_id = data.get("id") or path.stem
        if not isinstance(session_id, str):
            continue
        data = dict(data)
        data["id"] = session_id
        items.append(data)

    def sort_key(d: dict[str, Any]) -> str:
        updated = d.get("updated_at")
        created = d.get("created_at")
        return str(updated or created or "")

    return sorted(items, key=sort_key, reverse=True)


def select_session(session_id: str) -> None:
    # Validate it exists first
    _ = load_session(session_id)
    _set_latest_session(session_id)


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
