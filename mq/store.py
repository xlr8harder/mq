from __future__ import annotations

import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import ConfigError, UserError

CONFIG_VERSION = 1
SESSION_VERSION = 1
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")

_CONFIG_PATH_OVERRIDE: Path | None = None


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
    if _CONFIG_PATH_OVERRIDE is not None:
        return _CONFIG_PATH_OVERRIDE
    return ensure_home() / "config.json"


def set_config_path_override(path: Path | None) -> None:
    global _CONFIG_PATH_OVERRIDE
    _CONFIG_PATH_OVERRIDE = path


def last_conversation_path() -> Path:
    return ensure_home() / "last_conversation.json"


def sessions_dir() -> Path:
    path = ensure_home() / "sessions"
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    return path


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _session_filename(session_id: str) -> str:
    return f"{session_id}.json"


def session_path(session_id: str) -> Path:
    return sessions_dir() / _session_filename(session_id)

def validate_session_id(session_id: str) -> None:
    if not isinstance(session_id, str) or not session_id:
        raise UserError("Session id must be non-empty")
    if not SESSION_ID_RE.fullmatch(session_id):
        raise UserError("Invalid session id (use only letters, digits, '_' and '-', no spaces)")


def session_exists(session_id: str) -> bool:
    try:
        validate_session_id(session_id)
    except UserError:
        return False
    return session_path(session_id).exists()


def _set_latest_session(session_id: str) -> None:
    target_name = _session_filename(session_id)
    last = last_conversation_path()
    try:
        if last.exists() or last.is_symlink():
            last.unlink()
        os.symlink(str(Path("sessions") / target_name), last)
    except OSError:
        # Fallback if symlinks are not available.
        try:
            last.write_text(session_id + "\n", encoding="utf-8")
        except OSError:
            pass


def _read_latest_session_id_from_last_conversation() -> str | None:
    last = last_conversation_path()
    if last.is_symlink():
        try:
            target = os.readlink(last)
        except OSError:
            target = ""
        name = Path(target).name
        if name.endswith(".json"):
            return name[: -len(".json")]
    if last.exists():
        try:
            text = last.read_text(encoding="utf-8").strip()
        except OSError:
            text = ""
        if text:
            return text
    return None


def create_session(
    *,
    model_shortname: str,
    provider: str,
    model: str,
    sysprompt: str | None,
    messages: list[dict],
    session_id: str | None = None,
) -> str:
    if session_id is None:
        session_id = uuid.uuid4().hex
    validate_session_id(session_id)
    if session_path(session_id).exists():
        raise UserError(f"Session already exists: {session_id!r}")
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
    validate_session_id(session_id)
    session["updated_at"] = _now_iso()
    _write_json_atomic(session_path(session_id), session)
    _set_latest_session(session_id)


def load_latest_session() -> dict[str, Any]:
    # Fast path: ~/.mq/last_conversation.json is maintained as a symlink/pointer
    # to the latest session file. Reading it avoids any sessions/ directory scan.
    last = last_conversation_path()
    if last.exists() or last.is_symlink():
        try:
            data = _read_json(last)
            if isinstance(data, dict) and isinstance(data.get("id"), str) and data.get("id"):
                return data
        except Exception:
            pass

    session_id = _read_latest_session_id_from_last_conversation()
    if session_id:
        try:
            return load_session(session_id)
        except UserError:
            pass

    # Last resort: pick the newest session file.
    candidates = sorted(
        (p for p in sessions_dir().glob("*.json")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise UserError("No previous conversation found")
    return load_session(candidates[0].stem)


def list_sessions() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in sessions_dir().glob("*.json"):
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


def rename_session(old_id: str, new_id: str) -> None:
    validate_session_id(old_id)
    validate_session_id(new_id)
    if old_id == new_id:
        return
    old_path = session_path(old_id)
    if not old_path.exists():
        raise UserError(f"Unknown session id: {old_id!r}")
    new_path = session_path(new_id)
    if new_path.exists():
        raise UserError(f"Session already exists: {new_id!r}")

    session = load_session(old_id)
    session["id"] = new_id
    session["updated_at"] = _now_iso()
    _write_json_atomic(new_path, session)
    try:
        old_path.unlink()
    except OSError:
        pass

    current = _read_latest_session_id_from_last_conversation()
    if current == old_id:
        _set_latest_session(new_id)

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


def upsert_model(
    shortname: str,
    provider: str,
    model: str,
    sysprompt: str | None,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
) -> None:
    if not shortname.strip():
        raise UserError("Shortname must be non-empty")
    if not provider.strip():
        raise UserError("Provider must be non-empty")
    if not model.strip():
        raise UserError("Model must be non-empty")

    config = load_config()
    models = dict(config.get("models", {}))
    entry: dict[str, Any] = {"provider": provider, "model": model, "sysprompt": sysprompt}
    if temperature is not None:
        entry["temperature"] = float(temperature)
    if top_p is not None:
        entry["top_p"] = float(top_p)
    if top_k is not None:
        entry["top_k"] = int(top_k)
    models[shortname] = entry
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
    temperature = entry.get("temperature")
    top_p = entry.get("top_p")
    top_k = entry.get("top_k")
    if not isinstance(provider, str) or not isinstance(model, str):
        raise ConfigError(f"Invalid model entry for {shortname!r} (missing provider/model)")
    if sysprompt is not None and not isinstance(sysprompt, str):
        raise ConfigError(f"Invalid sysprompt for {shortname!r} (expected string)")

    if temperature is not None and not isinstance(temperature, (int, float)):
        raise ConfigError(f"Invalid temperature for {shortname!r} (expected number)")
    if top_p is not None and not isinstance(top_p, (int, float)):
        raise ConfigError(f"Invalid top_p for {shortname!r} (expected number)")
    if top_k is not None and not isinstance(top_k, int):
        raise ConfigError(f"Invalid top_k for {shortname!r} (expected int)")

    return {
        "provider": provider,
        "model": model,
        "sysprompt": sysprompt,
        "temperature": float(temperature) if temperature is not None else None,
        "top_p": float(top_p) if top_p is not None else None,
        "top_k": int(top_k) if top_k is not None else None,
    }


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
    # Back-compat: historically this was a real JSON file. Now we keep it as a
    # symlink/pointer to the latest session for convenience.
    path = last_conversation_path()
    if path.exists() or path.is_symlink():
        try:
            data = _read_json(path)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return load_latest_session()


def save_last_conversation(conversation: dict[str, Any]) -> None:
    _write_json_atomic(last_conversation_path(), conversation)
