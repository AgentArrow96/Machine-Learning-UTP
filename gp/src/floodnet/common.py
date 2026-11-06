"""Project paths, atomic records, and content identities (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["Other", "Flooded building", "Flooded road"]
COLOURS = ["#94a3b8", "#f59e0b", "#0891b2"]
SPLITS = ("train", "val", "test")


class ProjectError(RuntimeError):
    pass


def inside(path: str | Path) -> Path:
    path = Path(path)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(ROOT):
        raise ProjectError(f"Project output/input must stay inside {ROOT}: {path}")
    return path


def environment() -> None:
    for key, relative in {"TORCH_HOME": ".cache/torch", "HF_HOME": ".cache/huggingface",
                          "PIP_CACHE_DIR": ".cache/pip", "TMP": ".tmp", "TEMP": ".tmp"}.items():
        path = inside(relative)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: str | Path):
    with inside(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value) -> None:
    path = inside(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False, sort_keys=True)
        stream.write("\n")
    replace_file(temp, path)


def replace_file(source, target):
    source, target = inside(source), inside(target)
    # Windows indexers/antivirus can briefly hold a destination without delete sharing.
    for attempt in range(8):
        try:
            source.replace(target)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.1 * 2**attempt, 2))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with inside(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def config(path: str | Path | None = None) -> dict:
    cfg = read_json(path or "configs/base.json")
    if cfg["ignore_index"] != 255 or cfg["size"] != 256:
        raise ProjectError("The frozen comparison requires size=256 and ignore_index=255.")
    if not cfg["train_seeds"] or len(set(cfg["train_seeds"])) != len(cfg["train_seeds"]):
        raise ProjectError("train_seeds must be a non-empty list of distinct seeds.")
    return cfg


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def relative(path: str | Path) -> str:
    return inside(path).relative_to(ROOT).as_posix()
