from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def stable_hash_int(value: str, modulo: int | None = None) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    number = int(digest[:16], 16)
    return number if modulo is None else number % modulo


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, (list, tuple)):
        return " ".join(normalize_text(item) for item in text)
    if isinstance(text, dict):
        return " ".join(f"{key}: {normalize_text(value)}" for key, value in text.items())
    return str(text).strip()


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if not tokens and text:
        tokens = list(text.lower())
    return tokens


def hash_text_vector(text: str, dim: int = 128) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    tokens = tokenize(text)
    for token in tokens:
        bucket = stable_hash_int(token, dim)
        sign = 1.0 if stable_hash_int(token + "::sign", 2) == 0 else -1.0
        vector[bucket] += sign
    for left, right in zip(tokens, tokens[1:]):
        token = f"{left}_{right}"
        bucket = stable_hash_int(token, dim)
        vector[bucket] += 0.5
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)


def clip01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if isinstance(value, tuple):
        return [normalize_text(item) for item in value if normalize_text(item)]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                return as_list(parsed)
            except json.JSONDecodeError:
                pass
        parts = re.split(r"[,，;；/、]", stripped)
        return [part.strip() for part in parts if part.strip()]
    return [normalize_text(value)]


def resolve_path(path: str | Path, base_dir: str | Path | None = None) -> Path:
    candidate = Path(os.path.expandvars(str(path))).expanduser()
    if candidate.is_absolute():
        return candidate
    base = Path(base_dir or os.getcwd())
    return (base / candidate).resolve()
