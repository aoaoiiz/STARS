from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .utils import sha256_file, sha256_json


CHECKPOINT_MANIFEST_SCHEMA = "stars_checkpoint_manifest_v1"
LOCAL_CHECKPOINT_KIND = "local_checkpoint"

_IGNORED_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    "__pycache__",
}
_IGNORED_FILE_NAMES = {
    ".DS_Store",
}
_IGNORED_FILE_SUFFIXES = {
    ".lock",
    ".tmp",
}


def build_local_checkpoint_manifest(
    model_path: str | Path,
    *,
    model_id: str,
    revision: str,
) -> dict[str, Any]:
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    if not str(model_id).strip():
        raise ValueError("Checkpoint manifest requires a non-empty model_id.")
    if not str(revision).strip():
        raise ValueError(
            "Checkpoint manifest requires an immutable revision or acquisition identifier."
        )

    files = []
    for path in _iter_checkpoint_files(root):
        relative_path = path.relative_to(root).as_posix()
        files.append(
            {
                "path": relative_path,
                "size_bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError(f"No checkpoint files were found under {root}.")

    payload = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "kind": LOCAL_CHECKPOINT_KIND,
        "model_id": str(model_id).strip(),
        "revision": str(revision).strip(),
        "root_name": root.name,
        "file_count": len(files),
        "total_bytes": sum(int(item["size_bytes"]) for item in files),
        "files": files,
    }
    return {**payload, "identity_sha256": sha256_json(payload)}


def write_local_checkpoint_manifest(
    output_path: str | Path,
    manifest: dict[str, Any],
) -> Path:
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    validate_checkpoint_manifest(manifest)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_checkpoint_manifest(
    manifest_path: str | Path,
    *,
    model_path: str | Path | None = None,
    verify_files: bool = False,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest not found: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Checkpoint manifest is not valid JSON: {path}") from exc
    validate_checkpoint_manifest(manifest)
    if verify_files:
        if model_path is None:
            raise ValueError("model_path is required when verify_files=True.")
        verify_checkpoint_files(manifest, model_path)
    return manifest


def validate_checkpoint_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise TypeError("Checkpoint manifest must be a JSON object.")
    if manifest.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA:
        raise ValueError("Checkpoint manifest has an unsupported schema_version.")
    if manifest.get("kind") != LOCAL_CHECKPOINT_KIND:
        raise ValueError("Checkpoint manifest kind must be local_checkpoint.")
    if not str(manifest.get("model_id", "")).strip():
        raise ValueError("Checkpoint manifest model_id is empty.")
    if not str(manifest.get("revision", "")).strip():
        raise ValueError("Checkpoint manifest revision is empty.")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Checkpoint manifest files must be a non-empty array.")

    seen: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("Each checkpoint manifest file entry must be an object.")
        relative_path = str(item.get("path", ""))
        candidate = Path(relative_path)
        if (
            not relative_path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or relative_path in seen
        ):
            raise ValueError(f"Invalid or duplicate checkpoint path: {relative_path!r}")
        seen.add(relative_path)
        size_bytes = item.get("size_bytes")
        digest = str(item.get("sha256", ""))
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"Invalid checkpoint size for {relative_path!r}.")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"Invalid checkpoint SHA256 for {relative_path!r}.")
        total_bytes += size_bytes

    if int(manifest.get("file_count", -1)) != len(files):
        raise ValueError("Checkpoint manifest file_count does not match files.")
    if int(manifest.get("total_bytes", -1)) != total_bytes:
        raise ValueError("Checkpoint manifest total_bytes does not match files.")
    expected_identity = sha256_json(_identity_payload(manifest))
    if str(manifest.get("identity_sha256", "")) != expected_identity:
        raise ValueError("Checkpoint manifest identity_sha256 is invalid.")


def verify_checkpoint_files(
    manifest: dict[str, Any],
    model_path: str | Path,
) -> None:
    validate_checkpoint_manifest(manifest)
    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {root}")
    if root.name != str(manifest.get("root_name", "")):
        raise RuntimeError(
            "Checkpoint root name differs from the sidecar manifest: "
            f"{root.name!r} != {manifest.get('root_name')!r}."
        )
    for item in manifest["files"]:
        path = root / str(item["path"])
        if not path.is_file():
            raise RuntimeError(f"Checkpoint file is missing: {path}")
        actual_size = int(path.stat().st_size)
        if actual_size != int(item["size_bytes"]):
            raise RuntimeError(f"Checkpoint file size differs: {path}")
        if sha256_file(path) != str(item["sha256"]):
            raise RuntimeError(f"Checkpoint file SHA256 differs: {path}")


def checkpoint_identity_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    validate_checkpoint_manifest(manifest)
    return {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA,
        "kind": LOCAL_CHECKPOINT_KIND,
        "model_id": str(manifest["model_id"]),
        "revision": str(manifest["revision"]),
        "identity_sha256": str(manifest["identity_sha256"]),
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
    }


def validate_checkpoint_identity(identity: dict[str, Any]) -> None:
    if not isinstance(identity, dict):
        raise TypeError("checkpoint_identity must be an object.")
    kind = identity.get("kind")
    if kind == LOCAL_CHECKPOINT_KIND:
        required = {
            "schema_version",
            "kind",
            "model_id",
            "revision",
            "identity_sha256",
            "file_count",
            "total_bytes",
        }
        if set(identity) != required:
            raise ValueError("Local checkpoint identity has unexpected or missing fields.")
        if identity.get("schema_version") != CHECKPOINT_MANIFEST_SCHEMA:
            raise ValueError("Local checkpoint identity has the wrong schema_version.")
        if not str(identity.get("model_id", "")).strip() or not str(
            identity.get("revision", "")
        ).strip():
            raise ValueError("Local checkpoint identity model_id/revision is empty.")
        digest = str(identity.get("identity_sha256", ""))
        if len(digest) != 64:
            raise ValueError("Local checkpoint identity digest is invalid.")
        if int(identity.get("file_count", 0)) <= 0 or int(
            identity.get("total_bytes", 0)
        ) <= 0:
            raise ValueError("Local checkpoint identity has no hashed files.")
        return
    raise ValueError(f"Unsupported checkpoint identity kind: {kind!r}")


def _identity_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in manifest.items()
        if key != "identity_sha256"
    }


def _iter_checkpoint_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative_parts = path.relative_to(root).parts
        if any(part in _IGNORED_DIRECTORY_NAMES for part in relative_parts[:-1]):
            continue
        if not path.is_file():
            continue
        if path.name in _IGNORED_FILE_NAMES or path.suffix.lower() in _IGNORED_FILE_SUFFIXES:
            continue
        yield path
