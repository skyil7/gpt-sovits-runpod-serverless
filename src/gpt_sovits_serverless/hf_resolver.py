from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable

from .errors import ConfigurationError


def validate_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        raise ConfigurationError("Path must not be empty")
    if (
        normalized.startswith("/")
        or Path(normalized).is_absolute()
        or PureWindowsPath(normalized).is_absolute()
    ):
        raise ConfigurationError(f"Hugging Face paths must be relative: {path!r}")
    parsed = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in parsed.parts):
        raise ConfigurationError(f"Unsafe relative path: {path!r}")
    return str(parsed)


def safe_join(root: Path, relative_path: str) -> Path:
    relative = validate_relative_path(relative_path)
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not _is_relative_to(candidate, root):
        raise ConfigurationError(f"Path escapes root: {relative_path!r}")
    return candidate


def resolve_asset_path(
    path_value: str,
    *,
    repo_id: str | None,
    revision: str,
    cache_dir: Path,
    token: str | None,
) -> Path:
    local_candidate = Path(path_value).expanduser()
    if local_candidate.is_absolute() or PureWindowsPath(path_value).is_absolute():
        if not local_candidate.exists():
            raise ConfigurationError(f"Local asset path does not exist: {local_candidate}")
        return local_candidate.resolve()

    if local_candidate.exists():
        return local_candidate.resolve()

    if not repo_id:
        raise ConfigurationError(f"No Hugging Face repo configured to resolve {path_value!r}")

    relative = validate_relative_path(path_value)
    snapshot_dir = download_snapshot(
        repo_id=repo_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        allow_patterns=_allow_patterns(relative),
    )
    resolved = safe_join(snapshot_dir, relative)
    if not resolved.exists():
        raise ConfigurationError(
            f"Asset {relative!r} was not found in Hugging Face repo {repo_id}@{revision}"
        )
    return resolved


def download_snapshot(
    *,
    repo_id: str,
    revision: str,
    cache_dir: Path,
    token: str | None,
    allow_patterns: Iterable[str] | None = None,
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ConfigurationError("huggingface_hub is required to download model assets") from exc

    cache_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "repo_id": repo_id,
        "revision": revision,
        "cache_dir": str(cache_dir),
        "token": token,
    }
    if allow_patterns:
        kwargs["allow_patterns"] = list(allow_patterns)
    return Path(snapshot_download(**kwargs)).resolve()


def _allow_patterns(relative: str) -> list[str]:
    return [relative, f"{relative}/**"]


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def link_or_copy(src: Path, dest: Path) -> None:
    src = src.resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        try:
            if dest.resolve() == src:
                return
        except OSError:
            pass
        dest.unlink()
    try:
        os.symlink(src, dest)
    except OSError:
        import shutil

        shutil.copy2(src, dest)
