from __future__ import annotations

import base64
import binascii
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .errors import ReferenceAudioError
from .hf_resolver import resolve_asset_path
from .request_schema import ReferenceVoice, TTSJobRequest
from .settings import Settings


@dataclass(frozen=True)
class ResolvedReferences:
    primary: Path
    aux: tuple[Path, ...]


@contextmanager
def resolved_reference_audio(request: TTSJobRequest, settings: Settings):
    with tempfile.TemporaryDirectory(prefix="gsv_job_", dir=str(settings.reference_tmp_dir)) as tmp:
        resolver = ReferenceAudioResolver(settings=settings, tmp_dir=Path(tmp))
        primary = resolver.resolve(request.reference_voice, name="reference")
        aux = tuple(
            resolver.resolve(reference, name=f"aux_{index}")
            for index, reference in enumerate(request.aux_reference_voices)
        )
        yield ResolvedReferences(primary=primary, aux=aux)


class ReferenceAudioResolver:
    def __init__(self, *, settings: Settings, tmp_dir: Path):
        self.settings = settings
        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, reference: ReferenceVoice, *, name: str) -> Path:
        if reference.source == "base64":
            return self._resolve_base64(reference, name=name)
        if reference.source == "url":
            return self._resolve_url(reference, name=name)
        if reference.source == "hf":
            return self._resolve_hf(reference, name=name)
        if reference.source == "path":
            return self._resolve_path(reference, name=name)
        raise ReferenceAudioError(f"Unsupported reference source: {reference.source}")

    def _resolve_base64(self, reference: ReferenceVoice, *, name: str) -> Path:
        assert reference.data is not None
        payload = reference.data
        if payload.startswith("data:") and "," in payload:
            payload = payload.split(",", 1)[1]
        try:
            audio = base64.b64decode(payload, validate=True)
        except binascii.Error as exc:
            raise ReferenceAudioError("reference_voice.data is not valid base64") from exc
        return self._write_bytes(audio, media_type=reference.media_type, name=name)

    def _resolve_url(self, reference: ReferenceVoice, *, name: str) -> Path:
        assert reference.data is not None
        parsed = urlparse(reference.data)
        if parsed.scheme not in {"http", "https"}:
            raise ReferenceAudioError("reference_voice url must use http or https")

        request = Request(reference.data, headers={"User-Agent": "gpt-sovits-runpod-serverless/0.1"})
        try:
            with urlopen(request, timeout=self.settings.reference_download_timeout_s) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) > self.settings.max_reference_audio_bytes:
                    raise ReferenceAudioError("reference_voice url exceeds MAX_REFERENCE_AUDIO_MB")
                audio = _read_limited(response, self.settings.max_reference_audio_bytes)
        except ReferenceAudioError:
            raise
        except Exception as exc:
            raise ReferenceAudioError(f"Failed to download reference audio: {exc}") from exc
        return self._write_bytes(audio, media_type=reference.media_type, name=name)

    def _resolve_hf(self, reference: ReferenceVoice, *, name: str) -> Path:
        assert reference.path is not None
        repo_id = self.settings.hf_model_repo_id or self.settings.hf_pretrained_repo_id
        revision = (
            self.settings.hf_model_revision
            if self.settings.hf_model_repo_id
            else self.settings.hf_pretrained_revision
        )
        source = resolve_asset_path(
            reference.path,
            repo_id=repo_id,
            revision=revision,
            cache_dir=self.settings.hf_cache_dir,
            token=self.settings.hf_token,
        )
        return self._copy_file(source, media_type=reference.media_type, name=name)

    def _resolve_path(self, reference: ReferenceVoice, *, name: str) -> Path:
        if not self.settings.allow_local_reference_paths:
            raise ReferenceAudioError("source='path' requires ALLOW_LOCAL_REFERENCE_PATHS=true")
        assert reference.path is not None
        source = Path(reference.path).expanduser().resolve()
        if not source.exists() or not source.is_file():
            raise ReferenceAudioError(f"Reference audio path does not exist: {source}")
        if not any(_is_relative_to(source, root) for root in self.settings.local_reference_roots):
            raise ReferenceAudioError("Reference audio path is outside LOCAL_REFERENCE_ROOTS")
        return self._copy_file(source, media_type=reference.media_type, name=name)

    def _write_bytes(self, audio: bytes, *, media_type: str, name: str) -> Path:
        if len(audio) > self.settings.max_reference_audio_bytes:
            raise ReferenceAudioError("reference audio exceeds MAX_REFERENCE_AUDIO_MB")
        if not audio:
            raise ReferenceAudioError("reference audio is empty")
        path = self.tmp_dir / f"{name}.{_extension(media_type)}"
        path.write_bytes(audio)
        return path

    def _copy_file(self, source: Path, *, media_type: str, name: str) -> Path:
        size = source.stat().st_size
        if size > self.settings.max_reference_audio_bytes:
            raise ReferenceAudioError("reference audio exceeds MAX_REFERENCE_AUDIO_MB")
        if size == 0:
            raise ReferenceAudioError("reference audio is empty")
        dest = self.tmp_dir / f"{name}.{source.suffix.lstrip('.') or _extension(media_type)}"
        shutil.copy2(source, dest)
        return dest


def _read_limited(response, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ReferenceAudioError("reference audio exceeds MAX_REFERENCE_AUDIO_MB")
        chunks.append(chunk)
    return b"".join(chunks)


def _extension(media_type: str) -> str:
    return "pcm" if media_type == "raw" else media_type


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False
