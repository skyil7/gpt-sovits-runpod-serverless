from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .errors import ConfigurationError


SUPPORTED_VERSIONS = {"v2ProPlus"}
SUPPORTED_OUTPUT_MODES = {"base64", "s3"}
SUPPORTED_MEDIA_TYPES = {"wav", "raw", "ogg", "aac"}


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _bool(value: str | bool | None, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value!r}")


def _int(value: str | None, *, default: int, minimum: int | None = None) -> int:
    if value is None or value == "":
        result = default
    else:
        try:
            result = int(value)
        except ValueError as exc:
            raise ConfigurationError(f"Invalid integer value: {value!r}") from exc
    if minimum is not None and result < minimum:
        raise ConfigurationError(f"Integer value must be >= {minimum}: {result}")
    return result


def _float(value: str | None, *, default: float, minimum: float | None = None) -> float:
    if value is None or value == "":
        result = default
    else:
        try:
            result = float(value)
        except ValueError as exc:
            raise ConfigurationError(f"Invalid float value: {value!r}") from exc
    if minimum is not None and result < minimum:
        raise ConfigurationError(f"Float value must be >= {minimum}: {result}")
    return result


def _path(value: str | None, *, default: str) -> Path:
    return Path(value or default).expanduser()


def _split_paths(value: str | None) -> tuple[Path, ...]:
    if not value:
        return ()
    return tuple(Path(item).expanduser().resolve() for item in value.split(os.pathsep) if item.strip())


@dataclass(frozen=True)
class Settings:
    model_version: str
    gpt_sovits_root: Path
    hf_model_repo_id: str | None
    hf_model_revision: str
    hf_pretrained_repo_id: str
    hf_pretrained_revision: str
    gpt_weight_path: str
    sovits_weight_path: str
    base_gpt_weight_path: str
    base_sovits_weight_path: str
    bert_model_path: str
    cnhubert_model_path: str
    sv_model_path: str
    device: str
    is_half: bool
    output_mode: str
    default_media_type: str
    max_text_chars: int
    max_reference_audio_mb: int
    model_cache_dir: Path
    reference_tmp_dir: Path
    reference_download_timeout_s: float
    hf_token: str | None
    load_model_at_startup: bool
    allow_local_reference_paths: bool
    local_reference_roots: tuple[Path, ...]
    s3_endpoint_url: str | None
    s3_bucket: str | None
    s3_access_key_id: str | None
    s3_secret_access_key: str | None
    s3_region: str | None
    s3_public_base_url: str | None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        if env is None:
            env = os.environ
        model_cache_dir = _path(env.get("MODEL_CACHE_DIR"), default="/runpod-volume/gpt-sovits")
        gpt_sovits_root = _path(env.get("GPT_SOVITS_ROOT"), default="/opt/GPT-SoVITS")
        local_roots = _split_paths(env.get("LOCAL_REFERENCE_ROOTS"))
        if not local_roots:
            local_roots = (model_cache_dir.resolve(), gpt_sovits_root.resolve())

        settings = cls(
            model_version=env.get("MODEL_VERSION", "v2ProPlus").strip(),
            gpt_sovits_root=gpt_sovits_root,
            hf_model_repo_id=_clean(env.get("HF_MODEL_REPO_ID")),
            hf_model_revision=env.get("HF_MODEL_REVISION", "main").strip() or "main",
            hf_pretrained_repo_id=env.get("HF_PRETRAINED_REPO_ID", "lj1995/GPT-SoVITS").strip(),
            hf_pretrained_revision=env.get("HF_PRETRAINED_REVISION", "main").strip() or "main",
            gpt_weight_path=env.get("GPT_WEIGHT_PATH", "s1v3.ckpt").strip(),
            sovits_weight_path=env.get("SOVITS_WEIGHT_PATH", "v2Pro/s2Gv2ProPlus.pth").strip(),
            base_gpt_weight_path=env.get("BASE_GPT_WEIGHT_PATH", "s1v3.ckpt").strip(),
            base_sovits_weight_path=env.get(
                "BASE_SOVITS_WEIGHT_PATH", "v2Pro/s2Gv2ProPlus.pth"
            ).strip(),
            bert_model_path=env.get("BERT_MODEL_PATH", "chinese-roberta-wwm-ext-large").strip(),
            cnhubert_model_path=env.get("CNHUBERT_MODEL_PATH", "chinese-hubert-base").strip(),
            sv_model_path=env.get("SV_MODEL_PATH", "sv/pretrained_eres2netv2w24s4ep4.ckpt").strip(),
            device=env.get("DEVICE", "cuda").strip() or "cuda",
            is_half=_bool(env.get("IS_HALF"), default=True),
            output_mode=env.get("OUTPUT_MODE", "base64").strip().lower(),
            default_media_type=env.get("DEFAULT_MEDIA_TYPE", "wav").strip().lower(),
            max_text_chars=_int(env.get("MAX_TEXT_CHARS"), default=1200, minimum=1),
            max_reference_audio_mb=_int(env.get("MAX_REFERENCE_AUDIO_MB"), default=20, minimum=1),
            model_cache_dir=model_cache_dir,
            reference_tmp_dir=_path(env.get("REFERENCE_TMP_DIR"), default="/tmp"),
            reference_download_timeout_s=_float(
                env.get("REFERENCE_DOWNLOAD_TIMEOUT_S"), default=30.0, minimum=0.1
            ),
            hf_token=_clean(env.get("HF_TOKEN")),
            load_model_at_startup=_bool(env.get("LOAD_MODEL_AT_STARTUP"), default=False),
            allow_local_reference_paths=_bool(env.get("ALLOW_LOCAL_REFERENCE_PATHS"), default=False),
            local_reference_roots=local_roots,
            s3_endpoint_url=_clean(env.get("S3_ENDPOINT_URL")),
            s3_bucket=_clean(env.get("S3_BUCKET")),
            s3_access_key_id=_clean(env.get("S3_ACCESS_KEY_ID")),
            s3_secret_access_key=_clean(env.get("S3_SECRET_ACCESS_KEY")),
            s3_region=_clean(env.get("S3_REGION")),
            s3_public_base_url=_clean(env.get("S3_PUBLIC_BASE_URL")),
        )
        settings.validate()
        return settings

    @property
    def max_reference_audio_bytes(self) -> int:
        return self.max_reference_audio_mb * 1024 * 1024

    @property
    def hf_cache_dir(self) -> Path:
        return self.model_cache_dir / "huggingface"

    def validate(self) -> None:
        if self.model_version not in SUPPORTED_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_VERSIONS))
            raise ConfigurationError(f"Unsupported MODEL_VERSION={self.model_version!r}; supported: {supported}")
        if self.output_mode not in SUPPORTED_OUTPUT_MODES:
            raise ConfigurationError(f"Unsupported OUTPUT_MODE={self.output_mode!r}")
        if self.default_media_type not in SUPPORTED_MEDIA_TYPES:
            raise ConfigurationError(f"Unsupported DEFAULT_MEDIA_TYPE={self.default_media_type!r}")
        if not self.hf_pretrained_repo_id:
            raise ConfigurationError("HF_PRETRAINED_REPO_ID is required")
        if not self.gpt_weight_path:
            raise ConfigurationError("GPT_WEIGHT_PATH is required")
        if not self.sovits_weight_path:
            raise ConfigurationError("SOVITS_WEIGHT_PATH is required")
        if self.output_mode == "s3":
            missing = [
                key
                for key, value in {
                    "S3_BUCKET": self.s3_bucket,
                    "S3_ACCESS_KEY_ID": self.s3_access_key_id,
                    "S3_SECRET_ACCESS_KEY": self.s3_secret_access_key,
                }.items()
                if not value
            ]
            if missing:
                raise ConfigurationError(f"OUTPUT_MODE=s3 requires: {', '.join(missing)}")
