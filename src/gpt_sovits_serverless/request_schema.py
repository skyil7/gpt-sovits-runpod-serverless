from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import RequestValidationError
from .settings import SUPPORTED_MEDIA_TYPES, Settings


SUPPORTED_LANGS = {
    "auto",
    "auto_yue",
    "en",
    "zh",
    "ja",
    "yue",
    "ko",
    "all_zh",
    "all_ja",
    "all_yue",
    "all_ko",
}
SUPPORTED_REFERENCE_SOURCES = {"base64", "url", "hf", "path"}
SUPPORTED_CUT_METHODS = {"cut0", "cut1", "cut2", "cut3", "cut4", "cut5"}


@dataclass(frozen=True)
class ReferenceVoice:
    source: str
    data: str | None
    path: str | None
    media_type: str
    prompt_text: str
    prompt_lang: str

    @classmethod
    def from_mapping(cls, value: Any, *, role: str, default_media_type: str) -> "ReferenceVoice":
        if not isinstance(value, Mapping):
            raise RequestValidationError(f"{role} must be an object")

        source = _string(value.get("source", "base64"), f"{role}.source").lower()
        if source not in SUPPORTED_REFERENCE_SOURCES:
            raise RequestValidationError(f"{role}.source must be one of {sorted(SUPPORTED_REFERENCE_SOURCES)}")

        media_type = _string(value.get("media_type", default_media_type), f"{role}.media_type").lower()
        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise RequestValidationError(f"{role}.media_type must be one of {sorted(SUPPORTED_MEDIA_TYPES)}")

        data = value.get("data")
        path = value.get("path")
        if source == "url":
            data = value.get("url", data)
        if source in {"hf", "path"}:
            path = value.get("path", data)

        if source in {"base64", "url"} and not _optional_string(data):
            raise RequestValidationError(f"{role}.data is required for source={source!r}")
        if source in {"hf", "path"} and not _optional_string(path):
            raise RequestValidationError(f"{role}.path is required for source={source!r}")

        prompt_lang_value = value.get("prompt_lang")
        if prompt_lang_value is None and role.startswith("aux_reference_voices"):
            prompt_lang_value = "auto"
        prompt_lang = _string(prompt_lang_value, f"{role}.prompt_lang").lower()
        if prompt_lang not in SUPPORTED_LANGS:
            raise RequestValidationError(f"{role}.prompt_lang must be one of {sorted(SUPPORTED_LANGS)}")

        return cls(
            source=source,
            data=_optional_string(data),
            path=_optional_string(path),
            media_type=media_type,
            prompt_text=_optional_string(value.get("prompt_text")) or "",
            prompt_lang=prompt_lang,
        )


@dataclass(frozen=True)
class GenerationSettings:
    top_k: int = 15
    top_p: float = 1.0
    temperature: float = 1.0
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    parallel_infer: bool = True
    repetition_penalty: float = 1.35
    sample_steps: int = 32
    super_sampling: bool = False
    media_type: str = "wav"

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None, *, default_media_type: str) -> "GenerationSettings":
        value = value or {}
        media_type = _string(value.get("media_type", default_media_type), "generation.media_type").lower()
        if media_type not in SUPPORTED_MEDIA_TYPES:
            raise RequestValidationError(f"generation.media_type must be one of {sorted(SUPPORTED_MEDIA_TYPES)}")

        text_split_method = _string(
            value.get("text_split_method", "cut5"), "generation.text_split_method"
        ).lower()
        if text_split_method not in SUPPORTED_CUT_METHODS:
            raise RequestValidationError(
                f"generation.text_split_method must be one of {sorted(SUPPORTED_CUT_METHODS)}"
            )

        settings = cls(
            top_k=_int(value.get("top_k", 15), "generation.top_k", minimum=1, maximum=100),
            top_p=_float(value.get("top_p", 1.0), "generation.top_p", minimum=0.0, maximum=1.0),
            temperature=_float(value.get("temperature", 1.0), "generation.temperature", minimum=0.01, maximum=5.0),
            text_split_method=text_split_method,
            batch_size=_int(value.get("batch_size", 1), "generation.batch_size", minimum=1, maximum=32),
            batch_threshold=_float(
                value.get("batch_threshold", 0.75), "generation.batch_threshold", minimum=0.0, maximum=1.0
            ),
            split_bucket=_bool(value.get("split_bucket", True), "generation.split_bucket"),
            speed_factor=_float(value.get("speed_factor", 1.0), "generation.speed_factor", minimum=0.25, maximum=4.0),
            fragment_interval=_float(
                value.get("fragment_interval", 0.3), "generation.fragment_interval", minimum=0.01, maximum=5.0
            ),
            seed=_int(value.get("seed", -1), "generation.seed", minimum=-1),
            parallel_infer=_bool(value.get("parallel_infer", True), "generation.parallel_infer"),
            repetition_penalty=_float(
                value.get("repetition_penalty", 1.35), "generation.repetition_penalty", minimum=0.1, maximum=10.0
            ),
            sample_steps=_int(value.get("sample_steps", 32), "generation.sample_steps", minimum=1, maximum=128),
            super_sampling=_bool(value.get("super_sampling", False), "generation.super_sampling"),
            media_type=media_type,
        )
        return settings

    def to_tts_kwargs(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "text_split_method": self.text_split_method,
            "batch_size": self.batch_size,
            "batch_threshold": self.batch_threshold,
            "split_bucket": self.split_bucket,
            "speed_factor": self.speed_factor,
            "fragment_interval": self.fragment_interval,
            "seed": self.seed,
            "parallel_infer": self.parallel_infer,
            "repetition_penalty": self.repetition_penalty,
            "sample_steps": self.sample_steps,
            "super_sampling": self.super_sampling,
            "streaming_mode": False,
            "return_fragment": False,
            "fixed_length_chunk": False,
            "overlap_length": 2,
            "min_chunk_length": 16,
        }


@dataclass(frozen=True)
class TTSJobRequest:
    text: str
    text_lang: str
    reference_voice: ReferenceVoice
    aux_reference_voices: tuple[ReferenceVoice, ...]
    generation: GenerationSettings

    def to_tts_inputs(self, *, ref_audio_path: str, aux_ref_audio_paths: list[str]) -> dict[str, Any]:
        payload = {
            "text": self.text,
            "text_lang": self.text_lang,
            "ref_audio_path": ref_audio_path,
            "aux_ref_audio_paths": aux_ref_audio_paths,
            "prompt_text": self.reference_voice.prompt_text,
            "prompt_lang": self.reference_voice.prompt_lang,
        }
        payload.update(self.generation.to_tts_kwargs())
        return payload


GENERATION_KEYS = set(GenerationSettings.__dataclass_fields__.keys())


def parse_job_input(payload: Any, settings: Settings) -> TTSJobRequest:
    if isinstance(payload, Mapping) and "input" in payload and isinstance(payload["input"], Mapping):
        payload = payload["input"]
    if not isinstance(payload, Mapping):
        raise RequestValidationError("RunPod input must be an object")

    text = _string(payload.get("text"), "text")
    if len(text) > settings.max_text_chars:
        raise RequestValidationError(
            f"text exceeds MAX_TEXT_CHARS={settings.max_text_chars}",
            details={"actual": len(text)},
        )

    text_lang = _string(payload.get("text_lang"), "text_lang").lower()
    if text_lang not in SUPPORTED_LANGS:
        raise RequestValidationError(f"text_lang must be one of {sorted(SUPPORTED_LANGS)}")

    reference_payload = payload.get("reference_voice")
    if reference_payload is None and "ref_audio_path" in payload:
        reference_payload = {
            "source": "path",
            "path": payload.get("ref_audio_path"),
            "prompt_text": payload.get("prompt_text", ""),
            "prompt_lang": payload.get("prompt_lang"),
        }
    reference_voice = ReferenceVoice.from_mapping(
        reference_payload,
        role="reference_voice",
        default_media_type=settings.default_media_type,
    )

    aux_payload = payload.get("aux_reference_voices", [])
    if aux_payload is None:
        aux_payload = []
    if not isinstance(aux_payload, list):
        raise RequestValidationError("aux_reference_voices must be a list")
    aux_reference_voices = tuple(
        ReferenceVoice.from_mapping(item, role=f"aux_reference_voices[{index}]", default_media_type=settings.default_media_type)
        for index, item in enumerate(aux_payload)
    )

    generation_payload = dict(payload.get("generation") or {})
    for key in GENERATION_KEYS:
        if key in payload:
            generation_payload[key] = payload[key]
    generation = GenerationSettings.from_mapping(generation_payload, default_media_type=settings.default_media_type)

    return TTSJobRequest(
        text=text,
        text_lang=text_lang,
        reference_voice=reference_voice,
        aux_reference_voices=aux_reference_voices,
        generation=generation,
    )


def _string(value: Any, field: str) -> str:
    value = _optional_string(value)
    if value is None:
        raise RequestValidationError(f"{field} is required")
    return value


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RequestValidationError("Expected a string value")
    value = value.strip()
    return value or None


def _int(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{field} must be an integer") from exc
    if minimum is not None and result < minimum:
        raise RequestValidationError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise RequestValidationError(f"{field} must be <= {maximum}")
    return result


def _float(value: Any, field: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError(f"{field} must be a number") from exc
    if minimum is not None and result < minimum:
        raise RequestValidationError(f"{field} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise RequestValidationError(f"{field} must be <= {maximum}")
    return result


def _bool(value: Any, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    raise RequestValidationError(f"{field} must be a boolean")
