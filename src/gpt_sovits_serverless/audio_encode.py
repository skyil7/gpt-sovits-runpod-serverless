from __future__ import annotations

import base64
import subprocess
import uuid
import wave
from io import BytesIO
from typing import Any

from .errors import ConfigurationError, InferenceError
from .settings import Settings


CONTENT_TYPES = {
    "wav": "audio/wav",
    "raw": "audio/L16",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
}


def encode_audio(audio: Any, sample_rate: int, media_type: str) -> bytes:
    pcm = _pcm16_bytes(audio)
    if media_type == "raw":
        return pcm
    if media_type == "wav":
        return _encode_wav(pcm, sample_rate)
    if media_type == "aac":
        return _encode_ffmpeg(pcm, sample_rate, codec="aac", container="adts")
    if media_type == "ogg":
        return _encode_ffmpeg(pcm, sample_rate, codec="libvorbis", container="ogg")
    raise InferenceError(f"Unsupported media_type={media_type!r}")


def build_audio_response(settings: Settings, audio_bytes: bytes, media_type: str) -> dict[str, str]:
    if settings.output_mode == "base64":
        return {
            "media_type": media_type,
            "content_type": CONTENT_TYPES.get(media_type, "application/octet-stream"),
            "encoding": "base64",
            "data": base64.b64encode(audio_bytes).decode("ascii"),
        }
    if settings.output_mode == "s3":
        return _upload_s3(settings, audio_bytes, media_type)
    raise ConfigurationError(f"Unsupported OUTPUT_MODE={settings.output_mode!r}")


def _pcm16_bytes(audio: Any) -> bytes:
    if isinstance(audio, bytes):
        return audio
    if isinstance(audio, bytearray):
        return bytes(audio)

    try:
        import numpy as np
    except ImportError as exc:
        raise InferenceError("numpy is required to encode generated audio") from exc

    array = np.asarray(audio)
    if array.ndim > 1:
        array = array.reshape(-1)
    if array.dtype.kind == "f":
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767.0).astype("<i2")
    elif array.dtype != np.dtype("<i2"):
        array = array.astype("<i2")
    return array.tobytes()


def _encode_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)
    return buffer.getvalue()


def _encode_ffmpeg(pcm: bytes, sample_rate: int, *, codec: str, container: str) -> bytes:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-c:a",
        codec,
        "-f",
        container,
        "pipe:1",
    ]
    try:
        process = subprocess.run(command, input=pcm, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise InferenceError("ffmpeg is required for ogg/aac output") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace")
        raise InferenceError(f"ffmpeg audio encoding failed: {stderr}") from exc
    return process.stdout


def _upload_s3(settings: Settings, audio_bytes: bytes, media_type: str) -> dict[str, str]:
    if not settings.s3_bucket:
        raise ConfigurationError("S3_BUCKET is required for OUTPUT_MODE=s3")
    try:
        import boto3
    except ImportError as exc:
        raise ConfigurationError("boto3 is required for OUTPUT_MODE=s3") from exc

    key = f"outputs/{uuid.uuid4().hex}.{media_type}"
    client = boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        region_name=settings.s3_region,
    )
    client.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=audio_bytes,
        ContentType=CONTENT_TYPES.get(media_type, "application/octet-stream"),
    )
    if settings.s3_public_base_url:
        url = f"{settings.s3_public_base_url.rstrip('/')}/{key}"
    else:
        url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.s3_bucket, "Key": key},
            ExpiresIn=3600,
        )
    return {
        "media_type": media_type,
        "content_type": CONTENT_TYPES.get(media_type, "application/octet-stream"),
        "encoding": "url",
        "url": url,
    }
