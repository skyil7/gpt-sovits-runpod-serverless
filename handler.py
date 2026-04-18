from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpt_sovits_serverless.audio_encode import build_audio_response, encode_audio
from gpt_sovits_serverless.errors import ServerlessError
from gpt_sovits_serverless.pipeline import GPTSoVITSPipeline
from gpt_sovits_serverless.reference_audio import resolved_reference_audio
from gpt_sovits_serverless.request_schema import parse_job_input
from gpt_sovits_serverless.settings import Settings


SETTINGS = Settings.from_env()
PIPELINE = GPTSoVITSPipeline(SETTINGS)

if SETTINGS.load_model_at_startup:
    PIPELINE.load()


def handler(event: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = event.get("input", event)
        request = parse_job_input(payload, SETTINGS)
        with resolved_reference_audio(request, SETTINGS) as references:
            tts_inputs = request.to_tts_inputs(
                ref_audio_path=str(references.primary),
                aux_ref_audio_paths=[str(path) for path in references.aux],
            )
            sample_rate, audio = PIPELINE.synthesize(tts_inputs)
            audio_bytes = encode_audio(audio, sample_rate, request.generation.media_type)
            audio_response = build_audio_response(SETTINGS, audio_bytes, request.generation.media_type)

        return {
            "status": "success",
            "audio": audio_response,
            "meta": {
                "version": SETTINGS.model_version,
                "sample_rate": sample_rate,
                "inference_ms": int((time.perf_counter() - started) * 1000),
            },
        }
    except ServerlessError as exc:
        error = exc.to_dict()
        return {
            "status": "error",
            "error": error,
            "error_info": error,
            "message": error["message"],
            "meta": {"elapsed_ms": int((time.perf_counter() - started) * 1000)},
        }
    except Exception as exc:
        error: dict[str, Any] = {
            "code": "unexpected_error",
            "message": str(exc),
        }
        if os.getenv("DEBUG_ERRORS", "").lower() in {"1", "true", "yes"}:
            error["traceback"] = traceback.format_exc()
        return {
            "status": "error",
            "error": error,
            "error_info": error,
            "message": error["message"],
            "meta": {"elapsed_ms": int((time.perf_counter() - started) * 1000)},
        }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
