from __future__ import annotations

from typing import Any


class ServerlessError(Exception):
    code = "serverless_error"
    status_code = 400

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigurationError(ServerlessError):
    code = "configuration_error"
    status_code = 500


class RequestValidationError(ServerlessError):
    code = "request_validation_error"
    status_code = 400


class ReferenceAudioError(ServerlessError):
    code = "reference_audio_error"
    status_code = 400


class ModelLoadError(ServerlessError):
    code = "model_load_error"
    status_code = 500


class InferenceError(ServerlessError):
    code = "inference_error"
    status_code = 500
