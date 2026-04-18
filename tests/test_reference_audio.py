import base64

import pytest

from gpt_sovits_serverless.errors import ReferenceAudioError
from gpt_sovits_serverless.reference_audio import resolved_reference_audio
from gpt_sovits_serverless.request_schema import parse_job_input
from gpt_sovits_serverless.settings import Settings


def test_base64_reference_is_written_to_temp_file(tmp_path):
    audio = b"fake-wav"
    payload = {
        "text": "Hello.",
        "text_lang": "en",
        "reference_voice": {
            "source": "base64",
            "data": base64.b64encode(audio).decode("ascii"),
            "media_type": "wav",
            "prompt_text": "Hello.",
            "prompt_lang": "en",
        },
    }
    settings = Settings.from_env({"REFERENCE_TMP_DIR": str(tmp_path)})
    request = parse_job_input(payload, settings)

    with resolved_reference_audio(request, settings) as references:
        assert references.primary.read_bytes() == audio

    assert list(tmp_path.iterdir()) == []


def test_path_reference_is_disabled_by_default(tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"fake")
    payload = {
        "text": "Hello.",
        "text_lang": "en",
        "reference_voice": {
            "source": "path",
            "path": str(ref),
            "media_type": "wav",
            "prompt_text": "Hello.",
            "prompt_lang": "en",
        },
    }
    settings = Settings.from_env({"REFERENCE_TMP_DIR": str(tmp_path)})
    request = parse_job_input(payload, settings)

    with pytest.raises(ReferenceAudioError):
        with resolved_reference_audio(request, settings):
            pass


def test_local_path_reference_requires_allowed_root(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    ref = allowed / "ref.wav"
    ref.write_bytes(b"fake")
    payload = {
        "text": "Hello.",
        "text_lang": "en",
        "reference_voice": {
            "source": "path",
            "path": str(ref),
            "media_type": "wav",
            "prompt_text": "Hello.",
            "prompt_lang": "en",
        },
    }
    settings = Settings.from_env(
        {
            "REFERENCE_TMP_DIR": str(tmp_path),
            "ALLOW_LOCAL_REFERENCE_PATHS": "true",
            "LOCAL_REFERENCE_ROOTS": str(allowed),
        }
    )
    request = parse_job_input(payload, settings)

    with resolved_reference_audio(request, settings) as references:
        assert references.primary.read_bytes() == b"fake"
