import pytest

from gpt_sovits_serverless.errors import RequestValidationError
from gpt_sovits_serverless.request_schema import parse_job_input
from gpt_sovits_serverless.settings import Settings


def valid_payload():
    return {
        "text": "Hello.",
        "text_lang": "en",
        "reference_voice": {
            "source": "base64",
            "data": "UklGRgAAAAA=",
            "media_type": "wav",
            "prompt_text": "Hello.",
            "prompt_lang": "en",
        },
        "generation": {
            "text_split_method": "cut5",
            "media_type": "wav",
        },
    }


def test_parse_valid_payload():
    request = parse_job_input(valid_payload(), Settings.from_env({}))

    assert request.text == "Hello."
    assert request.text_lang == "en"
    assert request.reference_voice.source == "base64"
    assert request.generation.text_split_method == "cut5"


def test_parse_rejects_long_text():
    payload = valid_payload()
    payload["text"] = "x" * 6

    with pytest.raises(RequestValidationError):
        parse_job_input(payload, Settings.from_env({"MAX_TEXT_CHARS": "5"}))


def test_parse_rejects_bad_language():
    payload = valid_payload()
    payload["text_lang"] = "fr"

    with pytest.raises(RequestValidationError):
        parse_job_input(payload, Settings.from_env({}))


def test_generation_can_be_top_level_for_compatibility():
    payload = valid_payload()
    payload["temperature"] = 0.5

    request = parse_job_input(payload, Settings.from_env({}))

    assert request.generation.temperature == 0.5


def test_aux_reference_voice_does_not_require_prompt_lang():
    payload = valid_payload()
    payload["aux_reference_voices"] = [
        {
            "source": "base64",
            "data": "UklGRgAAAAA=",
            "media_type": "wav",
        }
    ]

    request = parse_job_input(payload, Settings.from_env({}))

    assert request.aux_reference_voices[0].prompt_lang == "auto"
