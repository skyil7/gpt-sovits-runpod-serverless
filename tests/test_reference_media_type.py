from gpt_sovits_serverless.request_schema import parse_job_input
from gpt_sovits_serverless.settings import Settings


def test_reference_media_type_accepts_flac_independently_from_output_media_type():
    payload = {
        "text": "Hello.",
        "text_lang": "en",
        "reference_voice": {
            "source": "url",
            "url": "https://example.com/ref.flac",
            "media_type": "flac",
            "prompt_text": "Hello.",
            "prompt_lang": "en",
        },
        "generation": {
            "media_type": "wav",
        },
    }

    request = parse_job_input(payload, Settings.from_env({}))

    assert request.reference_voice.media_type == "flac"
    assert request.generation.media_type == "wav"
