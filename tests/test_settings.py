import pytest

from gpt_sovits_serverless.errors import ConfigurationError
from gpt_sovits_serverless.settings import Settings


def test_settings_defaults_to_v2proplus():
    settings = Settings.from_env({})

    assert settings.model_version == "v2ProPlus"
    assert settings.gpt_weight_path == "s1v3.ckpt"
    assert settings.sovits_weight_path == "v2Pro/s2Gv2ProPlus.pth"
    assert settings.output_mode == "base64"


def test_settings_rejects_other_versions():
    with pytest.raises(ConfigurationError):
        Settings.from_env({"MODEL_VERSION": "v4"})


def test_settings_parses_boolean_values():
    settings = Settings.from_env({"IS_HALF": "false", "LOAD_MODEL_AT_STARTUP": "yes"})

    assert settings.is_half is False
    assert settings.load_model_at_startup is True


def test_s3_output_requires_credentials():
    with pytest.raises(ConfigurationError):
        Settings.from_env({"OUTPUT_MODE": "s3"})
