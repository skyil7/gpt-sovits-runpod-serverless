from pathlib import Path

import pytest

from gpt_sovits_serverless.errors import ConfigurationError
from gpt_sovits_serverless.hf_resolver import safe_join, validate_relative_path


def test_validate_relative_path_allows_nested_paths():
    assert validate_relative_path("v2Pro/s2Gv2ProPlus.pth") == "v2Pro/s2Gv2ProPlus.pth"


@pytest.mark.parametrize("value", ["../secret", "/tmp/secret", "a/../../b", ""])
def test_validate_relative_path_rejects_unsafe_paths(value):
    with pytest.raises(ConfigurationError):
        validate_relative_path(value)


def test_safe_join_stays_inside_root(tmp_path: Path):
    assert safe_join(tmp_path, "a/b.txt") == (tmp_path / "a" / "b.txt").resolve()
