from handler import handler


def test_healthcheck_returns_without_tts_payload():
    response = handler({"input": {"healthcheck": True}})

    assert response["status"] == "success"
    assert response["healthcheck"] is True
    assert response["meta"]["version"] == "v2ProPlus"


def test_healthcheck_accepts_mode_alias():
    response = handler({"input": {"mode": "healthcheck"}})

    assert response["status"] == "success"
    assert response["healthcheck"] is True
