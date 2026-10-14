from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.core.settings import Settings
from backend.app.main import create_app


def test_environment_overrides_dotenv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        'APP_NAME="Name from file"\nAPP_DEBUG=false\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("APP_NAME", "Name from environment")
    monkeypatch.delenv("APP_DEBUG", raising=False)

    settings = Settings(_env_file=env_file)

    assert settings.name == "Name from environment"
    assert settings.debug is False


def test_invalid_debug_value_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_DEBUG", "invalid")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, name="Space Explorer Test")


def test_application_uses_explicit_settings() -> None:
    settings = Settings(
        _env_file=None,
        name="Custom Application",
        debug=True,
    )

    application = create_app(settings)

    assert application.title == "Custom Application"
    assert application.debug is True
