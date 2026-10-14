from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app


def test_health_returns_ok() -> None:
    settings = Settings(
        _env_file=None,
        name="Space Explorer Test",
        debug=False,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
