import pytest
from fastapi.testclient import TestClient

from backend.app.core.settings import Settings
from backend.app.main import create_app


@pytest.mark.parametrize(
    ("path", "expected_content_type"),
    [
        ("/", "text/html"),
        ("/static/styles.css", "text/css"),
        ("/static/app.js", "text/javascript"),
    ],
)
def test_frontend_resources_are_available(
    path: str,
    expected_content_type: str,
) -> None:
    settings = Settings(
        _env_file=None,
        name="Space Explorer Test",
        debug=False,
    )

    with TestClient(create_app(settings)) as client:
        response = client.get(path)

    assert response.status_code == 200
    content_type = response.headers["content-type"].split(";")[0].strip()
    assert content_type == expected_content_type
