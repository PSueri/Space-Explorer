import mimetypes

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.controllers.chat import router as chat_router
from backend.app.controllers.health import router as health_router
from backend.app.controllers.web import router as web_router
from backend.app.core.settings import PROJECT_ROOT, Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    mimetypes.add_type("text/javascript", ".js")

    application = FastAPI(
        title=settings.name,
        debug=settings.debug,
    )
    application.state.settings = settings

    application.mount(
        "/static",
        StaticFiles(directory=PROJECT_ROOT / "frontend"),
        name="static",
    )

    application.include_router(health_router)
    application.include_router(chat_router)
    application.include_router(web_router)
    return application


app = create_app()
