from fastapi import FastAPI

from backend.app.controllers.chat import router as chat_router
from backend.app.controllers.health import router as health_router
from backend.app.core.settings import Settings


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = Settings()

    application = FastAPI(
        title=settings.name,
        debug=settings.debug,
    )
    application.state.settings = settings
    application.include_router(health_router)
    application.include_router(chat_router)
    return application


app = create_app()
