from fastapi import FastAPI

from backend.app.controllers.health import router as health_router


def create_app() -> FastAPI:
    application = FastAPI(title="Space Explorer")
    application.include_router(health_router)
    return application


app = create_app()
