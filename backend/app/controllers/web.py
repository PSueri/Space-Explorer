from fastapi import APIRouter
from fastapi.responses import FileResponse

from backend.app.core.settings import PROJECT_ROOT

router = APIRouter(include_in_schema=False)


@router.get("/", response_class=FileResponse)
async def home() -> FileResponse:
    return FileResponse(
        PROJECT_ROOT / "frontend" / "index.html",
        media_type="text/html",
    )
