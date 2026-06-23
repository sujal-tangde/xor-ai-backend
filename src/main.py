import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.agent.chat_agent import get_agent
from src.core.config import APP_NAME
from src.core.db import ensure_schema
from src.routers import auth, chat, files, health, projects, reports

app = FastAPI(title=APP_NAME)


@app.on_event("startup")
async def on_startup() -> None:
    ensure_schema()
    try:
        from src.services.reports import ensure_reports_bucket

        await asyncio.to_thread(ensure_reports_bucket)
    except Exception:
        pass
    await asyncio.to_thread(get_agent)

    # Relay worker-published realtime events (insight counts) to connected
    # websockets. Kept on app.state so it isn't garbage-collected.
    from src.services.realtime_bridge import run_subscriber

    app.state.realtime_subscriber = asyncio.create_task(run_subscriber())

# Allow the frontend to connect (tighten allow_origins for production).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(projects.router, prefix="/api", tags=["projects"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])
