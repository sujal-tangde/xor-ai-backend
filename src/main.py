from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import APP_NAME
from src.core.db import ensure_schema
from src.routers import auth, chat, files, health, projects

app = FastAPI(title=APP_NAME)


@app.on_event("startup")
def on_startup() -> None:
    ensure_schema()

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
