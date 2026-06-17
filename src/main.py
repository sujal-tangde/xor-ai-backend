from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import APP_NAME
from src.routers import chat, files, health
from src.services.file_storage import ensure_files_table

app = FastAPI(title=APP_NAME)


@app.on_event("startup")
def on_startup() -> None:
    ensure_files_table()

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
app.include_router(files.router, prefix="/api/files", tags=["files"])
