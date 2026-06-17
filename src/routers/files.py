from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.core.auth import get_current_user
from src.services import projects_service as ps
from src.services.file_storage import delete_file, list_files, upload_file

router = APIRouter(tags=["files"])


@router.post("/upload")
async def upload(
    project_id: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    if ps.get_project(user["id"], project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        record = await upload_file(
            file.filename, data, file.content_type, user["id"], project_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return record


@router.get("/")
async def get_files(project_id: str, user=Depends(get_current_user)):
    if ps.get_project(user["id"], project_id) is None:
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        return await list_files(user["id"], project_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {exc}") from exc


@router.delete("/{file_id}")
async def remove_file(file_id: str, user=Depends(get_current_user)):
    try:
        deleted = await delete_file(file_id, user["id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"ok": True, "id": file_id}
