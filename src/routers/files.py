from fastapi import APIRouter, File, HTTPException, UploadFile

from src.services.file_storage import delete_file, list_files, upload_file

router = APIRouter(tags=["files"])


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        record = await upload_file(file.filename, data, file.content_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

    return record


@router.get("/")
async def get_files():
    try:
        return await list_files()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to list files: {exc}") from exc


@router.delete("/{file_id}")
async def remove_file(file_id: str):
    try:
        deleted = await delete_file(file_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Delete failed: {exc}") from exc

    if not deleted:
        raise HTTPException(status_code=404, detail="File not found")
    return {"ok": True, "id": file_id}
