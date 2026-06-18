"""Supabase storage + Postgres metadata for uploaded files."""

from __future__ import annotations

import io
import uuid
from typing import Any

import pillow_heif
from PIL import Image
from supabase import Client, create_client

from src.core.config import (
    STORAGE_BUCKET_COMPRESSED,
    STORAGE_BUCKET_ORIGINAL,
    SUPABASE_SERVICE_KEY,
    SUPABASE_URL,
)

pillow_heif.register_heif_opener()

IMAGE_EXTS = {"png", "jpg", "jpeg", "heic", "heif"}
DOC_EXTS = {"pdf", "docx", "txt", "xlsx"}
ALLOWED_EXTS = IMAGE_EXTS | DOC_EXTS

MAX_FILE_BYTES = 10 * 1024 * 1024  # 10 MB

_supabase: Client | None = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def _ext(filename: str) -> str:
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[-1].lower()


def _public_url(bucket: str, path: str) -> str:
    base = SUPABASE_URL.rstrip("/")
    return f"{base}/storage/v1/object/public/{bucket}/{path}"


def _compress_image_to_jpg(data: bytes) -> bytes:
    """Convert any supported image to JPEG, targeting 80–90 KB."""
    img = Image.open(io.BytesIO(data))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    target_min = 80 * 1024
    target_max = 90 * 1024
    quality = 85
    scale = 1.0

    best: bytes | None = None

    for _ in range(30):
        working = img
        if scale < 1.0:
            w, h = img.size
            working = img.resize(
                (max(1, int(w * scale)), max(1, int(h * scale))),
                Image.LANCZOS,
            )

        buf = io.BytesIO()
        working.save(buf, format="JPEG", quality=quality, optimize=True)
        size = buf.tell()
        result = buf.getvalue()

        if target_min <= size <= target_max:
            return result

        if size > target_max:
            if quality > 25:
                quality -= 5
            elif scale > 0.2:
                scale *= 0.85
                quality = 85
            else:
                return result
        else:
            if best is None or abs(size - 85 * 1024) < abs(len(best) - 85 * 1024):
                best = result
            if quality < 95:
                quality += 5
            else:
                return result

    return best or result


def _upload_bytes(bucket: str, path: str, data: bytes, content_type: str) -> str:
    client = get_supabase()
    client.storage.from_(bucket).upload(
        path,
        data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return _public_url(bucket, path)


async def upload_file(
    filename: str,
    data: bytes,
    content_type: str | None,
    user_id: str,
    project_id: str,
) -> dict[str, Any]:
    ext = _ext(filename)
    if ext not in ALLOWED_EXTS:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTS))}"
        )

    if len(data) > MAX_FILE_BYTES:
        raise ValueError("File too large. Maximum allowed size is 10 MB.")

    file_id = str(uuid.uuid4())
    is_image = ext in IMAGE_EXTS
    file_type = "image" if is_image else "document"
    mime = content_type or "application/octet-stream"

    original_path = f"{file_id}/{filename}"
    original_url = _upload_bytes(STORAGE_BUCKET_ORIGINAL, original_path, data, mime)

    compressed_url: str | None = None
    compressed_data: bytes | None = None
    if is_image:
        compressed_data = _compress_image_to_jpg(data)
        compressed_path = f"{file_id}.jpg"
        compressed_url = _upload_bytes(
            STORAGE_BUCKET_COMPRESSED,
            compressed_path,
            compressed_data,
            "image/jpeg",
        )

    row = {
        "id": file_id,
        "name": filename,
        "original_url": original_url,
        "compressed_url": compressed_url,
        "file_type": file_type,
        "mime_type": mime,
        "size_bytes": len(data),
        "processing_status": "pending",
        "user_id": user_id,
        "project_id": project_id,
    }

    client = get_supabase()
    result = client.table("uploaded_files").insert(row).execute()
    record = result.data[0] if result.data else row

    # Hand the heavy work to the RQ worker so the upload request returns fast.
    from src.services.queue import (
        enqueue_document_processing,
        enqueue_image_processing,
    )

    if is_image and compressed_data is not None:
        enqueue_image_processing(
            file_id, project_id, user_id, compressed_data, filename
        )
    elif not is_image:
        enqueue_document_processing(file_id, project_id, user_id, data, ext, filename)

    return record


async def list_files(user_id: str, project_id: str) -> list[dict[str, Any]]:
    """List files for a project. Files are shared across all chats in the project."""
    client = get_supabase()
    result = (
        client.table("uploaded_files")
        .select(
            "id, name, original_url, compressed_url, file_type, mime_type, "
            "size_bytes, processing_status, created_at, project_id"
        )
        .eq("user_id", user_id)
        .eq("project_id", project_id)
        .order("created_at", desc=True)
        .execute()
    )
    return result.data or []


def get_files_by_ids(file_ids: list[str]) -> list[dict[str, Any]]:
    """Fetch file metadata and processing status for the given IDs."""
    if not file_ids:
        return []

    client = get_supabase()
    result = (
        client.table("uploaded_files")
        .select("id, name, file_type, processing_status, project_id")
        .in_("id", file_ids)
        .execute()
    )
    return result.data or []


def _storage_path_from_url(url: str, bucket: str) -> str | None:
    marker = f"/object/public/{bucket}/"
    idx = url.find(marker)
    if idx == -1:
        return None
    return url[idx + len(marker) :]


async def delete_file(file_id: str, user_id: str) -> bool:
    client = get_supabase()
    result = (
        client.table("uploaded_files")
        .select("*")
        .eq("id", file_id)
        .eq("user_id", user_id)
        .execute()
    )
    if not result.data:
        return False

    record = result.data[0]

    original_path = _storage_path_from_url(record["original_url"], STORAGE_BUCKET_ORIGINAL)
    if original_path:
        client.storage.from_(STORAGE_BUCKET_ORIGINAL).remove([original_path])

    if record.get("compressed_url"):
        compressed_path = _storage_path_from_url(
            record["compressed_url"], STORAGE_BUCKET_COMPRESSED
        )
        if compressed_path:
            client.storage.from_(STORAGE_BUCKET_COMPRESSED).remove([compressed_path])

    # Remove derived RAG chunks and the per-upload insight so deleted files don't
    # linger in search results. (The project_knowledge_base row is an accumulated
    # merge and is not un-merged here.)
    for table in ("image_chunks", "file_chunks", "project_insights"):
        try:
            client.table(table).delete().eq("file_id", file_id).execute()
        except Exception:
            pass

    client.table("uploaded_files").delete().eq("id", file_id).eq(
        "user_id", user_id
    ).execute()
    return True
