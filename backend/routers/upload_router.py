from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import aiofiles
from uuid import uuid4

router = APIRouter(prefix="/upload", tags=["upload"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
os.makedirs(os.path.join(os.path.dirname(UPLOAD_DIR), "uploads"), exist_ok=True)


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    ext = os.path.splitext(file.filename)[1]
    fname = f"{uuid4().hex}{ext}"
    out_path = os.path.join(os.path.dirname(UPLOAD_DIR), "uploads", fname)
    async with aiofiles.open(out_path, "wb") as f:
        content = await file.read()
        await f.write(content)
    return JSONResponse({"ok": True, "filename": fname, "content_type": file.content_type})
