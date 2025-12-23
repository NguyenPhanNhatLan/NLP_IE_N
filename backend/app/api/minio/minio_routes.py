# app/api/minio_upload/routes.py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile
import os

from app.db.minio import upload_file
from app.schema.minio import UploadToMinioResponse

router = APIRouter(prefix="/storage/minio", tags=["minio"])


@router.post(
    "/upload",
    response_model=UploadToMinioResponse,
    summary="Upload file lên MinIO",
)
async def upload_to_minio(
    file: UploadFile = File(...),
    object_name: str = Form(..., description="Object path trong bucket (vd: datasets/re/train.jsonl)"),
    bucket: str = Form("nlp-ie"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Empty filename")
    try:
        suffix = os.path.splitext(file.filename)[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        upload_file(
            file_path=tmp_path,
            object_name=object_name,
            bucket=bucket,
        )

        size_bytes = os.path.getsize(tmp_path)

        return UploadToMinioResponse(
            bucket=bucket,
            object_name=object_name,
            size_bytes=size_bytes,
            message="Upload successful",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if "tmp_path" in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
