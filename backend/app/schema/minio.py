from pydantic import BaseModel, Field


class UploadToMinioResponse(BaseModel):
    bucket: str
    object_name: str
    size_bytes: int
    message: str
