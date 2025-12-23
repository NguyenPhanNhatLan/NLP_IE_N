# app/api/normalize/schemas.py
from typing import List
from pydantic import BaseModel, Field


class NormalizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text to normalize")


class NormalizeResponse(BaseModel):
    original: str
    normalized: List[str] | str


class ApiResponse(BaseModel):
    status: int
    data: NormalizeResponse