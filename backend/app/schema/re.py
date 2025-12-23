
from typing import Any, Dict, List, Optional
import base64
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

class EntityIn(BaseModel):
    id: Optional[str] = None
    start: int
    end: int
    text: str
    type: str


class NEROutputIn(BaseModel):
    text: str
    entities: List[EntityIn] = Field(default_factory=list)


class VectorizeResponse(BaseModel):
    text: str
    prepared_entities: List[Dict[str, Any]]
    pairs_info: List[Dict[str, Any]]
    x_shape: List[int]
    # 2 cách output:
    # - x_list: list[list[float]] (dễ debug nhưng payload to)
    # - x_b64: base64 bytes (nhẹ hơn)
    x_list: Optional[List[List[float]]] = None
    x_b64: Optional[str] = None
    dtype: str
