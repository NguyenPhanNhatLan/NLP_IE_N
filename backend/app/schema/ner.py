# app/api/ner/schemas.py
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field



class NerEntity(BaseModel):
    text: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1, description="e.g., NAME, INCI, ORIGIN, BENEFITS, SKIN_CONCERNS")
    span: Dict[str, Any] = Field(..., description="e.g., {'seg':'A','start':0,'end':9}")


class NerOutputMock(BaseModel):
    raw_pred: Any = None
    tokens: Any = None
    tags: Any = None
    entities: List[NerEntity] = Field(default_factory=list)


class NerRunRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    model_version: str = Field(..., min_length=1)


class NerRunResponse(BaseModel):
    draft_id: str
    model: Dict[str, str]
    input: Dict[str, str]
    ner_output: NerOutputMock


class NerOverrideRequest(BaseModel):
    """
    Bạn có thể:
    - gửi full ner_output (mock) đã sửa
    hoặc
    - chỉ gửi entities_suggested để patch entities (đơn giản cho UI)
    """
    ner_output: Optional[NerOutputMock] = None
    entities_suggested: Optional[List[NerEntity]] = None


class NerOverrideResponse(BaseModel):
    draft_id: str
    model: Dict[str, str]
    input: Dict[str, str]
    ner_output: NerOutputMock
    re_inputs: List[Dict[str, Any]]
