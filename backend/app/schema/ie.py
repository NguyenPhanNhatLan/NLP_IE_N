from typing import Any, Dict, List, Optional, Literal, Union
from pydantic import BaseModel, ConfigDict, Field

# ===== Entities / Relations =====
class Span(BaseModel):
    start: int
    end: int
    label: str
    text: str
    id: Optional[str] = None

# class IEPipelineRequest(BaseModel):
#     text: str = Field(..., min_length=1, description="Input text for IE pipeline")

class IeInput(BaseModel):
    model_name: str = 're_svm_phober'
    model_verstion: str = '1'
    loader: str = 'pyfunc'
    text: str 
    

class IeFeInput(BaseModel):
    model_name: str = 're_svm_phober'
    model_version: str = '1'
    text: str 
    entities_output: List[Any]
class Relation(BaseModel):
    head_id: str
    tail_id: str
    relation: str
    confidence: float = 1.0
    meta: Dict[str, Any] = Field(default_factory=dict)

# ===== Pipeline IO =====
class IEPipelineRequest(BaseModel):
    text: str = Field(..., min_length=1)
    re_model: str = Field(..., description="Tên RE model (svm/logistic/rf/lstm/transformer...)")
    model_version: int =1
    
    
class Section(BaseModel):
    relation: str
    title: str
    items: List[str]

class Subject(BaseModel):
    subject_text: str
    subject_type: str
    sections: List[Section]

class SentenceOutput(BaseModel):
    sentence_id: int
    sentence: str
    subjects: List[Subject]
