import re
from typing import List
import unicodedata

def normalized(text: str) -> List[str]:
    
    if not text or not text.strip():
        return []

    text = unicodedata.normalize("NFC", text)
    text = text.strip()
    text = re.sub(r"[\x00-\x09\x0b-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)

    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]

    return sentences