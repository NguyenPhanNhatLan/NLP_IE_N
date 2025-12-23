from functools import lru_cache
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "vinai/phobert-base-v2"

@lru_cache(maxsize=1)
def get_phobert(device: str = "cpu"):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    enc = AutoModel.from_pretrained(MODEL_NAME).to(device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad = False
    return tok, enc
