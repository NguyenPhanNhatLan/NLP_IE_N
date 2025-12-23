import base64
import numpy as np

from fastapi import APIRouter, HTTPException
from app.schema.ie import IeFeInput, IeInput
from app.schema.normalize import NormalizeRequest, NormalizeResponse
from app.schema.re import NEROutputIn, VectorizeResponse

from app.services.ie_pipeline_service import run_ie_pipeline, run_ie_pipeline_from_fe
from app.services.ner_service import run_only_ner
from app.services.re_service import prepare_entities_for_re, vectorize
from app.services.text_service import normalized

router = APIRouter(prefix="/ie", tags=["IE Pipeline"])

def _to_b64_numpy(x: np.ndarray):
    raw = x.tobytes(order="C")
    return base64.b64encode(raw).decode("utf-8")

@router.post("/normalize", description="Tiền xử lý cơ bản")
def normalized_text(payload: NormalizeRequest):
    try:
        print(payload.text)
        text = normalized(payload.text)
        return {
            "status":200,
            "message": NormalizeResponse(
                original=payload.text,
                normalized=text
            )
        }
    except Exception as e:
        # log e nếu cần
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/vectorize", response_model=VectorizeResponse)
def re_vectorize(payload: NEROutputIn, output: str = "list"):
    try:
        ner_output = payload.model_dump()
        prepared = prepare_entities_for_re(ner_output)

        X, pairs_info = vectorize(
            prepared,
            ner_output["text"]
        )

        resp = {
            "text": ner_output["text"],
            "prepared_entities": prepared,
            "pairs_info": pairs_info,
            "x_shape": list(X.shape),
            "dtype": str(X.dtype),
            "x_list": None,
            "x_b64": None,
        }

        if output == "list":
            resp["x_list"] = X.tolist() if X.size else []
        elif output == "b64":
            resp["x_b64"] = _to_b64_numpy(X) if X.size else ""
        else:
            raise HTTPException(status_code=400, detail="output must be 'list' or 'b64'")

        return resp

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"vectorize failed: {e}")

@router.post("/run")
def ie_run(req: IeInput):
    try:
        data, model_name, model_version, cards = run_ie_pipeline(req)
        return {
                "status": 200,
                "step": "result",
                "message": "OK",
                "data" : {
                    "model_name": model_name,
                    "model_version": model_version,
                    "raw": data,
                    "ui" : cards
                    
                }
        }
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing field: {str(e)}")
    except Exception as e:
        # bạn có thể log e ở đây
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/ner")
def ner_endpoint(payload: dict):
    try:
        entities_output, ui = run_only_ner(payload)
        return {
            "status": 200,
            "entities": entities_output,
            "ui": ui
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"NER processing failed: {str(e)}"
        )


@router.post("/from-fe")
def ie_from_fe(payload: IeFeInput):
    try:
        result = run_ie_pipeline_from_fe(payload
        )
        return {
            "status": 200,
            "message": "OK",
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
