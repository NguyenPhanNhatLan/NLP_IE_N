import json
import logging
import os
import uuid
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from pymongo import UpdateOne
from app.api.deps import get_db


router = APIRouter(
    tags=["NER API"]
)
COLLECTION_NAME = os.getenv("NER_LABELED_COLLECTION", "ner_labeled")
    
@router.post("/ner/insert-json", status_code=status.HTTP_201_CREATED)
async def upload_json_to_mongo(
    file: UploadFile = File(...),
    db=Depends(get_db)
):
    if file.content_type not in ("application/json", "text/json"):
        raise HTTPException(status_code=400, detail="File must be JSON format")

    try:
        raw = await file.read()
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="JSON must be an array of objects")

    collection = db[COLLECTION_NAME]
    operations = []

    for item in data:
        if not isinstance(item, dict):
            continue

        item.pop("_id", None)

        if item.get("source_id") is None:
            item.pop("source_id", None)
            
        source_id = item.get("source_id")
        if not source_id:
            source_id = str(uuid.uuid4())

        item["source_id"] = source_id
        operations.append(
            UpdateOne(
                {"source_id": source_id},
                {"$set": item},
                upsert=True
            )
        )

    if not operations:
        return {
            "success": True,
            "inserted": 0,
            "updated": 0
        }

    try:
        result = await collection.bulk_write(operations, ordered=False)
    except Exception:
        logging.exception("MongoDB bulk_write failed")
        raise HTTPException(status_code=500, detail="Database write failed")

    return {
        "success": True,
        "inserted": result.upserted_count,
        "updated": result.modified_count,
        "matched": result.matched_count
    }

    
@router.delete("/ner/delete-by-ids", status_code=status.HTTP_200_OK)
async def delete_by_ids_mongo(file: UploadFile = File(...), db=Depends(get_db)):
    if file.content_type not in ("application/json", "text/json"):
        raise HTTPException(status_code=400, detail="File must be JSON format")
    raw = await file.read()
    
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    collection = db[COLLECTION_NAME]
    delete_ids = [d['id'] for d in data]
    result = await collection.delete_many({
        "id": {"$in": delete_ids}
    })
    try:
        return JSONResponse(
            status_code=200,
            content={
                "success": "true",
                "data": len(delete_ids),
                "deleted": result.deleted_count
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logging.exception(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail="Delete operation failed")
    
    
@router.get("/ner/get-ids", status_code=status.HTTP_200_OK)
async def get_all_ids_mongo(db=Depends(get_db)):
    
    try:
        collection = db[COLLECTION_NAME]
        ids = []
        async for doc in collection.find():
            ids.append(doc.get("id"))
        if not ids:
            raise HTTPException(status_code=401, detail="ids not found")
        return JSONResponse(
            status_code=200,
            content={
                "success": "true",
                "data": ids
            }
        )
    except Exception as e:
        logging.exception("Internal Server Error")
        raise HTTPException(status_code=500, detail="Internal Server Error")

