from contextlib import asynccontextmanager
import os
from fastapi import FastAPI
import uvicorn
from app.api.minio.minio_routes import router as minio_router
from app.api.ner import ner_routes
from app.api.re import re_routes
from app.db.mongo import init_indexes
from app.api.train_svm.train_routes import router as svm_router
from app.api.train_dl.train_routes import router as dl_router
from app.api.train_logistic.train_routes import router as logistic_router
from app.api.train_rf.train_routes import router as rf_router
from app.api.pipeline.pipeliene_route import router as ie_pipeline_router

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_indexes()
    yield
app = FastAPI(lifespan=lifespan)

    
app.include_router(re_routes.router)
app.include_router(ner_routes.router)
app.include_router(svm_router)
app.include_router(dl_router)
app.include_router(logistic_router)
app.include_router(rf_router)
app.include_router(minio_router)
app.include_router(ie_pipeline_router)

if __name__ == "__main__":
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
