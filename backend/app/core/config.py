import os
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel

load_dotenv(find_dotenv())


class Settings(BaseModel):
    mongo_uri: str = os.getenv("MONGO_URI")
    mongo_db: str = os.getenv("MONGO_DB_NAME")
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI")
    ner_collection_name:str = os.getenv("NER_LABELED_COLLECTION")
    re_collection_name:str = os.getenv("RE_LABELED_COLLECTION")
    
            

settings = Settings()