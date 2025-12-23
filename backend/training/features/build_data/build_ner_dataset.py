# training/data/build_ner_dataset.py
from training.features.build_data.utils_data import fetch_data_from_mongo, convert_to_bio, split_train_valid, save_ner_tuples_to_jsonl
import os
from dotenv import load_dotenv
load_dotenv()

# python -m training.features.build_data.build_ner_dataset
def build_ner_dataset():
    col = fetch_data_from_mongo(
        mongo_uri=os.getenv("MONGO_URI"),
        db_name=os.getenv("MONGO_DB_NAME"),
        collection=os.getenv("NER_LABELED_COLLECTION"),
    )
    
    cursor= col.find({})
    tasks = cursor.to_list()

    bio_samples = convert_to_bio(tasks)
    train_set, valid_set = split_train_valid(bio_samples, valid_ratio=0.1)
    
    return train_set, valid_set

# train ,valid = build_ner_dataset()
# print(train[0])