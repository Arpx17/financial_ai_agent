
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import os
from dotenv import load_dotenv
import os
import json
import logging
from pymongo import MongoClient
from pymongo.server_api import ServerApi

load_dotenv("financial_ai_agent/.env")

# MongoDB connection
mongo_key = os.getenv("MONGO_DB_KEY")
uri = f"mongodb+srv://arpx10:{mongo_key}@arpx-cluster.hfwgifb.mongodb.net/?appName=arpx-cluster"
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
db = client["fs_db_mongo"]  
collection = db["financial_scrapped_data"]  


# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




def store_in_mongodb_cloud(doc):
    if "url" not in doc:
        logger.warning("❌ Document skipped (no 'url' key): %s", doc)
        return

    if not doc.get("content"):  # Handles empty strings, None, empty lists, etc.
        logger.warning("❌ Document skipped (empty 'content'): %s", doc.get("url"))
        return

    existing = collection.find_one({"url": doc["url"]})
    if not existing:
        collection.insert_one(doc)
        logger.info(f"✅ Stored: {doc.get('title', doc['url'])}")
    else:
        logger.info(f"⚠️ Already exists: {doc.get('title', doc['url'])}")
def load_json_files(folder_path="json"):
    """Load all JSON files from a folder and store them into MongoDB."""
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            store_in_mongodb_cloud(item)
                    elif isinstance(data, dict):
                        store_in_mongodb_cloud(data)
                    else:
                        logger.warning(f"❌ Unsupported JSON format in file: {filename}")
            except Exception as e:
                logger.error(f"❌ Error processing {filename}: {e}")

# Example usage
if __name__ == "__main__":
    from pathlib import Path

    BASE_DIR = Path(__file__).resolve().parents[2]  # Go two levels up from json_upload.py
    json_folder = BASE_DIR / "data/json/"

    load_json_files(json_folder)
