import os
from dotenv import load_dotenv

load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
OR_TOKEN = os.getenv("OR_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING")
LLM = os.getenv("LLM")

DB_USERNAME = os.getenv("DB_USERNAME")
DB_PASSWORD = os.getenv("DB_PASSWORD")

DB_STORAGE_PATH = os.path.join(os.getcwd(), "vector-db")
# COLLECTION_NAME = "bd_laws"
COLLECTION_NAME = "bd_land_law"

INFERENCE_BASE_URL = "https://openrouter.ai/api/v1"
