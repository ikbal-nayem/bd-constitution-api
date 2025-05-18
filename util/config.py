import os
from dotenv import load_dotenv

load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
OR_TOKEN = os.getenv("OR_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING")
LLM = os.getenv("LLM")

DB_STORAGE_PATH = "vector-db"
COLLECTION_NAME = "bd_laws"

INFERENCE_BASE_URL = "https://openrouter.ai/api/v1"
