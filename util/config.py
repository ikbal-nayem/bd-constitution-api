import os

HF_TOKEN = os.getenv("HF_TOKEN")
EMBEDDING_MODEL = os.getenv("EMBEDDING")
LLM = os.getenv("LLM")

DB_STORAGE_PATH = "vector-db"