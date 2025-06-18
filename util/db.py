import datetime
from pymongo import MongoClient

from util.config import DB_PASSWORD, DB_USERNAME
from util.types import ConversationHistory

connection_string = f"mongodb://{DB_USERNAME}:{DB_PASSWORD}@localhost:27017/"

mongo_client = MongoClient("mongodb://localhost:27017/")

db = mongo_client["bdLaw"]

qa_collection = db['conversations']

def insertHistory(history: ConversationHistory):
  try:
    history.created_on = datetime.datetime.today().isoformat()
    qa_collection.insert_one(history.model_dump())
  except Exception as e:
    print(f"Error inserting document: {e}")
