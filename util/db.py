import datetime
from pymongo import MongoClient
import urllib

from util.config import DB_PASSWORD, DB_USERNAME
from util.types import ConversationHistory

connection_string = f"mongodb://{DB_USERNAME}:{urllib.parse.quote(DB_PASSWORD)}@localhost:27017/?directConnection=true&authSource=bdLaw"

mongo_client = MongoClient(connection_string)

print("Connected to MongoDB: ", mongo_client)

db = mongo_client["bdLaw"]

qa_collection = db['conversations']


def insertHistory(history: ConversationHistory):
    try:
        history.created_on = datetime.datetime.today().isoformat()
        qa_collection.insert_one(history.model_dump())
    except Exception as e:
        print(f"\nError inserting document: {e}")


def setFeedback(message_id: str, rating: str, feedback: str, suggested_answer: str = None):
    try:
        qa_collection.update_one(
            {"message_id": message_id},
            {"$set": {"rating": rating, "feedback": feedback,
                      "suggested_answer": suggested_answer}}
        )
        return True
    except Exception as e:
        print(f"\nError updating document: {e}")
        return False
