from fastapi import FastAPI

from retrival import getAnswer
from util.types import ChatRequest


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/chat")
def chat_response(request: ChatRequest):
    res = getAnswer(request)
    return {"message": res}
