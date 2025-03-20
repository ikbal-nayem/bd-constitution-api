from fastapi import FastAPI
from pydantic import BaseModel

from retrival import getAnswer


class Message(BaseModel):
    role: str
    content: str
    id: str = None


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 100
    temperature: float = 0.5


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/chat")
def chat_response(request: ChatRequest):
    res = getAnswer(request.messages[-1].content)
    return {"message": res}
