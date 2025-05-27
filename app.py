from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from retrival import getAnswer
from util.types import ChatRequest


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/chat")
async def chat_response(request: ChatRequest):
    # res = await getAnswer(request)
    return StreamingResponse(getAnswer(request), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
