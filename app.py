from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from responses import getAnswer
from util.types import ChatFeedback, ChatRequest


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/chat/feedback")
def chat_feedback(request: ChatFeedback):
    print(f"Feedback received: {request}")

    return Response(
        content=f"Feedback for message ID {request.message_id} received with rating {request.rating}.",
        status_code=200
    )


@app.post("/chat")
async def chat_response(request: ChatRequest):
    # res = await getAnswer(request)
    # return res
    return StreamingResponse(getAnswer(request), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
