from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse

from responses import getAnswer
from util.db import setFeedback
from util.types import ChatFeedback, ChatRequest


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.post("/chat/feedback")
def chat_feedback(request: ChatFeedback):
    print(f"Feedback received: ", request)
    isSuccess = setFeedback(
        message_id=request.message_id,
        feedback=request.feedback,
        rating=request.rating,
        suggested_answer=request.suggested_answer
    )

    return Response(
        content="Feedback recorded successfully." if isSuccess else "Failed to record feedback.",
        status_code=200 if isSuccess else 500
    )


@app.post("/chat")
async def chat_response(request: ChatRequest):
    # res = await getAnswer(request)
    # return res
    return StreamingResponse(getAnswer(request), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive"})
