from fastapi import FastAPI

from retrival import getAnswer

# Create a FastAPI application
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}


@app.get("/chat")
def chat_response():
    res = getAnswer("What is the capital of Bangladesh?")
    return {"message": res}
