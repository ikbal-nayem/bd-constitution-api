from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    id: str = None


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 100
    temperature: float = 0.5
    stream: bool = False