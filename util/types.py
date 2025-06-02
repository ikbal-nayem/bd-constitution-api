from typing import Literal, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str
    id: Optional[str] = None


class ChatRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 100
    temperature: float = 0.5
    stream: bool = False

class ChatFeedback(BaseModel):
    message_id: str
    rating: Literal['good', 'bad']
    suggested_answer: Optional[str] = None
    feedback: Optional[str] = None