from typing import Literal, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    id: str
    messages: list[Message]
    max_tokens: int = 100
    temperature: float = 0.5

class ChatFeedback(BaseModel):
    message_id: str
    rating: Literal['good', 'bad']
    suggested_answer: Optional[str] = None
    feedback: Optional[str] = None


class ConversationHistory(BaseModel):
    message_id: str
    question: str
    answer: str
    language: Literal['bn', 'en']
    user_id: Optional[str] = None
    rating: Optional[Literal['good', 'bad']] = None
    feedback: Optional[str] = None
    suggested_answer: Optional[str] = None
    created_on: str = ''