from typing import Literal, Optional
from pydantic import BaseModel
from sympy import Li


class Message(BaseModel):
    id: str
    role: str
    content: str
    createdAt: str


class ChatRequest(BaseModel):
    id: str
    act: Optional[Literal['LAND', 'DEFAULT']]
    messages: list[Message]
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