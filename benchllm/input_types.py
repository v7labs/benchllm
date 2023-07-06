import json
from typing import TypedDict

from pydantic import BaseModel


class ChatInputItem(TypedDict):
    role: str
    content: str


ChatInput = list[ChatInputItem]


class SimilarityInput(BaseModel):
    prompt_1: str
    prompt_2: str
