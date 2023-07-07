import json
from typing import TypedDict, Union

from pydantic import BaseModel

Json = Union[str, bool, list, dict]


class ChatInputItem(TypedDict):
    role: str
    content: str


ChatInput = list[ChatInputItem]


class SimilarityInput(BaseModel):
    prompt_1: str
    prompt_2: str
