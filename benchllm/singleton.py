from pathlib import Path
from typing import Any, Callable, Generic, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class FunctionRegistry(BaseModel, Generic[T]):
    func: Callable[[T], T]
    type: Any
    suite: Path


class TestSingleton(Generic[T]):
    _instance = None
    functions: list[FunctionRegistry[T]] = []

    def __new__(cls: Type["TestSingleton"], *args: list, **kwargs: dict) -> "TestSingleton":
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, func: Callable[[T], T], input_type: Type[T], suite: Path) -> None:
        self.functions.append(FunctionRegistry(func=func, type=input_type, suite=suite))

    def clear(self) -> None:
        self.functions = []
