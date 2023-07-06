from pathlib import Path
from typing import Any, Callable, Generic, Optional, Type, TypeVar

T = TypeVar("T")


class TestSingleton(Generic[T]):
    _instance = None
    func: Optional[Callable[[T], T]] = None
    type: Optional[Type[T]] = None
    suite: Optional[Path] = None

    def __new__(cls: Type["TestSingleton"], *args: list, **kwargs: dict) -> "TestSingleton":
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def register(self, func: Callable[[T], T], type: Type[T], suite: Path) -> None:
        self.func = func
        self.type = type
        self.suite = suite

    def clear(self) -> None:
        self.func = None
        self.type = None
        self.suite = None
