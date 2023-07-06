import inspect
from pathlib import Path
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from .data_types import Evaluation, Prediction, Test  # noqa
from .evaluator import Evaluator, SemanticEvaluator, StringMatchEvaluator  # noqa
from .input_types import ChatInput, SimilarityInput
from .similarity import semantically_similar  # noqa
from .singleton import TestSingleton  # noqa
from .tester import Tester  # noqa

T = TypeVar("T")

__all__ = [
    "test",
    "Tester",
    "Prediction",
    "Test",
    "Evaluation",
    "StringMatchEvaluator",
    "SemanticEvaluator",
    "Evaluator",
]


def test_wrapper(func: Callable[[T], str], type: Type[T], suite: Path) -> None:
    test_singleton = TestSingleton()
    test_singleton.register(func, type=type, suite=suite)


def test(*, suite: str = ".") -> Callable[[Callable[[T], str]], None]:
    def test_decorator(func: Callable[[T], str]) -> None:
        suite_path = Path(suite)
        if not suite_path.is_absolute():
            suite_path = Path(inspect.getfile(func)).parent / suite
        type = func.__annotations__.get("input")
        if type is None:
            raise Exception("Your test function needs to have an input parameter annotated with the input type")
        return test_wrapper(func, type, suite_path)

    return test_decorator
