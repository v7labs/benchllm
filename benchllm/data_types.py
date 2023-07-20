from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field


class TestCall(BaseModel):
    __test__ = False
    name: str
    arguments: dict[str, Any]
    returns: Any


class Test(BaseModel):
    __test__ = False
    id: str = Field(default_factory=lambda: str(uuid4()))
    input: Any
    expected: list[str]
    file_path: Optional[Path] = None
    calls: Optional[list[TestCall]] = None


class FunctionID(BaseModel):
    module_path: Path
    line_number: int
    name: str

    def __hash__(self) -> int:
        return hash((self.module_path, self.line_number))

    def __str__(self) -> str:
        return f"{self.module_path}:{self.line_number} ({self.name})"

    def relative_str(self, root_dir: Path) -> str:
        try:
            return str(
                FunctionID(
                    module_path=self.module_path.relative_to(root_dir), line_number=self.line_number, name=self.name
                )
            )
        except ValueError:
            # we can't be sure that the module_path loaded from json files is relative to the root_dir
            return str(FunctionID(module_path=self.module_path, line_number=self.line_number, name=self.name))

    @staticmethod
    def default() -> "FunctionID":
        return FunctionID(module_path=Path(""), line_number=0, name="default")


class Prediction(BaseModel):
    test: Test
    output: str
    time_elapsed: float
    function_id: FunctionID
    calls: dict[str, list[dict[str, Any]]] = {}


class CallErrorType(str, Enum):
    MISSING_FUNCTION = "Missing function"
    MISSING_ARGUMENT = "Missing argument"
    VALUE_MISMATCH = "Value mismatch"


class CallError(BaseModel):
    function_name: str
    argument_name: Optional[str] = None
    expected_value: Optional[Any] = None
    actual_value: Optional[Any] = None
    error_type: CallErrorType


class Evaluation(BaseModel):
    prediction: Prediction
    passed: bool
    eval_time_elapsed: float
    score: float


T = TypeVar("T")


class TestFunction(BaseModel, Generic[T]):
    function: Callable[[T], Any]
    function_id: FunctionID
    input_type: T
    suite: Optional[Path] = None
