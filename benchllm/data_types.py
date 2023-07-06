from pathlib import Path
from typing import Any, Callable, Generic, Optional, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field


class Test(BaseModel):
    __test__ = False
    id: str = Field(default_factory=lambda: str(uuid4()))
    input: Any
    expected: list[str]
    file_path: Optional[Path] = None


class FunctionID(BaseModel):
    module_path: Path
    line_number: int

    def __hash__(self) -> int:
        return hash((self.module_path, self.line_number))

    def __str__(self) -> str:
        return f"{self.module_path}:{self.line_number}"

    def relative_str(self, root_dir: Path) -> str:
        try:
            return str(FunctionID(module_path=self.module_path.relative_to(root_dir), line_number=self.line_number))
        except ValueError:
            # we can't be sure that the module_path loaded from json files is relative to the root_dir
            return str(FunctionID(module_path=self.module_path, line_number=self.line_number))

    @staticmethod
    def default() -> "FunctionID":
        return FunctionID(module_path=Path(""), line_number=0)


class Prediction(BaseModel):
    test: Test
    output: str
    time_elapsed: float
    function_id: FunctionID


class Evaluation(BaseModel):
    prediction: Prediction
    passed: bool
    eval_time_elapsed: float


T = TypeVar("T")


class TestFunction(BaseModel, Generic[T]):
    function: Callable[[T], Any]
    function_id: FunctionID
    input_type: T
    suite: Optional[Path] = None
