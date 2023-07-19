import importlib.util
import inspect
import json
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path
from timeit import default_timer as timer
from types import ModuleType
from typing import Any, Callable, Iterator, Optional, Union

import yaml
from pydantic import ValidationError, parse_obj_as

from .data_types import FunctionID, Prediction, Test, TestFunction
from .listener import TesterListener
from .singleton import TestSingleton

CallableTest = Union[TestFunction, Callable[[Any], Any]]


class Tester:
    __test__ = False

    def __init__(self, test_function: Optional[CallableTest] = None, *, retry_count: int = 1) -> None:
        self._tests: dict[FunctionID, list[Test]] = {}
        self._test_functions: dict[FunctionID, TestFunction] = {}
        self._listeners: list[TesterListener] = []
        self._predictions: list[Prediction] = []
        self._retry_count = retry_count

        if test_function:
            self.add_test_function(test_function=test_function)

    def add_listener(self, listener: TesterListener) -> None:
        self._listeners.append(listener)

    def add_tests(self, tests: list[Test], function_id: FunctionID = FunctionID.default()) -> None:
        self._tests.setdefault(function_id, []).extend(tests)

    def add_test(self, test: Test, function_id: FunctionID = FunctionID.default()) -> None:
        self._tests.setdefault(function_id, []).append(test)

    def add_test_function(self, test_function: CallableTest) -> None:
        """Adds a test function to the tester, either a TestFunction or a Callable, Callables will get a default FunctionID"""
        if isinstance(test_function, TestFunction):
            self._test_functions[test_function.function_id] = test_function
            return
        self.add_test_function(TestFunction(function=test_function, function_id=FunctionID.default(), input_type=Any))

    def load_tests(self, suite: Path, function_id: FunctionID) -> None:
        if self._test_functions.get(function_id) is None:
            raise Exception(f"No test function loaded for module {function_id}")

        for test in load_files(suite):
            self.add_test(test, function_id)

    def load_module(self, path: Union[str, Path]) -> None:
        path = Path(path)
        test_singleton = TestSingleton()
        test_singleton.clear()

        import_module_from_file(path)

        if not test_singleton.func or not test_singleton.type or not test_singleton.suite:
            raise NoBenchLLMTestFunction()

        function_id = FunctionID(module_path=path, line_number=inspect.getsourcelines(test_singleton.func)[1])

        self.add_test_function(
            TestFunction(
                function=test_singleton.func,
                function_id=function_id,
                input_type=test_singleton.type,
                suite=test_singleton.suite,
            )
        )

        self.load_tests(test_singleton.suite, function_id)

    def run(self) -> list[Prediction]:
        """Runs each test through the test function and stores the result"""

        self._broadcast_test_run_started()

        if not self._test_functions:
            raise Exception("No function loaded, run load_module() first")

        if not self._tests:
            raise Exception("No tests loaded, run load_tests() first")

        for test_function in self._test_functions.values():
            self._broadcast_test_function_started(test_function)
            for test in self._tests.get(test_function.function_id, []):
                for _ in range(self._retry_count):
                    # Checks that the arity of the function matches the number of inputs.
                    if "__annotations__" in dir(test_function.input_type):
                        if len(test.input) != len(test_function.input_type.__annotations__):
                            raise Exception(
                                f"Your test function needs to have an input parameter annotated with the input type, {test.input}\n\n{test_function.input_type.__annotations__}"
                            )

                    # Now, try to parse the input. If we fail, we will skip the test.
                    try:
                        input = parse_obj_as(test_function.input_type, test.input)
                    except ValidationError:
                        self._broadcast_test_skipped(test, error=True)
                        continue

                    self._broadcast_test_started(test)
                    start = timer()

                    # set up mock functions for the test calls
                    calls_made: dict[str, Any] = {}
                    with setup_mocks(test, calls_made):
                        output = test_function.function(input)

                    end = timer()
                    prediction = Prediction(
                        test=test,
                        output=output,
                        time_elapsed=end - start,
                        function_id=test_function.function_id,
                        calls=calls_made,
                    )
                    self._predictions.append(prediction)
                    self._broadcast_test_ended(prediction)
            self._broadcast_test_function_ended()
        self._broadcast_test_run_ended(self._predictions)
        return self._predictions

    @property
    def predictions(self) -> list[Prediction]:
        return self._predictions

    def tests(self, function_id: FunctionID = FunctionID.default()) -> list[Test]:
        return self._tests.get(function_id, [])

    def _broadcast_test_run_started(self) -> None:
        for listener in self._listeners:
            listener.test_run_started()

    def _broadcast_test_run_ended(self, predications: list[Prediction]) -> None:
        for listener in self._listeners:
            listener.test_run_ended(predications)

    def _broadcast_test_function_started(self, test_function: TestFunction) -> None:
        for listener in self._listeners:
            listener.test_function_started(test_function)

    def _broadcast_test_function_ended(self) -> None:
        for listener in self._listeners:
            listener.test_function_ended()

    def _broadcast_test_started(self, test: Test) -> None:
        for listener in self._listeners:
            listener.test_started(test)

    def _broadcast_test_ended(self, prediction: Prediction) -> None:
        for listener in self._listeners:
            listener.test_ended(prediction)

    def _broadcast_test_skipped(self, test: Test, error: bool = False) -> None:
        for listener in self._listeners:
            listener.test_skipped(test, error)


def load_files(directory: Union[str, Path]) -> list[Test]:
    directory_path = Path(directory)
    tests = []
    for file_path in directory_path.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix not in {".json", ".yml", ".yaml"}:
            continue
        with open(file_path, "r") as file:
            if file_path.suffix == ".json":
                data = json.load(file)
            elif file_path.suffix in {".yml", ".yaml"}:
                data = yaml.safe_load(file)
                try:
                    test = init_test({**data, **{"file_path": file_path}})
                except ValidationError:
                    raise TestLoadException(file_path, "failed to parse your test file") from None
                tests.append(test)
    return tests


def init_test(data: dict) -> Test:
    if "id" not in data:
        data["id"] = str(uuid.uuid4())

        dump_data = {k: v for k, v in data.items() if k != "file_path"}
        with open(data["file_path"], "w") as f:
            if data["file_path"].suffix == ".json":
                json.dump(dump_data, f, indent=2, sort_keys=True)
            elif data["file_path"].suffix in {".yml", ".yaml"}:
                yaml.safe_dump(dump_data, f, indent=2)

    return Test(**data)


class TestLoadException(Exception):
    def __init__(self, file_path: Path, error_message: str) -> None:
        self.file_path = file_path
        self.error_message = error_message

    def __str__(self) -> str:
        return f"Failed to load '{self.file_path}'\n{self.error_message}"


class NoBenchLLMTestFunction(Exception):
    pass


def import_module_from_file(file_path: Path) -> ModuleType:
    # Make sure the file exists.
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Get the module name from the file path (remove the .py extension).
    module_name = file_path.stem

    # Create a module specification from the file path.
    spec = importlib.util.spec_from_file_location(module_name, file_path)

    if not spec or not spec.loader:
        raise Exception(f"Failed to create module specification from {file_path}")

    # Create a new module based on the spec.
    module = importlib.util.module_from_spec(spec)

    if not module:
        raise Exception(f"Failed to load module from {file_path}")

    # Temporarly add the directory of the file to the system path so that the module can import other modules.
    file_module = file_path.resolve().parent
    old_sys_path = sys.path.copy()
    sys.path.append(str(file_module))

    # Execute the module.
    spec.loader.exec_module(module)

    # Restore the system path
    sys.path = old_sys_path

    # Return the module.
    return module


@contextmanager
def setup_mocks(test: Test, calls_made: dict[str, Any]) -> Iterator[None]:
    """Sets up mock functions for the test calls"""
    old_functions = []
    for call in test.calls or []:
        mock_name = call.name
        module_name, function_name = mock_name.rsplit(".", 1)
        # we need to import the module before we can mock the function
        module = importlib.import_module(module_name)
        old_functions.append((module, function_name, getattr(module, function_name)))

        def mock_function(*args: tuple, **kwargs: dict[str, Any]) -> Any:
            assert not args, "Positional arguments are not supported"
            if mock_name not in calls_made:
                calls_made[mock_name] = []
            calls_made[mock_name].append(kwargs)
            return call.returns

        try:
            setattr(module, function_name, mock_function)
        except AttributeError:
            print(f"Function {function_name} doesn't exist in module {module_name}")

    try:
        yield
    finally:
        # restore the old function
        for old_function in old_functions:
            setattr(old_function[0], old_function[1], old_function[2])
