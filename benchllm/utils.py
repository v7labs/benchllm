import ast
import json
from pathlib import Path

import yaml

from benchllm.data_types import CallError, CallErrorType, Prediction


class DecoratorFinder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.has_decorator: bool = False
        self.module_aliases: list[str] = []

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "benchllm":
                self.module_aliases.append(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute):
                decorator = decorator.func
                if decorator.attr == "test":
                    if isinstance(decorator.value, ast.Name) and decorator.value.id in self.module_aliases:
                        self.has_decorator = True
        self.generic_visit(node)


def check_file(path: Path) -> bool:
    with open(path, "r", encoding="utf8") as f:
        tree = ast.parse(f.read())
        finder = DecoratorFinder()
        finder.visit(tree)
        return finder.has_decorator


def find_files(paths: list[Path]) -> list[Path]:
    python_files = set()
    for path in paths:
        if path.suffix == ".py" and not path.name.startswith("."):
            if check_file(path):
                python_files.add(path)
        else:
            for file in path.rglob("*.py"):
                if file.name.startswith("."):
                    continue
                if check_file(file):
                    python_files.add(file)
    return list(python_files)


def find_json_yml_files(paths: list[Path]) -> list[Path]:
    files = []
    for path in paths:
        if path.is_file():
            if path.suffix in (".yml", ".json", ".yaml"):
                files.append(path)
            else:
                continue
        else:
            for file in path.rglob("*"):
                if file.suffix in (".yml", ".json", ".yaml"):
                    files.append(file)
    return list(set(files))


def load_prediction_files(paths: list[Path]) -> list[Prediction]:
    predictions = []
    for path in paths:
        for file_path in path.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in {".json", ".yml", ".yaml"}:
                continue
            with open(file_path, "r") as file:
                if file_path.suffix == ".json":
                    data = json.load(file)
                    predictions.append(Prediction(**data))
                elif file_path.suffix in {".yml", ".yaml"}:
                    data = yaml.safe_load(file)
                    predictions.append(Prediction(**data))
    return predictions


def collect_call_errors(prediction: Prediction) -> list[CallError]:
    """Assert that the calls in the prediction match the expected calls."""
    if prediction.test.calls is None:
        return []
    errors = []
    lookup = {call.name: call for call in prediction.test.calls}

    for function_name, invocations in prediction.calls.items():
        call = lookup[function_name]
        if not call:
            errors.append(CallError(function_name=function_name, error_type=CallErrorType.MISSING_FUNCTION))
            continue

        for arguments in invocations:
            for argument_name, argument_value in call.arguments.items():
                if argument_name not in arguments:
                    errors.append(
                        CallError(
                            function_name=function_name,
                            argument_name=argument_name,
                            error_type=CallErrorType.MISSING_ARGUMENT,
                        )
                    )
            for argument_name, argument_value in arguments.items():
                if argument_name in call.arguments and argument_value != call.arguments[argument_name]:
                    errors.append(
                        CallError(
                            function_name=function_name,
                            argument_name=argument_name,
                            expected_value=call.arguments[argument_name],
                            actual_value=argument_value,
                            error_type=CallErrorType.VALUE_MISMATCH,
                        )
                    )
    return errors
