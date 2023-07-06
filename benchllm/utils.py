import ast
from pathlib import Path


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
