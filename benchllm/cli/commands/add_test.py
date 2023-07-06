from pathlib import Path
from typing import Optional

import typer
import yaml


def add_test(*, input: str, expected: list[str], name: str, overwrite: bool, suite_path: Optional[Path]) -> None:
    if suite_path is None:
        typer.secho("No default suite was specified.", fg=typer.colors.RED, bold=True)
        raise typer.Exit()

    if not suite_path.exists():
        typer.secho("The specified suite does not exist.", fg=typer.colors.RED, bold=True)
        raise typer.Exit()

    test_path = suite_path / f"{name}.yml"
    if test_path.exists() and not overwrite:
        typer.secho(
            f"The test {test_path} already exists. Use --overwrite to overwrite it.",
            fg=typer.colors.RED,
            bold=True,
        )
        raise typer.Exit()

    with open(test_path, "w") as f:
        yaml.safe_dump({"input": input, "expected": expected}, f)
        typer.secho(f"{test_path} added successfully!", fg=typer.colors.GREEN, bold=True)
