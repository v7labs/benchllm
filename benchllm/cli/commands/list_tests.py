import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table


def list_tests(*, suite_path: Optional[Path]) -> None:
    if suite_path is None:
        typer.secho("No default suite was specified.", fg=typer.colors.RED, bold=True)
        raise typer.Exit()

    if not suite_path.exists():
        typer.secho("The specified suite does not exist.", fg=typer.colors.RED, bold=True)
        raise typer.Exit()

    console = Console()

    table = Table()
    table.add_column("Input")
    table.add_column("No.", justify="right")
    table.add_column("Expected")

    test_paths = list(suite_path.glob("*.yml"))
    for test_path in test_paths:
        with open(test_path, "r") as f:
            example = yaml.safe_load(f)
            for i, expected in enumerate(example["expected"], 1):
                if i == 1:
                    input = json.dumps(example["input"])
                else:
                    input = ""
                table.add_row(input, str(i), json.dumps(expected))
            table.add_section()

    if test_paths:
        console.print(table)
    else:
        typer.secho("No tests found in the specified suite directory.", fg=typer.colors.RED, bold=True)
