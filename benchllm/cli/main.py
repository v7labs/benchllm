from pathlib import Path
from typing import Annotated, Optional
from uuid import uuid4

import typer

from benchllm.cli import add_test, evaluate_predictions, list_tests, run_suite
from benchllm.cli.utils import output_dir_factory

app = typer.Typer(add_completion=False)


@app.command(help="Run tests and evaluations.")
def run(
    output_dir: Annotated[
        Path, typer.Option(help="Output directory to save evaluation reports into.", default_factory=output_dir_factory)
    ],
    file_or_dir: Annotated[
        Optional[list[Path]],
        typer.Argument(
            help="Paths to python files or directories implemented @benchllm.test functions.",
            exists=True,
            resolve_path=True,
        ),
    ] = None,
    model: Annotated[str, typer.Option(help="Model to use to run the evaluation.")] = "gpt-3",
    eval: Annotated[bool, typer.Option(help="Run final evaluation.")] = True,
    workers: Annotated[int, typer.Option(help="Number of workers to use to run the evaluation.")] = 1,
    retry_count: Annotated[int, typer.Option(help="Rerun tests to spot flaky output")] = 1,
    evaluator: Annotated[str, typer.Option(help="Evaluator to use to run the evaluation.")] = "semantic",
) -> None:
    if not file_or_dir:
        file_or_dir = [Path.cwd()]

    success = run_suite(
        file_search_paths=file_or_dir,
        model=model,
        output_dir=output_dir,
        workers=workers,
        evaluator_name=evaluator,
        no_eval=not eval,
        retry_count=retry_count,
    )
    if not success:
        raise typer.Exit(code=1)


@app.command(help="Evaluate predictions")
def eval(
    file_or_dir: Annotated[
        list[Path],
        typer.Argument(
            help="Paths to json files or directories containing json files to evaluate.",
            exists=True,
            resolve_path=True,
        ),
    ],
    output_dir: Annotated[
        Path, typer.Option(help="Output directory to save evaluation reports into.", default_factory=output_dir_factory)
    ],
    model: Annotated[str, typer.Option(help="Model to use to run the evaluation.")] = "gpt-3",
    workers: Annotated[int, typer.Option(help="Number of workers to use to run the evaluation.")] = 1,
    evaluator: Annotated[str, typer.Option(help="Evaluator to use to run the evaluation.")] = "semantic",
) -> None:
    success = evaluate_predictions(
        file_or_dir=file_or_dir,
        model=model,
        output_dir=output_dir,
        workers=workers,
        evaluator_name=evaluator,
    )
    if not success:
        raise typer.Exit(code=1)


@app.command(help="Add a new test case to a suite.")
def add(
    suite_path: Annotated[Optional[Path], typer.Argument(help="Test suite directory.")],
    input: Annotated[str, typer.Option(help="Input prompt to send to your model.")],
    expected: Annotated[
        list[str], typer.Option(help="Expected output prompt. You can use this option multiple times.")
    ],
    name: Annotated[
        str,
        typer.Option(
            help="Name of the test case. Generated UUID when not specified.",
            default_factory=uuid4,
        ),
    ],
    overwrite: Annotated[bool, typer.Option(help="Overwrite existing test case.")] = False,
) -> None:
    add_test(input=input, expected=expected, name=name, overwrite=overwrite, suite_path=suite_path)


@app.command(help="List all tests.")
def tests(suite_path: Annotated[Path, typer.Argument(help="Test suite directory.")]) -> None:
    list_tests(suite_path=suite_path)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
