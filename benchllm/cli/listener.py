import datetime
import json
from pathlib import Path

import typer
from rich import print
from rich.console import Console
from rich.markup import render
from rich.table import Table

from benchllm.data_types import Evaluation, FunctionID, Prediction, Test, TestFunction
from benchllm.listener import EvaluatorListener, TesterListener


class ReportListener(TesterListener, EvaluatorListener):
    def __init__(self, *, output_dir: Path) -> None:
        super().__init__()
        self.output_dir = output_dir

    def test_ended(self, prediction: Prediction) -> None:
        path = self.output_dir / "predictions" / f"{prediction.test.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(json.loads(prediction.json()), f, indent=2)

    def evaluate_prediction_ended(self, evaluation: Evaluation) -> None:
        evaluation_json = json.loads(evaluation.json())
        prediction_json = evaluation_json.pop("prediction")
        prediction_json["evaluation"] = evaluation_json

        path = self.output_dir / "evaluations" / f"{evaluation.prediction.test.id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(prediction_json, f, indent=2)


class RichCliListener(TesterListener, EvaluatorListener):
    def __init__(self, root_dir: Path, *, interactive: bool, test_only: bool = False, eval_only: bool = False) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.interactive = interactive
        self._eval_only = eval_only
        self._test_only = test_only

    def test_run_started(self) -> None:
        print_centered(" Run Tests ")

    def test_run_ended(self, predications: list[Prediction]) -> None:
        if not self._test_only:
            return
        total_test_time = sum(prediction.time_elapsed for prediction in predications) or 0.0
        tmp = f" [green]{len(predications)} tests[/green], in [blue]{format_time(total_test_time)}[/blue] "
        print_centered(tmp)

    def test_function_started(self, test_function: TestFunction) -> None:
        typer.echo(f"{test_function.function_id.relative_str(self.root_dir)} ", nl=False)

    def test_function_ended(self) -> None:
        typer.echo("")

    def test_started(self, test: Test) -> None:
        pass

    def test_ended(self, prediction: Prediction) -> None:
        typer.secho(".", fg=typer.colors.GREEN, bold=True, nl=False)

    def test_skipped(self, test: Test, error: bool = False) -> None:
        if error:
            typer.secho("E", fg=typer.colors.RED, bold=True, nl=False)
        else:
            typer.secho("s", fg=typer.colors.YELLOW, bold=True, nl=False)

    def evaluate_started(self) -> None:
        print_centered(" Evaluate Tests ")

    def evaluate_module_started(self, function_id: FunctionID) -> None:
        typer.echo(f"{function_id.relative_str(self.root_dir)} ", nl=False)

    def evaluate_module_ended(self) -> None:
        typer.echo("")

    def evaluate_prediction_started(self, prediction: Prediction) -> None:
        pass

    def evaluate_prediction_ended(self, evaluation: Evaluation) -> None:
        if self.interactive:
            return

        if evaluation.passed:
            typer.secho(".", fg=typer.colors.GREEN, bold=True, nl=False)
        else:
            typer.secho("F", fg=typer.colors.RED, bold=True, nl=False)

    def evaluate_ended(self, evaluations: list[Evaluation]) -> None:
        failed = [evaluation for evaluation in evaluations if not evaluation.passed]
        total_test_time = (
            0.0 if self._eval_only else sum(evaluation.prediction.time_elapsed for evaluation in evaluations) or 0.0
        )
        total_eval_time = sum(evaluation.eval_time_elapsed for evaluation in evaluations) or 0.0
        if failed:
            print_centered(" Failures ")
            for failure in failed:
                prediction = failure.prediction
                relative_path = prediction.function_id.relative_str(self.root_dir)
                print_centered(f" [red]{relative_path}[/red] :: [red]{prediction.test.file_path}[/red] ", "-")

                console = Console()

                table = Table(show_header=False, show_lines=True)
                table.add_row(f"Input", str(prediction.test.input))
                table.add_row(f"Output", f"[red]{prediction.output}[/red]")
                for i, answer in enumerate(prediction.test.expected):
                    table.add_row(f"Expected #{i+1}", str(answer))
                console.print(table)

        tmp = f" [red]{len(failed)} failed[/red], [green]{len(evaluations) - len(failed)} passed[/green], in [blue]{format_time(total_eval_time + total_test_time)}[/blue] "
        print_centered(tmp)


def print_centered(text: str, sep: str = "=") -> None:
    console = Console()
    terminal_width = console.width

    padding = (terminal_width - len(render(text))) // 2
    print(sep * padding, f"[bold]{text}[/bold]", sep * padding, sep="")


def format_time(seconds: float) -> str:
    delta = datetime.timedelta(seconds=seconds)
    if seconds < 1:
        milliseconds = int(seconds * 1000)
        return f"{milliseconds:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return str(delta)
