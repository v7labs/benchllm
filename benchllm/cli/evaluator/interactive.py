from typing import Optional

import click
import typer

from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator


class InteractiveEvaluator(Evaluator):
    def evaluate_prediction(self, prediction: Prediction) -> list[Evaluator.Candidate]:
        header = (
            f'{typer.style("Does ", bold=True)}'
            f"{typer.style(prediction.output, fg=typer.colors.BRIGHT_BLUE, bold=True)}"
            f'{typer.style(" match any of the following expected prompts?", bold=True)}'
        )
        typer.echo("")
        typer.echo(header)

        for i, expected in enumerate(prediction.test.expected, start=1):
            typer.secho(f"{i}. ", fg=typer.colors.BRIGHT_BLUE, bold=True, nl=False)
            typer.secho(expected, bold=True)

        options = [str(idx) for idx, _ in enumerate(prediction.test.expected, start=1)] + ["n"]

        prompt_string = f"[{typer.style('matching number', fg=typer.colors.GREEN, bold=True)} or {typer.style('n', fg=typer.colors.RED, bold=True)}]"
        click_choice = click.Choice(options)
        response = typer.prompt(prompt_string, default="n", type=click_choice, show_choices=False).lower()
        if response == "n":
            return [
                Evaluator.Candidate(prediction=prediction.output, expected=expected, score=0.0, passed=False)
                for expected in prediction.test.expected
            ]
        return [
            Evaluator.Candidate(
                prediction=prediction.output,
                expected=prediction.test.expected[int(response) - 1],
                score=1.0,
                passed=True,
            )
        ]
