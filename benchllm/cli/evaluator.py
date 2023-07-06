import signal

import typer
from pywebio import session
from pywebio.input import actions
from pywebio.output import put_markdown, put_table

from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator


class InteractiveEvaluator(Evaluator):
    def evaluate_prediction(self, prediction: Prediction) -> bool:
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

        while True:
            prompt_string = (
                f'{typer.style("[")}'
                f'{typer.style("y", fg=typer.colors.GREEN, bold=True)}'
                f'{typer.style("/")}'
                f'{typer.style("n", fg=typer.colors.RED, bold=True)}'
                f'{typer.style("]")}'
            )

            response = response = typer.prompt(prompt_string).lower()
            if response == "y":
                return True
            elif response == "n":
                return False
            else:
                typer.secho(
                    'Invalid answer. Please just use "y" to mark the test as correct, and "n" to mark the test as incorrect',
                    fg=typer.colors.RED,
                    bold=True,
                )
                continue


class WebEvaluator(Evaluator):
    def __init__(self):
        super().__init__(workers=1)

        @session.defer_call
        def on_close():
            typer.secho(
                f"The evaluation was interrupted. Run bench eval to start again", fg=typer.colors.RED, bold=True
            )
            # sys.exit doesn't work here, so we have to raise a signal to kill the process
            signal.raise_signal(signal.SIGINT)

        put_markdown("# BenchLLM Web Evaluator")

    def evaluate_prediction(self, prediction: Prediction) -> bool:
        test_name = prediction.test.file_path or prediction.test.id

        put_markdown(f"## {test_name}")
        table = [["Question:", f"{prediction.test.input}"], ["Prediction:", prediction.output]]
        for i, expected in enumerate(prediction.test.expected):
            table.append([f"Expected ({i+1}):", expected])

        put_table(table)

        result = actions(
            label="Does the prediction match any of the answers?",
            buttons=[{"label": "Yes", "value": True}, {"label": "No", "value": False}],
        )

        return bool(result)
