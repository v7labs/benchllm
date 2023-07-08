import signal
from typing import Optional

import typer
from pywebio import session
from pywebio.input import radio
from pywebio.output import put_markdown

from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator


class WebEvaluator(Evaluator):
    def __init__(self) -> None:
        super().__init__(workers=1)

        @session.defer_call
        def on_close() -> None:
            print("shutting down")
            typer.secho(
                f"The evaluation was interrupted. Run bench eval to start again", fg=typer.colors.RED, bold=True
            )
            # sys.exit doesn't work here, so we have to raise a signal to kill the process
            signal.raise_signal(signal.SIGINT)

        put_markdown("# BenchLLM Web Evaluator")

    def evaluate_prediction(self, prediction: Prediction) -> Optional[Evaluator.Match]:
        test_name = prediction.test.file_path or prediction.test.id

        put_markdown(f"## {test_name}")
        put_markdown(f"*Question*: `{prediction.test.input}`")
        put_markdown(f"*Prediction*: `{prediction.output}`")

        table = [["Question:", f"{prediction.test.input}", ""], ["Prediction:", prediction.output], ""]
        label = f"Question: {prediction.test.input}Prediction: {prediction.output}"

        options: list[dict[str, Optional[int | str]]] = [
            {"label": expected, "value": idx} for idx, expected in enumerate(prediction.test.expected)
        ]
        options.append({"label": "None", "value": None, "selected": True})
        answer = radio("Pick the matching answer", options=options, required=True)

        if answer and isinstance(answer, int):
            return Evaluator.Match(prediction=prediction.output, expected=prediction.test.expected[answer])
        else:
            return None
