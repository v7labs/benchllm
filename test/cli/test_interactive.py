from unittest import mock

import typer

from benchllm.cli.evaluator import InteractiveEvaluator
from benchllm.data_types import FunctionID, Prediction, Test

TEST_PREDICTION = [
    Prediction(
        test=Test(input="Who are you?", expected=["Yoda I am.", "Yoda"]),
        output="I am Yoda.",
        time_elapsed=0,
        function_id=FunctionID.default(),
    )
]


def test_interactive_press_y_passes():
    evalautor = InteractiveEvaluator()
    evalautor.load(TEST_PREDICTION)
    with mock.patch.object(typer, "prompt", lambda *args, **kwargs: "1"):
        result = evalautor.run()
    assert result[0].passed

    with mock.patch.object(typer, "prompt", lambda *args, **kwargs: "2"):
        result = evalautor.run()
    assert result[0].passed


def test_interactive_press_n_fails():
    evalautor = InteractiveEvaluator()
    evalautor.load(TEST_PREDICTION)
    with mock.patch.object(typer, "prompt", lambda *args, **kwargs: "n"):
        result = evalautor.run()
    assert not result[0].passed
