from test.utils import create_openai_object
from unittest.mock import MagicMock, patch

from benchllm import Prediction, SemanticEvaluator, Test
from benchllm.data_types import FunctionID


@patch("openai.Completion.create", return_value=create_openai_object("same"))
def test_semantic_passes_if_output_is_equal(completion_mock: MagicMock):
    evaluator = SemanticEvaluator(model="gpt-3")
    evaluator.load(
        [
            Prediction(
                test=Test(input="Who are you?", expected=["Yoda I am."]),
                output="I am Yoda.",
                time_elapsed=0,
                function_id=FunctionID.default(),
            )
        ]
    )
    evaluations = evaluator.run()
    completion_mock.assert_called_once()
    assert evaluations[0].passed


@patch("openai.Completion.create", return_value=create_openai_object("different"))
def test_semantic_fails_if_output_is_unequal(completion_mock: MagicMock):
    evaluator = SemanticEvaluator(model="gpt-3")
    evaluator.load(
        [
            Prediction(
                test=Test(input="What are you?", expected=["Everything"]),
                output="Nothing",
                time_elapsed=0,
                function_id=FunctionID.default(),
            ),
        ]
    )
    evaluations = evaluator.run()
    completion_mock.assert_called_once()
    assert not evaluations[0].passed


@patch("openai.Completion.create", return_value=create_openai_object("same"))
def test_semantic_passes_if_output_is_equal_multiple_workers(completion_mock: MagicMock):
    evaluator = SemanticEvaluator(model="gpt-3", workers=10)
    evaluator.load(
        [
            Prediction(
                test=Test(input="Who are you?", expected=["Yoda I am."]),
                output="I am Yoda.",
                time_elapsed=0,
                function_id=FunctionID.default(),
            )
            for _ in range(100)
        ]
    )
    evaluations = evaluator.run()
    assert completion_mock.call_count == 100
    assert all([evaluation.passed for evaluation in evaluations])
