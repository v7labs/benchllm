from benchllm import Prediction, StringMatchEvaluator, Test
from benchllm.data_types import FunctionID


def test_string_match_passes_if_output_is_equal_to_expected():
    evaluator = StringMatchEvaluator()
    evaluator.load(
        [
            Prediction(
                test=Test(input="foo", expected=["bar"]), output="42", time_elapsed=0, function_id=FunctionID.default()
            ),
            Prediction(
                test=Test(input="foo", expected=["42"]), output="42", time_elapsed=0, function_id=FunctionID.default()
            ),
            Prediction(
                test=Test(input="foo", expected=["BAR"]), output="bar", time_elapsed=0, function_id=FunctionID.default()
            ),
        ]
    )
    evaluations = evaluator.run()
    assert not evaluations[0].passed
    assert evaluations[1].passed
    assert evaluations[2].passed


def test_string_match_passes_if_output_is_equal_to_expected_case_sensitive():
    evaluator = StringMatchEvaluator(case_sensitive=True)
    evaluator.load(
        [
            Prediction(
                test=Test(input="foo", expected=["BAR"]), output="BAR", time_elapsed=0, function_id=FunctionID.default()
            ),
            Prediction(
                test=Test(input="foo", expected=["BAR"]), output="bar", time_elapsed=0, function_id=FunctionID.default()
            ),
        ]
    )
    evaluations = evaluator.run()
    assert evaluations[0].passed
    assert not evaluations[1].passed


def test_string_match_passes_if_output_is_equal_to_expected_fuzzy():
    evaluator = StringMatchEvaluator(fuzzy=True)
    evaluator.load(
        [
            Prediction(
                test=Test(input="foo", expected=["abc def ghi"]),
                output="def",
                time_elapsed=0,
                function_id=FunctionID.default(),
            ),
            Prediction(
                test=Test(input="foo", expected=["abc def ghi"]),
                output="adg",
                time_elapsed=0,
                function_id=FunctionID.default(),
            ),
        ]
    )
    evaluations = evaluator.run()
    assert evaluations[0].passed
    assert not evaluations[1].passed
