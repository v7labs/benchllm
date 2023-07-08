from unittest.mock import patch

from benchllm import Prediction, StringMatchEvaluator, Test
from benchllm.cache import MemoryCache
from benchllm.data_types import FunctionID

EXAMPLE_PREDICTIONS = [
    Prediction(
        test=Test(input="foo", expected=["abc", "def", "ghi"]),
        output="no-match",
        time_elapsed=0,
        function_id=FunctionID.default(),
    ),
    Prediction(
        test=Test(input="foo", expected=["abc", "def", "ghi"]),
        output="def",
        time_elapsed=0,
        function_id=FunctionID.default(),
    ),
    Prediction(
        test=Test(input="foo", expected=["abc", "def", "ghi"]),
        output="no-match",
        time_elapsed=0,
        function_id=FunctionID.default(),
    ),
]

EXAMPLE_PREDICTIONS_ALL_SAME = [
    Prediction(
        test=Test(input="foo", expected=["match"]),
        output="match",
        time_elapsed=0,
        function_id=FunctionID.default(),
    ),
    Prediction(
        test=Test(input="foo", expected=["match"]),
        output="match",
        time_elapsed=0,
        function_id=FunctionID.default(),
    ),
]


def test_memory_cache_will_prevent_calls_to_evaluate_prediction_on_second_run():
    with patch.object(
        StringMatchEvaluator, "evaluate_prediction", side_effect=StringMatchEvaluator().evaluate_prediction
    ) as mock_method:
        evaluator = MemoryCache(StringMatchEvaluator())
        evaluator.load(EXAMPLE_PREDICTIONS)
        evaluations = evaluator.run()
        assert not evaluations[0].passed
        assert evaluations[1].passed
        assert not evaluations[2].passed
        assert mock_method.call_count == 2
        assert evaluator.num_cache_hits == 1
        mock_method.reset_mock()

        # second run will use cache
        evaluations = evaluator.run()
        assert not evaluations[0].passed
        assert evaluations[1].passed
        assert not evaluations[2].passed
        assert evaluator.num_cache_hits == 4
        assert mock_method.call_count == 0


def test_memory_cache_caches_during_run():
    with patch.object(
        StringMatchEvaluator, "evaluate_prediction", side_effect=StringMatchEvaluator().evaluate_prediction
    ) as mock_method:
        evaluator = MemoryCache(StringMatchEvaluator())
        evaluator.load(EXAMPLE_PREDICTIONS_ALL_SAME)

        evaluations = evaluator.run()
        assert evaluations[0].passed
        assert evaluations[1].passed
        assert mock_method.call_count == 1
        assert evaluator.num_cache_hits == 1


def test_memory_cache_supports_numbers():
    with patch.object(
        StringMatchEvaluator, "evaluate_prediction", side_effect=StringMatchEvaluator().evaluate_prediction
    ) as mock_method:
        evaluator = MemoryCache(StringMatchEvaluator())
        evaluator.load(
            [
                Prediction(
                    test=Test(input="foo", expected=["42"]),
                    output="42",
                    time_elapsed=0,
                    function_id=FunctionID.default(),
                ),
                Prediction(
                    test=Test(input="foo", expected=["42"]),
                    output="42",
                    time_elapsed=0,
                    function_id=FunctionID.default(),
                ),
                Prediction(
                    test=Test(input="foo", expected=["42"]),
                    output="24",
                    time_elapsed=0,
                    function_id=FunctionID.default(),
                ),
            ]
        )
        evaluations = evaluator.run()
        assert evaluations[0].passed
        assert evaluations[1].passed
        assert not evaluations[2].passed
        assert mock_method.call_count == 2
        assert evaluator.num_cache_hits == 1
