import tempfile
from pathlib import Path
from unittest.mock import patch

from benchllm import Prediction, StringMatchEvaluator, Test
from benchllm.cache import FileCache
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


def test_file_writes_at_end():
    with patch.object(
        StringMatchEvaluator, "evaluate_prediction", side_effect=StringMatchEvaluator().evaluate_prediction
    ) as mock_method:
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir, "cache.json")
            evaluator = FileCache(StringMatchEvaluator(), cache_path)
            evaluator.load(EXAMPLE_PREDICTIONS)

            evaluations = evaluator.run()
            assert cache_path.exists()
            assert not evaluations[0].passed
            assert evaluations[1].passed
            assert not evaluations[2].passed
            assert mock_method.call_count == 2
            assert evaluator.num_cache_hits == 1
            mock_method.reset_mock()

            # second run will use cache
            evaluator = FileCache(StringMatchEvaluator(), cache_path)
            evaluator.load(EXAMPLE_PREDICTIONS)

            evaluations = evaluator.run()
            assert cache_path.exists()
            assert not evaluations[0].passed
            assert evaluations[1].passed
            assert not evaluations[2].passed
            assert mock_method.call_count == 0
            assert evaluator.num_cache_hits == 3
