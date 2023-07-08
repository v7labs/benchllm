import json
import tempfile
from pathlib import Path
from test.utils import create_openai_object
from unittest.mock import MagicMock, Mock, call, patch

from benchllm import Prediction, SemanticEvaluator, StringMatchEvaluator, Test
from benchllm.cache import MemoryCache
from benchllm.data_types import FunctionID
from benchllm.evaluator import Evaluator


class NoopEvaluator(Evaluator):
    def evaluate_prediction(self, prediction: Prediction) -> Evaluator.Match:
        return Evaluator.Match(prediction=prediction.output, expected=prediction.output)


def test_evaluator_can_load_prediction_file():
    prediction = {
        "output": "42",
        "test": {"input": "1+1", "expected": ["2"]},
        "time_elapsed": 0,
        "function_id": {"module_path": "test", "line_number": 1},
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        prediction_path = Path(tmpdir, "prediction.json")
        prediction_path.write_bytes(json.dumps(prediction).encode())

        evaluator = NoopEvaluator()
        evaluator.load_prediction_file(prediction_path)

        assert evaluator.predictions[0].output == "42"
        assert evaluator.predictions[0].test.input == "1+1"
        assert evaluator.predictions[0].test.expected == ["2"]
