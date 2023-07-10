import json
from typing import Optional

from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator


class StringMatchEvaluator(Evaluator):
    def __init__(self, *, case_sensitive: bool = False, fuzzy: bool = False, workers: int = 1):
        super().__init__(workers=workers)

        self._case_sensitive = case_sensitive
        self._fuzzy = fuzzy

    def match_strings(self, expected: str, output: str) -> bool:
        if not self._case_sensitive:
            expected = expected.lower()
            output = output.lower()

        if self._fuzzy:
            return expected in output or output in expected

        return expected == output

    def evaluate_prediction(self, prediction: Prediction) -> Optional[Evaluator.Match]:
        output = prediction.output
        for expected in prediction.test.expected:
            if self.match_strings(expected, output):
                return Evaluator.Match(prediction=prediction.output, expected=expected)
        return None
