from typing import Optional

from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator
from benchllm.similarity import semantically_similar


class SemanticEvaluator(Evaluator):
    def __init__(self, *, model: str = "gpt-3", workers: int = 1):
        super().__init__(workers=workers)
        self.model = model

    def evaluate_prediction(self, prediction: Prediction) -> Optional[Evaluator.Match]:
        for expected in prediction.test.expected:
            if semantically_similar(expected, prediction.output, model=self.model):
                return Evaluator.Match(prediction=prediction.output, expected=expected)
        return None
