from typing import Optional

from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator
from benchllm.similarity import semantically_similar


class SemanticEvaluator(Evaluator):
    def __init__(self, *, model: str = "gpt-3", workers: int = 1, early_quitting: bool = True):
        super().__init__(workers=workers)
        self.model = model
        self.early_quitting = early_quitting

    def evaluate_prediction(self, prediction: Prediction) -> list[Evaluator.Candidate]:
        candidates = []
        for expected in prediction.test.expected:
            if semantically_similar(expected, prediction.output, model=self.model):
                candidate = Evaluator.Candidate(prediction=prediction.output, expected=expected, score=1.0, passed=True)
                if self.early_quitting:
                    return [candidate]
                else:
                    candidates.append(candidate)
            else:
                candidates.append(
                    Evaluator.Candidate(prediction=prediction.output, expected=expected, score=0.0, passed=False)
                )
        return candidates
