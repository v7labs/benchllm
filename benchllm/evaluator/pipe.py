from benchllm.data_types import Prediction
from benchllm.evaluator import Evaluator


class PipeEvaluator(Evaluator):
    def __init__(self, evaluators: list[Evaluator], workers: int = 1):
        super().__init__(workers=workers)
        self._evaluators: list[Evaluator] = evaluators

    def evaluate_prediction(self, prediction: Prediction) -> list[Evaluator.Candidate]:
        candidates = []
        for evaluator in self._evaluators:
            candidates = evaluator.evaluate_prediction(prediction)
            # only return passed candidates so negative predication don't get cached
            # since they might have passed further down the chain
            passed_candidates = [candidate for candidate in candidates if candidate.passed]
            if passed_candidates:
                return passed_candidates
        return candidates
