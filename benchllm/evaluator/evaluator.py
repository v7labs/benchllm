import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from operator import attrgetter
from pathlib import Path
from timeit import default_timer as timer
from typing import Optional

import yaml
from pydantic import BaseModel

from benchllm.data_types import Evaluation, FunctionID, Prediction
from benchllm.input_types import Json
from benchllm.listener import EvaluatorListener


class Evaluator(ABC):
    def __init__(self, workers: int = 1):
        self._predictions: list[Prediction] = []
        self._listeners: list[EvaluatorListener] = []
        self._evaluations: list[Evaluation] = []
        self._workers: int = workers

    class Candidate(BaseModel):
        prediction: Json
        expected: Json
        score: float
        passed: bool

    def add_listener(self, listener: EvaluatorListener) -> None:
        self._listeners.append(listener)

    def load(self, predictions: list[Prediction]) -> None:
        self._predictions.extend(predictions)

    def load_prediction_file(self, path: Path) -> None:
        if path.suffix == ".yml" or path.suffix == ".yaml":
            data = yaml.safe_load(path.read_bytes())
            self.load([Prediction(**data)])
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="UTF-8"))
            self.load([Prediction(**data)])

    def run(self) -> list[Evaluation]:
        self._broadcast_evaluate_started()
        sorted_predictions = sorted(self._predictions, key=lambda x: str(x.function_id))
        grouped_predictions_by_function = [
            (function, list(group)) for function, group in groupby(sorted_predictions, key=attrgetter("function_id"))
        ]
        with ThreadPoolExecutor(max_workers=self._workers) as executor:
            for function, predictions in grouped_predictions_by_function:
                self._broadcast_evaluate_module_started(function)
                for evaluation in executor.map(self._run_evaluation, predictions):
                    self._evaluations.append(evaluation)
                self._broadcast_evaluate_module_ended()
            self._broadcast_evaluate_ended(self._evaluations)
        return self._evaluations

    def _run_evaluation(self, prediction: Prediction) -> Evaluation:
        self._broadcast_evaluate_prediction_started(prediction)
        start = timer()
        candidates = self.evaluate_prediction(prediction)
        end = timer()

        evaluation = Evaluation(
            prediction=prediction,
            passed=any([candidate.passed for candidate in candidates]),
            eval_time_elapsed=end - start,
            score=max([candidate.score for candidate in candidates], default=0.0),
        )
        self._broadcast_evaluate_prediction_ended(evaluation)
        return evaluation

    @property
    def passed(self) -> list[Evaluation]:
        return [evaluation for evaluation in self._evaluations if evaluation.passed]

    @property
    def failed(self) -> list[Evaluation]:
        return [evaluation for evaluation in self._evaluations if not evaluation.passed]

    @property
    def evaluations(self) -> list[Evaluation]:
        return self._evaluations

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def predictions(self) -> list[Prediction]:
        return self._predictions

    @abstractmethod
    def evaluate_prediction(self, prediction: Prediction) -> list[Candidate]:
        """Evaluate a single prediction, return a Match if the prediction matches the expected output."""
        pass

    def max_threads(self) -> int:
        return 1

    def _broadcast_evaluate_started(self) -> None:
        for listener in self._listeners:
            listener.evaluate_started()

    def _broadcast_evaluate_prediction_started(self, prediction: Prediction) -> None:
        for listener in self._listeners:
            listener.evaluate_prediction_started(prediction)

    def _broadcast_evaluate_prediction_ended(self, evaluation: Evaluation) -> None:
        for listener in self._listeners:
            listener.evaluate_prediction_ended(evaluation)

    def _broadcast_evaluate_module_started(self, function_id: FunctionID) -> None:
        for listener in self._listeners:
            listener.evaluate_module_started(function_id)

    def _broadcast_evaluate_module_ended(self) -> None:
        for listener in self._listeners:
            listener.evaluate_module_ended()

    def _broadcast_evaluate_ended(self, evaluations: list[Evaluation]) -> None:
        for listener in self._listeners:
            listener.evaluate_ended(evaluations)
