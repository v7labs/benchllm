from .data_types import Evaluation, FunctionID, Prediction, Test, TestFunction


class TesterListener:
    def test_run_started(self) -> None:
        pass

    def test_run_ended(self, predications: list[Prediction]) -> None:
        pass

    def test_function_started(self, test_function: TestFunction) -> None:
        pass

    def test_function_ended(self) -> None:
        pass

    def test_started(self, test: Test) -> None:
        pass

    def test_ended(self, prediction: Prediction) -> None:
        pass

    def test_skipped(self, test: Test, error: bool = False) -> None:
        pass


class EvaluatorListener:
    def evaluate_started(self) -> None:
        pass

    def evaluate_prediction_started(self, prediction: Prediction) -> None:
        pass

    def evaluate_prediction_ended(self, evaluation: Evaluation) -> None:
        pass

    def evaluate_module_started(self, function_id: FunctionID) -> None:
        pass

    def evaluate_module_ended(self) -> None:
        pass

    def evaluate_ended(self, evaluations: list[Evaluation]) -> None:
        pass
