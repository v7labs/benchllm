from benchllm import Prediction, Test
from benchllm.data_types import FunctionID
from benchllm.evaluator import Evaluator, PipeEvaluator

PREDICTION_EXAMPLE = Prediction(
    test=Test(input="Who are you?", expected=["Yoda I am.", "Yoda"]),
    output="I am Yoda.",
    time_elapsed=0,
    function_id=FunctionID.default(),
)


class HardCoded(Evaluator):
    def __init__(self, passes: list[bool]):
        super().__init__(workers=1)
        self.passes = passes

    def evaluate_prediction(self, prediction: Prediction) -> list[Evaluator.Candidate]:
        assert len(self.passes) == len(prediction.test.expected)
        candidates = []
        for passed, expected in zip(self.passes, prediction.test.expected):
            candidates.append(
                Evaluator.Candidate(
                    prediction=prediction.output, expected=expected, score=1.0 if passed else 0, passed=passed
                )
            )
        return candidates


def test_evaluator_runs_all_evaluators():
    pipe = PipeEvaluator([HardCoded([False, False]), HardCoded([True, False])])
    pipe.load([PREDICTION_EXAMPLE])
    evaluations = pipe.run()
    assert len(evaluations) == 1
    assert evaluations[0].passed


def test_evaluator_runs_all_evaluators_with_all_failures():
    pipe = PipeEvaluator([HardCoded([False, False]), HardCoded([False, False])])
    pipe.load([PREDICTION_EXAMPLE])

    evaluations = pipe.run()
    assert len(evaluations) == 1
    assert not evaluations[0].passed


def test_evaluator_runs_all_evaluators_with_early_stopping():
    pipe = PipeEvaluator([HardCoded([False, True]), HardCoded([])])
    pipe.load([PREDICTION_EXAMPLE])

    evaluations = pipe.run()
    assert len(evaluations) == 1
    assert evaluations[0].passed
