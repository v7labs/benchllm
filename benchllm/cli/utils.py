import datetime
from pathlib import Path

from benchllm.cli.evaluator import InteractiveEvaluator, WebEvaluator
from benchllm.evaluator import Evaluator, SemanticEvaluator, StringMatchEvaluator


def output_dir_factory() -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path.cwd() / "output" / str(timestamp)
    output_dir.mkdir(exist_ok=True, parents=True)

    latest = Path.cwd() / "output" / "latest"
    if latest.exists():
        latest.unlink()
    latest.symlink_to(output_dir)
    return output_dir


def get_evaluator(evaluator_name: str, model: str, workers: int) -> Evaluator:
    if evaluator_name == "semantic":
        return SemanticEvaluator(model=model, workers=workers)
    elif evaluator_name == "interactive":
        return InteractiveEvaluator()
    elif evaluator_name == "string-match":
        return StringMatchEvaluator(workers=workers)
    elif evaluator_name == "web":
        return WebEvaluator()
    else:
        raise ValueError(f"Unknown evaluator {evaluator_name}")
