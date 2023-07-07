import datetime
from pathlib import Path

from benchllm.cache import FileCache, MemoryCache
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


def add_cache(cache_name: str, evaluator: Evaluator, cache_path: Path) -> Evaluator:
    if cache_name == "file":
        return FileCache(evaluator, cache_path)
    elif cache_name == "memory":
        return MemoryCache(evaluator)
    elif cache_name == "none":
        return evaluator
    else:
        raise ValueError(f"Unknown cache {cache_name}, valid values are 'file', 'memory', 'none'")
