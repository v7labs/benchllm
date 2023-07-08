from pathlib import Path

from benchllm.cache import FileCache
from benchllm.cli.listener import ReportListener, RichCliListener
from benchllm.cli.utils import add_cache, get_evaluator
from benchllm.utils import find_json_yml_files, load_prediction_files


def evaluate_predictions(
    file_or_dir: list[Path], model: str, output_dir: Path, workers: int, evaluator_name: str, cache: str
) -> bool:
    files = find_json_yml_files(file_or_dir)

    cli_listener = RichCliListener(root_dir=Path.cwd(), interactive=evaluator_name == "interactive", eval_only=True)
    report_listener = ReportListener(output_dir=output_dir)

    load_prediction_files(file_or_dir)

    evaluator = get_evaluator(evaluator_name, model, workers)
    evaluator = add_cache(cache, evaluator, output_dir.parent / "cache.json")

    cli_listener.set_evaulator(evaluator)

    evaluator.add_listener(cli_listener)
    evaluator.add_listener(report_listener)
    for file in files:
        evaluator.load_prediction_file(file)

    evaluator.run()
    return not evaluator.failed
