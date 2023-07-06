from pathlib import Path

from benchllm.cli.listener import ReportListener, RichCliListener
from benchllm.cli.utils import get_evaluator
from benchllm.evaluator import load_prediction_files
from benchllm.utils import find_json_yml_files


def evaluate_predictions(
    file_or_dir: list[Path], model: str, output_dir: Path, workers: int, evaluator_name: str
) -> bool:
    files = find_json_yml_files(file_or_dir)

    cli_listener = RichCliListener(root_dir=Path.cwd(), interactive=evaluator_name == "interactive", eval_only=True)
    report_listener = ReportListener(output_dir=output_dir)

    load_prediction_files(file_or_dir)

    evaluator = get_evaluator(evaluator_name, model, workers)
    evaluator.add_listener(cli_listener)
    evaluator.add_listener(report_listener)
    for file in files:
        evaluator.load_prediction_file(file)

    evaluator.run()
    return not evaluator.failed
