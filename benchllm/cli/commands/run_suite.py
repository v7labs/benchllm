from pathlib import Path

import typer

from benchllm.cli.listener import ReportListener, RichCliListener
from benchllm.cli.utils import get_evaluator
from benchllm.tester import Tester
from benchllm.utils import find_files


def run_suite(
    *,
    file_search_paths: list[Path],
    model: str,
    output_dir: Path,
    no_eval: bool,
    workers: int,
    evaluator_name: str,
    retry_count: int,
) -> bool:
    files = find_files(file_search_paths)
    if not files:
        typer.secho(
            f"No python files with @benchllm.test found in {', '.join(map(str, file_search_paths))}",
            fg=typer.colors.RED,
            bold=True,
        )
        return False

    cli_listener = RichCliListener(root_dir=Path.cwd(), interactive=evaluator_name == "interactive", test_only=no_eval)
    report_listener = ReportListener(output_dir=output_dir)

    tester = Tester(retry_count=retry_count)
    tester.add_listener(cli_listener)
    tester.add_listener(report_listener)

    # Load the the python files first, then the tests.
    for file in files:
        tester.load_module(file)

    # Finally, start collecting the predictions.
    tester.run()

    if no_eval:
        return True

    evaluator = get_evaluator(evaluator_name, model, workers)
    evaluator.add_listener(cli_listener)
    evaluator.add_listener(report_listener)
    evaluator.load(tester.predictions)

    evaluator.run()
    return not evaluator.failed
