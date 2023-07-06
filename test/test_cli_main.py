from pathlib import Path
from test.utils import create_openai_object
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from benchllm.cli.main import app

runner = CliRunner()


@patch("openai.Completion.create", return_value=create_openai_object("Hello, user!"))
def test_run_multiple_suites(completion_mock: MagicMock):
    runner.invoke(app, ["run", "examples/qa", "examples/similarity"])
    completion_mock.assert_called()


@patch("openai.Completion.create", return_value=create_openai_object("Hello, user!"))
def test_run_target_suite(completion_mock: MagicMock):
    runner.invoke(app, ["run", "examples/qa"])
    completion_mock.assert_called()


def test_list_tests():
    result = runner.invoke(app, ["tests", str(Path.cwd() / "examples/qa")])
    assert "Input" in result.stdout
    assert "No." in result.stdout
    assert "Expected" in result.stdout
