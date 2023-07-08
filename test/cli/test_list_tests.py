from pathlib import Path

from typer.testing import CliRunner

from benchllm.cli.main import app

runner = CliRunner()


def test_list_tests():
    result = runner.invoke(app, ["tests", str(Path.cwd() / "examples/qa")])
    assert "Input" in result.stdout
    assert "No." in result.stdout
    assert "Expected" in result.stdout
