import tempfile
from pathlib import Path
from unittest.mock import Mock, call

from benchllm import Test, Tester


def test_tester_run_through_each_test_once():
    test_function = Mock(return_value="42")
    test = Tester(test_function=test_function)
    test.add_test(Test(input="1+1", expected=["2"]))
    test.add_test(Test(input="2+2", expected=["4"]))
    predictions = test.run()

    assert test_function.call_count == 2

    print(test_function.call_args_list)
    test_function.assert_has_calls([call("1+1"), call("2+2")])
    assert predictions[0].output == "42"
    assert predictions[1].output == "42"


def test_tester_parses_yml_correctly():
    python_code = """
import benchllm

@benchllm.test(suite=".")
def test(input: str):
    return "42"
    """
    test_case = """
input: 1+1
expected: [2]
"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        with open(temp_dir / "test.py", "w") as f:
            f.write(python_code)
        with open(temp_dir / "test.yaml", "w") as f:
            f.write(test_case)

        test = Tester()
        test.load_module(temp_dir / "test.py")
        predictions = test.run()

        assert predictions[0].output == "42"
        assert predictions[0].test.input == "1+1"
        assert predictions[0].test.expected == ["2"]
