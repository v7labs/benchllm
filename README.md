# 🏋️‍♂️ BenchLLM 🏋️‍♀️

🦾 Continuous Integration for LLM powered appplications 🦙🦅🤖 

[![GitHub Repo stars](https://img.shields.io/github/stars/v7labs/BenchLLM?style=social)](https://github.com/v7labs/BenchLLM/stargazers)
[![Twitter Follow](https://img.shields.io/twitter/follow/V7Labs?style=social)](https://twitter.com/V7Labs)
[![Discord Follow](https://dcbadge.vercel.app/api/server/HrhPdZf83u?style=flat)](https://discord.gg/HrhPdZf83u)

BenchLLM is a Python-based open-source library that streamlines the testing of Large Language Models (LLMs) and AI-powered applications. It measures the accuracy of your model, agents, or chains by validating responses on any number of tests via LLMs.

BenchLLM is actively used at [V7](https://www.v7labs.com) for improving our LLM applications and now Open Sourced under MIT License to share with the wider community


## 💡 Get help on [Discord](https://discord.gg/HrhPdZf83u) or [Tweet at us](https://twitter.com/V7Labs)

<hr/>

Use BenchLLM to:

- Test the responses of your LLM across any number of prompts.
- Continuous integration for chains like [Langchain](https://github.com/hwchase17/langchain), agents like [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT), or LLM models like [Llama](https://github.com/facebookresearch/llama) or GPT-4.
- Eliminate flaky chains and create confidence in your code.
- Spot inaccurate responses and hallucinations in your application at every version.

<hr/>

> ⚠️ **NOTE:** BenchLLM is in the early stage of development and will be subject to rapid changes.
>
>For bug reporting, feature requests, or contributions, please open an issue or submit a pull request (PR) on our GitHub page.

## 🧪 BenchLLM Testing Methodology

BenchLLM implements a distinct two-step methodology for validating your machine learning models:

1. **Testing**: This stage involves running your code against any number of expected responses and capturing the predictions produced by your model without immediate judgment or comparison.

2. **Evaluation**: The recorded predictions are compared against the expected output using LLMs to verify factual similarity (or optionally manually). Detailed comparison reports, including pass/fail status and other metrics, are generated.

This methodical separation offers a comprehensive view of your model's performance and allows for better control and refinement of each step.

## 🚀 Install

To install BenchLLM we use pip

```
pip install benchllm
```

## 💻 Usage

Start by importing the library and use the @benchllm.test decorator to mark the function you'd like to test:

```python
import benchllm

# Your custom model implementation
def run_my_model(input):
    # Your model's logic goes here.
    return some_result

@benchllm.test(suite="/path/to/test/suite") # If the tests are in the same directory, just use @benchllm.test.
def invoke_model(input: str):
    return run_my_model(input)
```

Next, prepare your tests. These are YAML/JSON files structured as follows:

```yml
input: What's 1+1? Be very terse, only numeric output
expected:
  - 2
  - 2.0
```

In the above example, the `input` is the query or instruction that your model will process, and `expected` contains the potential responses that your model should return. It's important to note that `input` can be a simple `str` or a more complex nested dictionary; BenchLLM will extract the type of the `input` argument in the Python code and load the `input` field from the YAML file accordingly.

By default, BenchLLM uses OpenAI's GPT-3 model for the `semantic` evaluator. This requires setting the `OPENAI_API_KEY` environment variable. If you do not want to use this default evaluator, you can specify an alternative one (discussed in further detail below):

```bash
export OPENAI_API_KEY='your-api-key'
```

Replace 'your-api-key' with your actual OpenAI API key.

To initiate testing, use the `bench run` command:

```bash
$ bench run
```

By default, the bench run command looks for Python files implementing the @test decorator in the current directory. To target a specific file or folder, specify it directly:

```bash
$ bench run path/to/my/file.py or/path/to/folder/with/files
```

The `--retry-count` parameter allows BenchLLM to run a test multiple times, useful for models that may have variability in their outputs:

```bash
$ bench run --retry-count 5
```

BenchLLM offers multiple evaluation methods to determine if the prediction matches the test case's expected values. You can use the `--evaluator` parameter to specify the evaluation method:

There are multiple ways to evaluate if the test functions prediction matches the test cases expected values.
By default GPT-3 is used to compare the output. You can use `--evaluator` to use a different method

- `semantic`, checks semantic similarity using language models like GPT-3, GPT-3.5, or GPT-4 (`--model` parameter). Please note, for this evaluator, you need to set the `OPENAI_API_KEY` environment variable.
- `string-match`, checks if the strings are matching (case insensitive)
- `interactive`, user manually accepts or fails tests in the terminal
- `web`, uses pywebio fora simple local web interface

The non interactive evaluators also supports `--workers N` to run in the evaluations in parallel

```bash
$ bench run --evaluator string-match --workers 5
```

### 🧮 Eval

While _bench run_ runs each test function and then evaluates their output, it can often be beneficial to separate these into two steps. For example, if you want a person to manually do the evaluation or if you want to try multiple evaluation methods on the same function.

```bash
$ bench run --no-eval
```

This will generate json files in `output/latest/predictions`
Then later you can evaluate them with

```bash
$ bench eval output/latest/predictions
```

## 🔌 API

For more detailed control, BenchLLM provides an API.
You are not required to add YML/JSON tests to be able to evaluate your model.
You can instead:

- Instantiate `Test` objects
- Use a `Tester` object to generate predictions
- Use an `Evaluator` object to evaluate your model

```python
from benchllm import StringMatchEvaluator, Test, Tester

# Instantiate your Test objects
tests = [
    Test(input="What's 1+1?", expected=["2", "It's 2"]),
    Test(input="First rule of fight club?", expected=["Do not talk about fight club"]),
]

# Use a Tester object to generate predictions using any test functions
tester = Tester(my_test_function)
tester.add_tests(tests)
predictions = tester.run()

# Use an Evaluator object to evaluate your model
evaluator = StringMatchEvaluator()
evaluator.load(predictions)
results = evaluator.run()

print(results)
```

## ☕️ Commands

- `bench add`: Add a new test to a suite.
- `bench tests`: List all tests in a suite.
- `bench run`: Run all or target test suites.
- `bench eval`: Runs the evaluation of an existing test run.

## 🙌 Contribute

BenchLLM is developed for Python 3.10, although it may work with other Python versions as well. We recommend using a Python 3.10 environment. You can use conda or any other environment manager to set up the environment:

```bash
$ conda create --name benchllm python=3.10
$ conda activate benchllm
$ pip install -e ".[dev]"
```

To run all the examples first install the examples extra dependencies

```bash
$ pip install -e ".[examples]"
```

Contribution steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes.
4. Test your changes.
5. Submit a pull request.

We adhere to PEP8 style guide. Please follow this guide when contributing.

For further information and advanced usage, please refer to the comprehensive BenchLLM documentation. If you need any support, feel free to open an issue on our GitHub page.
