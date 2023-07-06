from pathlib import Path

import yaml
from langchain.evaluation.loading import load_dataset

SUITE_NAME = "langchain_agent_search_calculator"
suite_path = Path("examples") / SUITE_NAME
suite_path.mkdir(exist_ok=True, parents=True)

dataset = load_dataset("agent-search-calculator")

for i, data in enumerate(dataset, start=1):
    # {
    #   'steps': [{'tool': 'Search', 'tool_input': 'Population of Canada 2023'}],
    #   'answer': 'approximately 38,625,801',
    #   'question': 'How many people live in canada as of 2023?'
    # }
    with open(suite_path / f"{i}.yml", "w") as fp:
        benchllm_dict = {"expected": [data["answer"]], "input": data["question"]}
        yaml.safe_dump(benchllm_dict, fp)
