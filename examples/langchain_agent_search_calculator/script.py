from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

from benchllm import SemanticEvaluator, Test, Tester

tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))
agent = initialize_agent(tools, OpenAI(temperature=0), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

tests = [Test(input="How many people live in canada as of 2023?", expected=["approximately 38,625,801"])]

tester = Tester(lambda input: agent(input)["output"])
tester.add_tests(tests)
predictions = tester.run()

evaluator = SemanticEvaluator()
evaluator.load(predictions)
report = evaluator.run()

print(report)
