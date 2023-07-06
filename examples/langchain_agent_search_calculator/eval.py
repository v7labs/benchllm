from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.llms import OpenAI

# import benchllm

tools = load_tools(["serpapi", "llm-math"], llm=OpenAI(temperature=0))
agent = initialize_agent(tools, OpenAI(temperature=0), agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# @benchllm.test
def run(input: str):
    try:
        return agent(input)["output"]
    except Exception as e:
        return str(e)
