import openai

import benchllm


def complete_text(prompt):
    full_prompt = f"""
    You are a friendly AI bot created by V7. You are tasked with answering questions about the world.
    Q: {prompt}
    A:"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=full_prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
    )
    return response.choices[0].text.strip()


@benchllm.test(suite=".")
def run(input: str):
    return complete_text(input)
