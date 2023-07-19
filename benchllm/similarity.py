import openai


def completion_func(prompt: str) -> str:
    response = openai.Completion.create(
        prompt=prompt, engine="text-davinci-003", max_tokens=100, temperature=0.7, n=1, stop=None
    )
    return response.choices[0].text.strip()


def chat_completion_func(prompt: str, *, model: str) -> str:
    response = openai.ChatCompletion.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=100, temperature=0.7, n=1, stop=None
    )
    return response.choices[0].message.content.strip()


def complete_text(prompt: str, *, model: str) -> str:
    full_prompt = f"""
    You will get two anwsers to a question, you should determine if they have the same meaning. Would two people agree that they mean the same, it's okay if different words are used.
    You can only answer "same" or "different", nothing else.

    input: {{ 
        "answer_1": "I was created by X",
        "answer_2": "X created me"
    }}
    output: same

    input: {{ 
        "answer_1": "There are 52 days in a year",
        "answer_2": "A year is fairly long"
    }}
    output: different

    input: {prompt}
    output:"""

    model_func = completion_func if model == "gpt-3" else lambda prompt: chat_completion_func(prompt, model=model)
    return model_func(prompt=full_prompt)


def semantically_similar(answer1: str, answer2: str, model: str = "gpt-3") -> bool:
    response = complete_text(
        f"""{{ 
        "answer_1": "{answer1}",
        "answer_2": "{answer2}"
    }}""",
        model=model,
    )
    if response not in ["same", "different"]:
        raise ValueError(f"Unexpected response: {response}")
    return response == "same"
