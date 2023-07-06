import openai

import benchllm
from benchllm.input_types import ChatInput


def chat(messages: ChatInput):
    messages = [{"role": message.role, "content": message.content} for message in messages]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None,
    )

    return response.choices[0].message.content.strip()


@benchllm.test(suite=".")
def run(input: ChatInput):
    value = chat(input)
    return value
