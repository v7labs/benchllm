import openai

import benchllm
from benchllm.input_types import ChatInput


def chat(messages: ChatInput):
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    return response.choices[0].message.content.strip()


@benchllm.test(suite=".")
def run(input: ChatInput):
    return chat(input)
