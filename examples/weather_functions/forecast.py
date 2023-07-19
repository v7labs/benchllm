import json

import openai


def get_n_day_weather_forecast(location: str, num_days: int):
    return f"The weather in {location} will be rainy for the next {num_days} days."


def chain(prompt: list[dict], functions):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=prompt, temperature=0.0, functions=functions
    )

    choice = response["choices"][0]
    if choice.get("finish_reason") == "function_call":
        function_call = choice["message"]["function_call"]
        function_name = function_call["name"]
        function_args = json.loads(function_call["arguments"])
        fn = globals()[function_name]
        output = fn(**function_args)
        prompt.append({"role": "function", "name": function_name, "content": output})
        return chain(prompt, functions)
    else:
        return response.choices[0].message.content.strip()


def run(question: str):
    messages = [
        {
            "role": "user",
            "content": "Only answer questions with 'yes', 'no' or 'unknown', you must not reply with anything else",
        },
        {"role": "system", "content": "Use the get_n_day_weather_forecast function for weather questions"},
        {"role": "user", "content": question},
    ]

    functions = [
        {
            "name": "get_n_day_weather_forecast",
            "description": "Get an N-day weather forecast",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "num_days": {
                        "type": "integer",
                        "description": "The number of days to forecast. E.g. 1 for today, 2 for tomorrow etc",
                    },
                },
                "required": ["location", "format", "num_days"],
            },
        },
    ]
    return chain(messages, functions)
