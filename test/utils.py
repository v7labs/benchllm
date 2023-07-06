from openai.openai_object import OpenAIObject


def create_openai_object(text):
    obj = OpenAIObject()
    message = OpenAIObject()
    message.text = text
    obj.choices = [message]
    return obj
