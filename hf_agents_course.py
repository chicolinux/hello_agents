'''
I am using Ollama to serve different LLMs locally.
I have installed the Ollama CLI and the Mistral 7B model.
You are free to use any other model you like.
More info about Ollama can be found here: https://ollama.com/docs/cli
'''

# from smolagents import LiteLLMModel
from transformers import AutoTokenizer

'''model = LiteLLMModel(
    model_id="mistral-small3.1:24b",
    api_base="http://127.0.0.1:11434",
    num_ctx=8192,
)'''

messages = [
    {"role": "system",
     "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
print(f"The messages object is of type {type(messages)} with length {len(messages)}")
# The messages object is a list of dictionaries, each dictionary representing a message in the chat.
# The first message is the system message, which sets the context for the conversation.
# The second message is from the user, and the third message is from the assistant.

# I'm going to show how to use different chat templates to interact with LLMs.
tokenizer_mistral = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1"
)
rendered_prompt = tokenizer_mistral.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f"The rendered prompt is of type {type(rendered_prompt)}")
print(f"The rendered prompt is {rendered_prompt}")
