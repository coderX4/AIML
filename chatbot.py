import openai
import os

# Set your OpenAI API key
openai.api_key = "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"
# Alternatively, you can set it as an environment variable and retrieve it like this:
# openai.api_key = os.getenv("OPENAI_API_KEY")

def chat_with_gpt(user_input):
    # Get the response from the API with only the current user input
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_input}
        ]
    )

    # Get the assistant's message
    assistant_message = response['choices'][0]['message']['content']
    return assistant_message

print("Chatbot is ready to talk! Type 'quit' to exit.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    bot_response = chat_with_gpt(user_input)
    print("ChatGPT:", bot_response)
