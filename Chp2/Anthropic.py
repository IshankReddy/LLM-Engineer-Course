import os
from dotenv import load_dotenv
import anthropic

load_dotenv()
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

# Connect to Anthropic
# Having problems with API files? You can use claude = anthropic.Anthropic(api_key="your-key-here")
claude = anthropic.Anthropic()

system_message = "You are an assistant that is great at telling jokes"
user_prompt = "Tell a light-hearted joke for an audience of Data Scientists"

# Claude 3.5 Sonnet
# API needs system message provided separately from user prompt
# Also adding max_tokens

message = claude.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=200,
    temperature=0.7,
    system=system_message,
    messages=[
        {"role": "user", "content": user_prompt},
    ]
)

print(message.content[0].text)

# Claude 3.5 Sonnet again
# Now let's add in streaming back results

# system_message = "You are an assistant that is great at telling a short story"
# user_prompt = "Tell a short story for an audience of Kids"

# result = claude.messages.stream(
#     model="claude-3-5-sonnet-20240620",
#     max_tokens=200,
#     temperature=0.7,
#     system=system_message,
#     messages=[
#         {"role": "user", "content": user_prompt},
#     ],
# )

# with result as stream:
#     for text in stream.text_stream:
#             print(text, end="", flush=True)