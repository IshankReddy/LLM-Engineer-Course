# imports
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

# Connect to OpenAI
# Having problems with API files? You can use openai = OpenAI(api_key="your-key-here")

openai = OpenAI()

prompt1 = [
    {"role": "system", "content": "You are an assistant that is great at telling jokes"},
    {"role": "user", "content": "Tell a light-hearted joke for an audience of Data Scientists"}
  ]

# GPT-4o
# Temperature setting controls creativity

completion = openai.chat.completions.create(
    model='gpt-4o',
    messages=prompt1,
    temperature=0.7
)
print(completion.choices[0].message.content)

# Have it stream back results in 

# prompt2 = [
#     {"role": "system", "content": "You are an assistant that is great at telling a short story"},
#     {"role": "user", "content": "Tell a short story for an audience of Kids"}
#   ]

# stream = openai.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=prompt2,
#     stream=True,
# )
# for chunk in stream:
#     if chunk.choices[0].delta.content is not None:
#         print(chunk.choices[0].delta.content, end="")