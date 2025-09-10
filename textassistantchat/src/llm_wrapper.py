import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

client = ChatNVIDIA(
    model="deepseek-ai/deepseek-r1-0528",
    api_key=NVIDIA_API_KEY,
    temperature=0.6,
    top_p=0.7,
    max_tokens=4096,
)

def generate(prompt):
    response = ""
    for chunk in client.stream([{"role": "user", "content": prompt}]):
        if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
            response += chunk.additional_kwargs["reasoning_content"]
        response += chunk.content
    return response