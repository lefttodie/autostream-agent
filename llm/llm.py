from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatOpenAI(
        model="mistralai/mistral-7b-instruct",  # OpenRouter model
        temperature=0.3,
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        default_headers={
            "HTTP-Referer": "http://localhost:8501",
            "X-Title": "AutoStream-Agent"
        }
    )