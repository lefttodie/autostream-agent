import os
import requests
from dotenv import load_dotenv
from langchain.embeddings.base import Embeddings

load_dotenv()

class OpenRouterEmbeddings(Embeddings):
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.url = "https://openrouter.ai/api/v1/embeddings"
        self.model = "nvidia/llama-nemotron-embed-vl-1b-v2"

    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            emb = self._get_embedding(text)
            embeddings.append(emb)
        return embeddings

    def embed_query(self, text):
        return self._get_embedding(text)

    def _get_embedding(self, text):
        response = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": text
            }
        )

        result = response.json()

        return result["data"][0]["embedding"]


def get_embeddings():
    return OpenRouterEmbeddings()