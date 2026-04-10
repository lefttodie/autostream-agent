import os
from dotenv import load_dotenv
from agent_logic import OpenRouterEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

load_dotenv()

def seed():
    data = [
        "AutoStream Basic Plan: $29/month, 10 videos/month, 720p.",
        "AutoStream Pro Plan: $79/month, Unlimited videos, 4K, AI captions.",
        "Policies: No refunds after 7 days. 24/7 support for Pro users only."
    ]
    docs = [Document(page_content=t) for t in data]
    PineconeVectorStore.from_documents(docs, OpenRouterEmbeddings(), index_name=os.getenv("PINECONE_INDEX"))
    print("✅ Knowledge Base Seeded (1536 Dims)")

if __name__ == "__main__":
    seed()