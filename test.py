from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()  # 🔥 THIS LINE IS MISSING

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index = pc.Index(os.getenv("PINECONE_INDEX"))

print(index.describe_index_stats())