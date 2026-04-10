from langchain_pinecone import PineconeVectorStore
from rag.embeddings import get_embeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv()

def get_retriever():
    embeddings = get_embeddings()

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index(os.getenv("PINECONE_INDEX"))

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )

    return vectorstore.as_retriever(search_kwargs={"k": 3})