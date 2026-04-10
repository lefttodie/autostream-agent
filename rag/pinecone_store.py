import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()

pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

def init_vectorstore(texts):
    embeddings = OpenAIEmbeddings()
    index_name = os.getenv("PINECONE_INDEX")

    return Pinecone.from_texts(texts, embeddings, index_name=index_name)