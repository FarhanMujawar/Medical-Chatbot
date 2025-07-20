from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv, find_dotenv
from pinecone import Pinecone
from pinecone import Pinecone, ServerlessSpec
import os

load_dotenv(dotenv_path="C:/Users/Farha/ZAI/Medical Chatbot/.env")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")

extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

print(f"Total text chunks: {len(text_chunks)}")
print(f"Sample chunk: {text_chunks[0]}")

# Using existing index
docsearch = PineconeVectorStore.from_existing_index(
    index_name="medical-bot",
    embedding=embeddings
)