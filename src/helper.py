from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import os


#Extract data from the PDF
def load_pdf(data):
    if not os.path.exists(data):
        raise FileNotFoundError(f"Directory '{data}' not found. Please create it or check the path.")

    loader = DirectoryLoader(data,
                              glob="*.pdf",
                              loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents


#Create text chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)

    return text_chunks



#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings