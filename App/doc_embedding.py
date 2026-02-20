from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings

# Load env files
load_dotenv()
CHROMA_PATH = os.getenv("CHROMA_DB_PATH", "Chroma")

# Setup FAST API Configurations
app = FastAPI()

origins = [
    "http://localhost:8501",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define embedding model (Local embedding model through hugging face)
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")


def process_documents(file_path):
    docs = load_documents(file_path)
    chunks = split_text(docs)
    result = save_to_database(chunks)
    return result



def load_documents(file_path):
    loader = DirectoryLoader(file_path)
    return loader.load()


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10,
        chunk_overlap=5,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks



def save_to_database(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        db = Chroma(
            embedding_function=embedding_model,
            persist_directory=CHROMA_PATH
        )
        print("Loaded existing database.")
    else:
        db = Chroma(
            embedding_function=embedding_model,
            persist_directory=CHROMA_PATH
        )
        print("Created new database.")

    # Load exsisting meta data
    existing_docs = db.get(include=["metadatas"])
    existing_sources = set()
    if existing_docs and "metadatas" in existing_docs:
        for metadata in existing_docs["metadatas"]:
            if isinstance(metadata, dict):
                filename = os.path.basename(metadata.get("source", ""))
                if filename:
                    existing_sources.add(filename)

    # Detect duplicate file names from exsisting meta data
    duplicate_files = set()
    new_chunks = []
    for chunk in chunks:
        chunk_source_name = os.path.basename(chunk.metadata.get("source", ""))
        if chunk_source_name in existing_sources:
            duplicate_files.add(chunk_source_name)
        else:
            new_chunks.append(chunk)

    if duplicate_files:
        print("\nDuplicate documents found:", duplicate_files)
        return {
            "status": "duplicate",
            "duplicates": list(duplicate_files)
        }

    # If no duplicates save the new file
    db.add_documents(new_chunks)
    db.persist()
    print(f"Saved {len(new_chunks)} chunks to {CHROMA_PATH}.")

    return {"status": "success", "duplicates": []}