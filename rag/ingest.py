# ingest.py - handles all the PDF stuff
# loads PDFs, splits them, embeds them, stores in Chroma

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import yaml


# load config once
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)


def load_pdfs(pdf_paths):
    # takes a list of PDF file paths, returns list of Document objects
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        pages = loader.load()
        docs.extend(pages)
    return docs


def split_docs(docs):
    # split documents into chunks using config values
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunking"]["chunk_size"],
        chunk_overlap=cfg["chunking"]["chunk_overlap"]
    )
    return splitter.split_documents(docs)


def get_embeddings():
    # returns the embedding model - free, runs locally
    return HuggingFaceEmbeddings(
        model_name=cfg["embeddings"]["model"]
    )


def create_vector_store(chunks, collection_name=None):
    # creates a NEW Chroma collection from chunks
    # call this when you first upload PDFs
    embeddings = get_embeddings()
    
    coll = collection_name or cfg["vector_store"]["collection_name"]
    persist = cfg["vector_store"]["persist_directory"]
    
    # remove old db if exists so we start fresh
    if os.path.exists(persist):
        import shutil
        shutil.rmtree(persist)
    
    store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=coll,
        persist_directory=persist
    )
    return store


def get_vector_store(collection_name=None):
    # loads an EXISTING Chroma collection
    # call this when answering questions
    embeddings = get_embeddings()
    
    coll = collection_name or cfg["vector_store"]["collection_name"]
    persist = cfg["vector_store"]["persist_directory"]
    
    store = Chroma(
        collection_name=coll,
        embedding_function=embeddings,
        persist_directory=persist
    )
    return store


def count_docs(store):
    # helper to check how many chunks are in the store
    return store._collection.count()