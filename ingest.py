import logging
import pickle
import json
import os
from utils.logger import logger
from pathlib import Path
from collections import defaultdict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import DATA_PATH, VECTOR_DB_PATH, EMBEDDING_MODEL, CHUNK_OVERLAP, CHUNK_SIZE, CHUNKS_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("faiss").disabled = True

logger = logging.getLogger(__name__)


def clean_documents(documents):
    for doc in documents:
        text = doc.page_content

        text = text.replace("\t", " ")
        text = text.replace("\r", "")

        lines = [line.rstrip() for line in text.splitlines()]
        text = "\n".join(lines)

        doc.page_content = text

    return documents


def load_topics_metadata():

    with open("data/topics.json", "r", encoding="utf-8") as f:
        topics = json.load(f)

    metadata = {}

    for topic in topics:

        metadata[topic["title"].lower()] = topic

    return metadata


def load_documents(data_path):

    documents = []

    topics_metadata = load_topics_metadata()

    md_files = Path(data_path).glob("*.md")

    for md_file in md_files:

        loader = TextLoader(
            str(md_file),
            encoding="utf-8"
        )

        docs = loader.load()

        title = md_file.stem.replace("_", " ").lower()

        topic = topics_metadata.get(title)

        for doc in docs:

            doc.metadata["title"] = md_file.stem

            if topic:

                doc.metadata["url"] = topic["url"]
                doc.metadata["categories"] = topic["categories"]

        documents.extend(docs)

    if not documents:
        raise ValueError("No markdown files found.")

    return documents


def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    page_chunk_counter = defaultdict(int)

    for chunk in chunks:
        
        document_name = chunk.metadata["title"]

        page_chunk_counter[document_name] += 1

        chunk.metadata["chunk_id"] = (
            f"{document_name}_c{page_chunk_counter[document_name]}"
        )

        chunk.metadata["document_name"] = document_name
        
        text = chunk.page_content

        chunk.metadata["char_count"] = len(text)

        chunk.metadata["word_count"] = len(text.split())

    return chunks


def save_chunks(chunks):

    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    logger.info("Chunks saved successfully")


def analyze_chunks(chunks):
    lengths = [len(chunk.page_content) for chunk in chunks]
    word_counts = [chunk.metadata["word_count"] for chunk in chunks]

    logger.info(f"Average words/chunk: {sum(word_counts)/len(word_counts):.1f}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Average chunk length: {sum(lengths) / len(lengths):.1f} characters")
    logger.info(f"Smallest chunk: {min(lengths)} characters")
    logger.info(f"Largest chunk: {max(lengths)} characters")

    
def create_embeddings():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    return embeddings


def create_vectorstore(chunks, embeddings):
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    vectorstore.save_local(VECTOR_DB_PATH)

    return vectorstore

    
def main():

    try:
        logger.info("Loading documents...")

        documents = load_documents(DATA_PATH)

        logger.info(f"Loaded {len(documents)} documents")

        logger.info("Cleaning documents...")

        documents = clean_documents(documents)

        logger.info("Splitting documents...")

        chunks = split_documents(documents)

        logger.info(f"Created {len(chunks)} chunks")

        analyze_chunks(chunks)

        logger.info("Loading embedding model...")

        embeddings = create_embeddings()

        logger.info("Creating vector database...")

        create_vectorstore(
            chunks,
            embeddings
        )

        logger.info("Vector database saved successfully")

        save_chunks(chunks)
        
    except Exception:
        logger.exception("Ingestion failed.")
        

if __name__ == "__main__":
    main()
