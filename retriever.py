import os
import pickle
from utils.logger import logger
from time import perf_counter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from config import EMBEDDING_MODEL, VECTOR_DB_PATH, TOP_K, SEARCH_TYPE, FETCH_K, LAMBDA_MULT, DEBUG_RETRIEVAL, CHUNKS_PATH

_embeddings = None
_vectorstore = None

def load_embeddings():

    global _embeddings

    if _embeddings is None:

        logger.info("Loading embeddings...")

        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


def load_vectorstore(embeddings):

    global _vectorstore

    if _vectorstore is None:

        if not os.path.exists(VECTOR_DB_PATH):
            raise FileNotFoundError(
                f"Vector database not found: {VECTOR_DB_PATH}"
            )

        logger.info("Loading vector database...")

        _vectorstore = FAISS.load_local(
            folder_path=VECTOR_DB_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

    return _vectorstore

def load_chunks():

    if not os.path.exists(CHUNKS_PATH):
        raise FileNotFoundError(
            f"Chunks file not found: {CHUNKS_PATH}"
        )

    logger.info("Loading chunks...")

    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)

    return chunks

def create_bm25_retriever(chunks, k):

    logger.info("Initializing BM25 retriever...")

    retriever = BM25Retriever.from_documents(chunks)

    retriever.k = k

    return retriever


def create_retriever(search_type=SEARCH_TYPE, k=TOP_K, fetch_k=FETCH_K, lambda_mult=LAMBDA_MULT, faiss_weight=1.0,  bm25_weight=0.0):

    try:
        
        valid = {"similarity", "mmr"}

        if search_type not in valid:
            raise ValueError(f"search_type must be one of {valid}")
        if search_type == "mmr" and fetch_k < k:
            raise ValueError("fetch_k must be greater than or equal to k for MMR.")
        if not 0 <= lambda_mult <= 1:
            raise ValueError("lambda_mult must be between 0 and 1.")

        embeddings = load_embeddings()

        vectorstore = load_vectorstore(embeddings)

        bm25_retriever = None

        if bm25_weight > 0:
            chunks = load_chunks()

            bm25_retriever = create_bm25_retriever(
                chunks,
                k,
            )

        logger.info("Initializing faiss Retriever...")

        faiss_retriever = vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={
                "k":k,
                "fetch_k":fetch_k,
                "lambda_mult":lambda_mult
                }
            )
        
        if bm25_weight == 0.0:

            logger.info("Using FAISS only retriever")

            retriever = faiss_retriever

        elif faiss_weight == 0.0:

            logger.info("Using BM25 only retriever")

            retriever = bm25_retriever

        else:

            logger.info("Using Hybrid retriever")


            retriever = EnsembleRetriever(
                retrievers=[
                    faiss_retriever,
                    bm25_retriever,
                ],
                weights=[
                    faiss_weight,
                    bm25_weight,
                ],
            )
        
        return retriever
        
    except Exception:
        logger.exception("Retriever initialization failed.")
        raise


def retrieve_documents(retriever, query: str,):

    try:
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Query: {query}")

        start = perf_counter()

        documents = retriever.invoke(query)

        end = perf_counter()

        logger.info(f"Retrieved {len(documents)} documents in {(end-start):.3f} seconds")

        if DEBUG_RETRIEVAL:
            for i, doc in enumerate(documents, start=1):
                logger.info(f"[{i}] {doc.metadata['chunk_id']} " f"({doc.metadata['word_count']} words)")

        return documents
    
    except Exception:
        logger.exception("Retrieval failed.")
        raise
