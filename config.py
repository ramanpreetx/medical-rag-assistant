VECTOR_DB_PATH = "vectorstore/faiss_index"
CHUNKS_PATH = "vectorstore/chunks.pkl"

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

DATA_PATH = "data/scraped_data"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

MODEL_NAME = "llama-3.3-70b-versatile"

TOP_K = 10

FETCH_K = 10
LAMBDA_MULT = 0.5

SEARCH_TYPE = "similarity"

DEBUG_RETRIEVAL = True

CROSS_ENCODER_MODEL = "BAAI/bge-reranker-base"

RERANK_TOP_K = 5

RERANK_THRESHOLD = 0.045