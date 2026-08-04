import json
import pickle
from pathlib import Path
from langchain_classic.retrievers import EnsembleRetriever
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from retriever import (load_embeddings, load_vectorstore, load_chunks, create_bm25_retriever,)
from config import (FAISS_WEIGHT, BM25_WEIGHT,)

BENCHMARK_PATH = "Eval/data/benchmark.json"
CHUNKS_PATH = "vectorstore/chunks.pkl"
OUTPUT_PATH = "Eval/data/pool.json"

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

embeddings = load_embeddings()

vectorstore = load_vectorstore(embeddings)

chunks = load_chunks()

faiss_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 10,
    },
)

bm25_retriever = create_bm25_retriever(
    chunks,
    k=10,
)

hybrid_retriever = EnsembleRetriever(
    retrievers=[
        faiss_retriever,
        bm25_retriever,
    ],
    weights=[
        FAISS_WEIGHT,
        BM25_WEIGHT,
    ],
)

retrievers = {
    "faiss": faiss_retriever,
    "bm25": bm25_retriever,
    "hybrid": hybrid_retriever,
}

print("Retrievers initialized.")

pool = []

for idx, item in enumerate(benchmark, start=1):

    question = item["question"]

    retrieved_chunk_ids = set()


    for name, retriever in retrievers.items():

        docs = retriever.invoke(question)

        for doc in docs:

            chunk_id = doc.metadata.get("chunk_id")

            if chunk_id:
                retrieved_chunk_ids.add(chunk_id)


    pool.append(
        {
            "id": idx,
            "question": question,
            "ground_truth": list(retrieved_chunk_ids),
        }
    )


# Save
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        pool,
        f,
        indent=2,
        ensure_ascii=False
    )


print(f"Saved pool to {OUTPUT_PATH}")