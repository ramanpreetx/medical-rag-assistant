import json
from pathlib import Path
import sys
import math
from statistics import mean

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from retriever import create_retriever
from reranker import rerank_documents

BENCHMARK_PATH = "Eval/data/benchmark.json"
LABELS_PATH = "Eval/data/ground_truth.json"

with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
    benchmark = json.load(f)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    ground_truth = json.load(f)


def recall_at_k(retrieved_ids, relevances, k):

    retrieved_topk = retrieved_ids[:k]

    relevant_chunks = {
        chunk_id
        for chunk_id, relevance in relevances.items()
        if relevance > 0
    }

    hits = sum(
        1
        for chunk_id in retrieved_topk
        if chunk_id in relevant_chunks
    )

    return hits / len(relevant_chunks) if relevant_chunks else 0.0

def precision_at_k(retrieved_ids, relevances, k):
    retrieved_topk = retrieved_ids[:k]

    relevant_chunks = {
        chunk_id
        for chunk_id, relevance in relevances.items()
        if relevance > 0
    }

    hits = sum(
        1
        for chunk_id in retrieved_topk
        if chunk_id in relevant_chunks
    )

    return hits / len(retrieved_topk)

def reciprocal_rank(retrieved_ids, relevances):
    relevant_chunks = {
        chunk_id
        for chunk_id, relevance in relevances.items()
        if relevance > 0
    }

    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_chunks:
            return 1 / rank

    return 0.0

def dcg_at_k(retrieved_ids, relevances, k):
    dcg = 0.0

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        relevance = relevances.get(chunk_id, 0)

        dcg += relevance / math.log2(rank + 1)

    return dcg

def idcg_at_k(relevances, k):
    ideal_relevances = sorted(
        [r for r in relevances.values() if r > 0],
    reverse=True
)

    idcg = 0.0

    for rank, relevance in enumerate(ideal_relevances[:k], start=1):

        idcg += relevance / math.log2(rank + 1)

    return idcg

def ndcg_at_k(retrieved_ids, relevances, k):
    dcg = dcg_at_k(retrieved_ids, relevances, k)
    idcg = idcg_at_k(relevances, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg

    
def evaluate_retriever(retriever, use_reranker=False, rerank_top_k=5):

    recall5_scores = []
    recall10_scores = []

    precision5_scores = []
    precision10_scores = []

    rr_scores = []

    ndcg5_scores = []
    ndcg10_scores = []

    precision3_scores = []

    for question_data, truth_data in zip(benchmark, ground_truth):

        question = question_data["question"]

        results = retriever.invoke(question)

        if use_reranker:

            reranked = rerank_documents(
                question,
                results,
                top_k=rerank_top_k
        )

            results = [
                doc for doc, score in reranked
            ]

        retrieved_ids = [
            doc.metadata["chunk_id"]
            for doc in results
        ]

        relevances = {
            label["chunk_id"]: label["relevance"]
            for label in truth_data["labels"]
        }

        recall5_scores.append(recall_at_k(retrieved_ids, relevances, 5))
        recall10_scores.append(recall_at_k(retrieved_ids, relevances, 10))

        precision5_scores.append(precision_at_k(retrieved_ids, relevances, 5))
        precision10_scores.append(precision_at_k(retrieved_ids, relevances, 10))

        if use_reranker and rerank_top_k == 3:

            precision3_scores.append(precision_at_k(retrieved_ids, relevances, 3))

        rr_scores.append(reciprocal_rank(retrieved_ids, relevances))

        ndcg5_scores.append(ndcg_at_k(retrieved_ids, relevances, 5))
        ndcg10_scores.append(ndcg_at_k(retrieved_ids, relevances, 10))

    metrics = {
        "Recall@5": mean(recall5_scores),
        "Recall@10": mean(recall10_scores),
        "Precision@5": mean(precision5_scores),
        "Precision@10": mean(precision10_scores),
        "MRR": mean(rr_scores),
        "nDCG@5": mean(ndcg5_scores),
        "nDCG@10": mean(ndcg10_scores),
    }

    if use_reranker and rerank_top_k == 3:

        metrics["Precision@3"] = mean(precision3_scores)

    return metrics