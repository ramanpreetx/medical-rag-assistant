from evaluate import evaluate_retriever
from retriever import create_retriever
import pandas as pd

experiments = [
    # no reranker baselines
    {
        "retrieve_k": 5,
        "rerank": False,
        "rerank_k": None
    },
    {
        "retrieve_k": 10,
        "rerank": False,
        "rerank_k": None
    },

    # reranker experiments
    {
        "retrieve_k": 10,
        "rerank": True,
        "rerank_k": 5
    },
    {
        "retrieve_k": 20,
        "rerank": True,
        "rerank_k": 5
    },
    {
        "retrieve_k": 20,
        "rerank": True,
        "rerank_k": 3
    },
]

results = []

for exp in experiments:

    retrieve_k = exp["retrieve_k"]
    use_reranker = exp["rerank"]
    rerank_k = exp["rerank_k"]

    print("="*80)
    print(f"Retrieve K={retrieve_k}, Final K={rerank_k}")

    retriever = create_retriever(
        search_type="similarity",
        k=retrieve_k,
        faiss_weight=1.0,
        bm25_weight=0.0
    )

    metrics = evaluate_retriever(
        retriever,
        use_reranker=use_reranker,
        rerank_top_k=rerank_k
    )

    results.append({
        "retrieve_k": retrieve_k,
        "reranker": use_reranker,
        "rerank_k": rerank_k,
        **metrics
    })

df = pd.DataFrame(results)

df.to_csv("Eval/results/reranker_experiments.csv",index=False)

print("\nSaved reranker_experiments.csv")