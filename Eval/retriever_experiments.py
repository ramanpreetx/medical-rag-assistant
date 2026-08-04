from evaluate import evaluate_retriever
from retriever import create_retriever
import pandas as pd

experiments = [
    ("similarity", 1.0, 0.0),
    ("similarity", 0.8, 0.2),
    ("similarity", 0.6, 0.4),
    ("similarity", 0.5, 0.5),
    ("similarity", 0.4, 0.6),
    ("similarity", 0.2, 0.8),
    ("similarity", 0.0, 1.0),

    ("mmr", 1.0, 0.0),
    ("mmr", 0.8, 0.2),
    ("mmr", 0.6, 0.4),
    ("mmr", 0.5, 0.5),
    ("mmr", 0.4, 0.6),
    ("mmr", 0.2, 0.8),
    ("mmr", 0.0, 1.0),
]

results = []

for search_type, faiss_weight, bm25_weight in experiments:

    print("=" * 80)
    print(f"Search Type : {search_type}")
    print(f"FAISS Weight: {faiss_weight}")
    print(f"BM25 Weight : {bm25_weight}")
    print("=" * 80)

    retriever = create_retriever(
        search_type=search_type,
        k=10,
        faiss_weight=faiss_weight,
        bm25_weight=bm25_weight,
    )

    metrics = evaluate_retriever(retriever)


    for metric, value in metrics.items():
        print(f"{metric:<15}: {value:.3f}")

    results.append({
        "search_type": search_type,
        "faiss_weight": faiss_weight,
        "bm25_weight": bm25_weight,
        **metrics
    })

    

df = pd.DataFrame(results)

df.to_csv("Eval/results/experiment.csv", index=False)

print("\nSaved results to experiment.csv")

print(df.sort_values(by="MRR", ascending=False))