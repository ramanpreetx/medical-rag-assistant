import pandas as pd
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from retriever import create_retriever
from reranker import rerank_documents
from config import RERANK_THRESHOLD

INPUT_FILE = "Eval/results/threshold_calibration.csv"   
OUTPUT_FILE = "Eval/results/threshold_summary.csv"

# ------------------------------------
# Load CSV
# ------------------------------------

df = pd.read_csv(INPUT_FILE)

retriever = create_retriever()

results = []

for _, row in df.iterrows():

    question = row["question"]

    # Optional: if your CSV has labels
    label = row["label"] if "label" in df.columns else "Unknown"

    docs = retriever.invoke(question)

    reranked = rerank_documents(
        query=question,
        documents=docs
    )

    if len(reranked):
        best_score = reranked[0][1]
    else:
        best_score = 0.0

    results.append({
        "question": question,
        "label": label,
        "top_score": best_score,
        "accepted": best_score >= RERANK_THRESHOLD
    })

results_df = pd.DataFrame(results)

# ------------------------------------
# Summary
# ------------------------------------

summary = (
    results_df.groupby("label")
    .agg(
        Total=("label", "count"),
        Accepted=("accepted", "sum"),
    )
)

summary["Rejected"] = summary["Total"] - summary["Accepted"]
summary["Acceptance (%)"] = (
    summary["Accepted"] / summary["Total"] * 100
).round(2)

summary["Rejection (%)"] = (
    summary["Rejected"] / summary["Total"] * 100
).round(2)

summary = summary.reset_index()

summary.to_csv(OUTPUT_FILE, index=False)

print(summary)
print(f"\nSaved to {OUTPUT_FILE}")