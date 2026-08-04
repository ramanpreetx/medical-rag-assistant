import json
import csv
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from retriever import create_retriever
from reranker import rerank_documents

QUESTIONS_PATH = "Eval/data/calibration_questions.json"
OUTPUT_PATH = "Eval/results/threshold_calibration.csv"


with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

medical_questions = questions[:50]
out_domain_questions = questions[50:100]
out_domain_related_medical = questions[100:150]

retriever = create_retriever(
    search_type="similarity",
    k=10,
    faiss_weight=1.0,
    bm25_weight=0.0
)

results = []

def get_top_score(question):

    documents = retriever.invoke(question)

    reranked = rerank_documents(
        question,
        documents,
        top_k=5
    )

    if not reranked:
        return None

    return float(reranked[0][1])

print("\nRunning medical questions...")

for i, item in enumerate(medical_questions, start=1):

    question = item["question"]

    score = get_top_score(question)

    print(f"[Medical {i}/50] Score={score:.4f}")

    results.append(
        {
            "question": question,
            "label": "medical",
            "top_score": score
        }
    )


print("\nRunning out-of-domain questions...")

for i, item in enumerate(out_domain_questions, start=1):

    question = item["question"]

    score = get_top_score(question)

    print(f"[OOD {i}/50] Score={score:.4f}")

    results.append(
        {
            "question": question,
            "label": "out_of_domain",
            "top_score": score
        }
    )

print("\nRunning out-of-domain-related-medical questions...")

for i, item in enumerate(out_domain_related_medical, start=1):

    question = item["question"]

    score = get_top_score(question)

    print(f"[OODRM {i}/50] Score={score:.4f}")

    results.append(
        {
            "question": question,
            "label": "out_of_domain_related_medical",
            "top_score": score
        }
    )

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "question",
            "label",
            "top_score"
        ]
    )

    writer.writeheader()
    writer.writerows(results)


print("\n" + "="*80)
print("Calibration finished")
print(f"Saved results: {OUTPUT_PATH}")