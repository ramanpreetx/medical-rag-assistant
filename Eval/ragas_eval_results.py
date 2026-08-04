import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from retriever import create_retriever, retrieve_documents
from reranker import load_reranker, rerank_documents
from formatter import format_context
from prompts import MEDICAL_PROMPT
from llm import get_llm
from config import RERANK_THRESHOLD
import time
import os

retriever = create_retriever()
llm = get_llm()                                     
load_reranker()

with open("Eval/data/ragas_eval.json", encoding="utf-8") as f:
    questions = json.load(f)

OUTPUT_FILE = "Eval/data/ragas_evaluation_results_top3.json"

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)
else:
    results = []

for sample in questions:

    question = sample["question"]

    documents = retrieve_documents(
        retriever,
        question
    )

    reranked = rerank_documents(
        question,
        documents
    )

    top_score = reranked[0][1] if reranked else 0

    accepted = bool(top_score >= RERANK_THRESHOLD)

    top_docs = reranked[:3]

    retrieved_contexts = [
        doc.page_content
        for doc, score in top_docs
    ]

    answer = ""

    if accepted:

        context = format_context(
            [doc for doc, score in top_docs]
        )

        prompt = MEDICAL_PROMPT.format(
            context=context,
            question=question
        )

        time.sleep(4.5)

        try:
            response = llm.invoke(prompt)

            if isinstance(response.content, list):
                answer = response.content[0]["text"]
            else:
                answer = response.content

        except Exception as e:
            print(f"LLM failed for: {question}")
            print(e)
            answer = ""

    results.append({
        "id": sample["id"],
        "question": question,
        "label": sample["label"],
        "accepted": accepted,
        "top_score": float(top_score),
        "contexts": retrieved_contexts,
        "answer": answer,
    })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump( results, f, indent=4, ensure_ascii=False)

    print(f"Saved question {sample['id']}")

print("Finished.")