import json
import pickle
from pathlib import Path
from prompt import LABEL_PROMPT
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from typing import List
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

POOL_PATH = "Eval/data/pool.json"
CHUNKS_PATH = "vectorstore/chunks.pkl"
OUTPUT_PATH = "Eval/data/ground_truth.json"

with open(POOL_PATH, "r", encoding="utf-8") as f:
    pool = json.load(f)

with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

if Path(OUTPUT_PATH).exists():
    with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
        labeled_pool = json.load(f)
else:
    labeled_pool = []

chunk_lookup = {}

for chunk in chunks:
    chunk_id = chunk.metadata["chunk_id"]
    chunk_lookup[chunk_id] = chunk

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0,
)

prompt = PromptTemplate(
    template=LABEL_PROMPT,
    input_variables=["question", "chunks"],
)

def format_chunks(chunk_ids, chunk_lookup):
    parts = []

    for chunk_id in chunk_ids:
        chunk = chunk_lookup[chunk_id]

        parts.append(
            f"""Chunk ID: {chunk_id}

{chunk.page_content}
"""
        )

    return "\n\n-----------------------------\n\n".join(parts)


class ChunkLabel(BaseModel):
    chunk_id: str = Field(description="Unique chunk ID")
    relevance: int = Field(description="0, 1, or 2")


class LabelResponse(BaseModel):
    labels: List[ChunkLabel]


structured_llm = llm.with_structured_output(LabelResponse)

chain = prompt | structured_llm


def review_labels(question, response, chunk_lookup):
    print("\n" + "=" * 100)
    print("QUESTION:")
    print(question)

    display_map = {}
    idx = 1

    for relevance in [2, 1]:
        print("\n" + "=" * 100)
        print(f"LABEL {relevance}")
        print("=" * 100)

        found = False

        for label in response.labels:
            if label.relevance == relevance:
                found = True

                display_map[idx] = label

                print(f"\n[{idx}] {label.chunk_id}")
                print("-" * 80)
                print(chunk_lookup[label.chunk_id].page_content)

                idx += 1

        if not found:
            print("None")

    return display_map


def apply_corrections(display_map, response):
    print("\nExamples:")
    print("none")
    print("exit")
    print("2=1")
    print("2=1 4=2 5=0")

    user_input = input("\nCorrections: ").strip().lower()

    if user_input == "exit":
        return "exit"

    if user_input in ("", "none"):
        return response

    changes = user_input.split()

    for change in changes:
        try:
            index, new_label = change.split("=")

            index = int(index)
            new_label = int(new_label)

            if index not in display_map:
                print(f"Invalid index: {index}")
                continue

            if new_label not in (0, 1, 2):
                print(f"Invalid label: {new_label}")
                continue

            display_map[index].relevance = new_label

        except Exception:
            print(f"Ignoring invalid input: {change}")

    return response


def print_final_labels(response):
    print("\nFinal Labels")
    print("=" * 60)

    for label in response.labels:
        print(f"{label.chunk_id:<45} {label.relevance}")


for i, sample in enumerate(pool[len(labeled_pool):], start=len(labeled_pool) + 1):

    print("\n" + "#" * 100)
    print(f"Question {i}/{len(pool)}")
    print("#" * 100)

    question = sample["question"]

    formatted_chunks = format_chunks(
        sample["ground_truth"],
        chunk_lookup,
    )

    print(f"\nQuestion {i}/{len(pool)}")
    print(question)

    response = chain.invoke(
        {
            "question": question,
            "chunks": formatted_chunks,
        }
    )

    display_map = review_labels(
        question,
        response,
        chunk_lookup,
    )

    response = apply_corrections(
        display_map,
        response,
    )

    if response == "exit":
        print("\nStopping... Progress has been saved.")
        break

    print_final_labels(response)

    labeled_pool.append(
        {
            "id": sample["id"],
            "question": question,
            "labels": [
                {
                    "chunk_id": label.chunk_id,
                    "relevance": label.relevance,
                }
                for label in response.labels
            ],
        }
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            labeled_pool,
            f,
            indent=2,
            ensure_ascii=False,
        )
        