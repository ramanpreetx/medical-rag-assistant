import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from ragas.run_config import RunConfig

run_config = RunConfig(
    max_workers=4,
    timeout=180,
)

load_dotenv()

INPUT_FILE = "Eval/data/ragas_evaluation_results_top5.json"
OUTPUT_FILE = "Eval/results/ragas_medical_scores_top5.csv"


with open(INPUT_FILE, encoding="utf-8") as f:
    results = json.load(f)


medical_results = [item for item in results if item["label"] == "medical"]


print(f"Medical questions: {len(medical_results)}")


data = {
    "question": [],
    "answer": [],
    "contexts": []
}


for item in medical_results:

    data["question"].append(
        item["question"]
    )

    data["answer"].append(
        item["answer"]
    )

    data["contexts"].append(
        item["contexts"]
    )


dataset = Dataset.from_dict(data)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)

score = evaluate(
    dataset,
    metrics=[faithfulness],
    llm=llm,
    run_config=run_config,
)

print(score)

# Save detailed scores

score.to_pandas().to_csv(
    OUTPUT_FILE,
    index=False
)

print("Finished.")