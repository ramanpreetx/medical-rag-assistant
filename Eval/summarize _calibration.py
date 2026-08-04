import pandas as pd

INPUT_FILE = "Eval/results/threshold_calibration.csv"
OUTPUT_FILE = "Eval/results/summary_calibration.csv"

df = pd.read_csv(INPUT_FILE)

summary = (
    df.groupby("label")["top_score"]
      .agg(
          Total="count",
          Minimum="min",
          Q1=lambda x: x.quantile(0.25),
          Median="median",
          Mean="mean",
          Q3=lambda x: x.quantile(0.75),
          Maximum="max",
          Std="std",
      )
      .round(4)
)

summary.to_csv(OUTPUT_FILE)

print(summary)