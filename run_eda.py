"""Run the teacher's EDA script and save reproducible outputs.

Run from any directory with the assignment's Python environment.
"""

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy

PROJECT = Path(__file__).resolve().parent
os.chdir(PROJECT)
OUTPUT = PROJECT / "outputs" / "eda"
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def save_teacher_plot(*args, **kwargs):
    """Save each original histogram when the teacher script calls show()."""
    column = next(plot_columns)
    plt.title(column)
    plt.tight_layout()
    plt.savefig(OUTPUT / f"hist_{column}.png", dpi=150)
    plt.close()


raw = pd.read_csv(PROJECT / "data/healthcare-dataset-stroke-data.csv")
plot_columns = iter(raw.columns)
original_show = plt.show
plt.show = save_teacher_plot
try:
    with (OUTPUT / "teacher_run.txt").open("w") as log:
        with contextlib.redirect_stdout(log):
            result = runpy.run_path(str(PROJECT / "00_data_exploration.py"))
finally:
    plt.show = original_show

# A bare .head() expression is displayed by notebook cells, but not by Python scripts.
raw.head().to_csv(OUTPUT / "raw_head.csv", index=False)
processed = result["data_loader"].data
processed.head().to_csv(OUTPUT / "processed_head.csv", index=False)
raw.describe(include="all").to_csv(OUTPUT / "descriptive_statistics.csv")
missing = raw.isna().sum()
missing.rename("missing_count").to_csv(OUTPUT / "missing_values.csv")
counts = raw["stroke"].value_counts().sort_index()
counts.rename("count").to_csv(OUTPUT / "stroke_counts.csv")

summary = {
    "raw_shape": list(raw.shape),
    "processed_shape": list(processed.shape),
    "stroke_counts": {str(k): int(v) for k, v in counts.items()},
    "stroke_percent": float(100 * raw["stroke"].mean()),
    "missing_counts": {k: int(v) for k, v in missing.items()},
    "bmi_missing_percent": float(100 * raw["bmi"].isna().mean()),
    "smoking_unknown_count": int(raw["smoking_status"].eq("Unknown").sum()),
    "duplicate_rows": int(raw.duplicated().sum()),
    "duplicate_ids": int(raw["id"].duplicated().sum()),
    "bmi_zeros_after_teacher_preprocessing": int(processed["bmi"].eq(0).sum()),
    "dataset_sha256": hashlib.sha256(
        (PROJECT / "data/healthcare-dataset-stroke-data.csv").read_bytes()
    ).hexdigest(),
    "versions": {
        name: importlib.metadata.version(name)
        for name in ["pandas", "numpy", "matplotlib", "scikit-learn", "imbalanced-learn"]
    },
}
(OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
bars = axes[0].bar(["stroke = 0", "stroke = 1"], counts.values,
                   color=["#4D7897", "#C35C50"])
axes[0].bar_label(bars, labels=[f"{n:,}\n({100*n/len(raw):.2f}%)" for n in counts], padding=5)
axes[0].set_ylim(0, 5800)
axes[0].set_title("Target distribution")
axes[0].set_ylabel("Number of records")
axes[1].hist(raw["bmi"].dropna(), bins=30, color="#4D7897", edgecolor="white")
axes[1].set_title(f"BMI before preprocessing\n{missing['bmi']} missing values")
axes[1].set_xlabel("BMI")
axes[1].set_ylabel("Number of records")
smoking = raw["smoking_status"].value_counts()
bars = axes[2].barh(smoking.index, smoking.values, color=[
    "#C18C37" if label == "Unknown" else "#4D7897" for label in smoking.index
])
axes[2].bar_label(bars, padding=4)
axes[2].set_xlim(0, 2250)
axes[2].set_title("Smoking status")
axes[2].set_xlabel("Number of records")
fig.tight_layout()
fig.savefig(OUTPUT / "eda_overview.png", dpi=160)
plt.close(fig)

assert counts.sum() == len(raw)
assert processed["stroke"].equals(raw["stroke"])
assert not processed.isna().any().any()
assert "id" not in processed.columns
assert summary["bmi_zeros_after_teacher_preprocessing"] == int(missing["bmi"])
print(json.dumps(summary, indent=2))
print(f"Saved EDA results to {OUTPUT}")
