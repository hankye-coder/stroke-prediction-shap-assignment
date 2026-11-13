"""Generate an approximate global Kernel SHAP explanation for the SVM."""
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from svm_model import X_test, y_test, model, preprocessor, train_transformed, test_transformed

output = Path(__file__).resolve().parent / "outputs" / "shap"
output.mkdir(parents=True, exist_ok=True)
names = preprocessor.get_feature_names_out()
rng = np.random.default_rng(2021)
background = train_transformed[rng.choice(len(train_transformed), 50, replace=False)]
# Retain the local case and draw two positive and 47 other negative cases.
positive = np.flatnonzero(y_test.to_numpy() == 1)
negative = np.flatnonzero(y_test.to_numpy() == 0)
negative = negative[negative != 0]
test_positions = np.concatenate(([0], rng.choice(positive, 2, replace=False),
                                 rng.choice(negative, 47, replace=False)))
test_sample = test_transformed[test_positions]
class_index = list(model.classes_).index(1)

def predict_stroke_probability(values):
    return model.predict_proba(values)[:, class_index]

np.random.seed(2021)
start = time.perf_counter()
explainer = shap.KernelExplainer(predict_stroke_probability, background, link="identity")
values = explainer.shap_values(test_sample, nsamples=256, silent=True)
elapsed = time.perf_counter() - start
prediction = predict_stroke_probability(test_sample)
max_residual = np.max(np.abs(float(explainer.expected_value) + values.sum(axis=1) - prediction))
assert np.allclose(float(explainer.expected_value) + values.sum(axis=1), prediction, atol=1e-5)
display_names = [name.replace("numeric__", "").replace("categorical__", "") for name in names]
features = pd.DataFrame(test_sample, columns=display_names)
shap.summary_plot(values, features, max_display=15, show=False, plot_size=(10, 8))
plt.xlabel("SHAP value for stroke = 1 SVM probability")
plt.tight_layout()
plt.savefig(output / "kernel_summary_plot.png", dpi=180, bbox_inches="tight")
plt.close()
ranking = pd.Series(np.abs(values).mean(axis=0), index=display_names, name="mean_abs_shap")
ranking.sort_values(ascending=False).to_csv(output / "kernel_summary_importance.csv")
pd.DataFrame({"row_index": X_test.index[test_positions], "stroke": y_test.iloc[test_positions].to_numpy()}).to_csv(output / "kernel_summary_cases.csv", index=False)
print("Cases:", len(test_sample), "positive:", int(y_test.iloc[test_positions].sum()))
print("Background: 50 training rows; test: local case + 2 positive + 47 negative")
print("Sampling seed: 2021; nsamples: 256")
print("SHAP shape:", values.shape, "maximum reconstruction residual:", max_residual)
print("Elapsed seconds:", round(elapsed, 2))
print(ranking.sort_values(ascending=False).head(8).to_string())
