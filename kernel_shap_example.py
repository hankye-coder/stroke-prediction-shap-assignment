"""Explain one held-out SVM prediction with Kernel SHAP."""
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
feature_names = preprocessor.get_feature_names_out()
background_indices = np.random.default_rng(2021).choice(len(train_transformed), 50, replace=False)
background = train_transformed[background_indices]
case = test_transformed[[0]]
class_index = list(model.classes_).index(1)

def predict_stroke_probability(values):
    return model.predict_proba(values)[:, class_index]

np.random.seed(2021)
start = time.perf_counter()
explainer = shap.KernelExplainer(predict_stroke_probability, background, link="identity")
values = explainer.shap_values(case, nsamples=1000, silent=True)
elapsed = time.perf_counter() - start
base = float(explainer.expected_value)
probability = float(predict_stroke_probability(case)[0])
contributions = values[0]
reconstructed = base + contributions.sum()
assert np.isclose(reconstructed, probability, atol=1e-5)
display_names = [name.replace("numeric__", "").replace("categorical__", "")
                 .replace("smoking_status_", "smoking: ")
                 .replace("avg_glucose_level", "glucose (scaled)")
                 for name in feature_names]
case_features = pd.Series(np.round(case[0], 2), index=display_names)
shap.force_plot(base, contributions, case_features, matplotlib=True,
                show=False, figsize=(18, 4), contribution_threshold=0.16)
plt.savefig(output / "kernel_force_plot.png", dpi=180, bbox_inches="tight")
plt.close()
shap.save_html(str(output / "kernel_force_plot.html"),
               shap.force_plot(base, contributions, case_features))
print("Case:", X_test.index[0], "actual class:", y_test.iloc[0])
print("Training background rows:", len(background), "seed: 2021")
print("SHAP array shape:", values.shape)
print("Base:", base, "sum:", contributions.sum())
print("Reconstructed:", reconstructed, "model output:", probability)
print("Residual:", reconstructed - probability)
print("Elapsed seconds:", round(elapsed, 2))
print("Largest absolute contributions (numeric features are standardized):")
for index in np.argsort(np.abs(contributions))[::-1][:8]:
    print(f"  {feature_names[index]} = {case[0,index]:.4f}: {contributions[index]:+.6f}")
