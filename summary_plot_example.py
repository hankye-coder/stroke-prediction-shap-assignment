"""Explain held-out predictions with a SHAP summary plot."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

from utils import DataLoader

project = Path(__file__).resolve().parent
loader = DataLoader()
loader.load_dataset(str(project / "data/healthcare-dataset-stroke-data.csv"))
loader.preprocess_data()
X_train, X_test, y_train, y_test = loader.get_data_split()
X_train, y_train = RandomOverSampler(
    sampling_strategy="minority", random_state=2021
).fit_resample(X_train, y_train)
model = RandomForestClassifier(random_state=2021, n_jobs=-1)
model.fit(X_train, y_train)
class_index = list(model.classes_).index(1)
explainer = shap.TreeExplainer(model)
explanation = explainer(X_test)
values = explanation.values[:, :, class_index]
predictions = model.predict_proba(X_test)[:, class_index]
reconstructed = explanation.base_values[:, class_index] + values.sum(axis=1)
max_residual = np.max(np.abs(reconstructed - predictions))
assert np.allclose(reconstructed, predictions, atol=1e-6)

output = project / "outputs" / "shap"
output.mkdir(parents=True, exist_ok=True)
shap.summary_plot(values, X_test, max_display=15, show=False, plot_size=(10, 8))
plt.xlabel("SHAP value for stroke = 1 model probability")
plt.tight_layout()
plt.savefig(output / "summary_plot.png", dpi=180, bbox_inches="tight")
plt.close()
ranking = pd.Series(np.abs(values).mean(axis=0), index=X_test.columns, name="mean_abs_shap")
ranking = ranking.sort_values(ascending=False)
ranking.to_csv(output / "summary_feature_importance.csv")
print(f"Explained cases: {len(X_test)}, stroke=1: {int(y_test.sum())}")
print(f"SHAP values shape: {values.shape}")
print(f"Maximum reconstruction residual: {max_residual:.3e}")
print(ranking.head(10).to_string())
print(f"Saved results to {output}")
