"""Create a local SHAP force plot for the first held-out record."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier

from utils import DataLoader

project = Path(__file__).resolve().parent
loader = DataLoader()
loader.load_dataset(str(project / "data/healthcare-dataset-stroke-data.csv"))
loader.preprocess_data()
X_train, X_test, y_train, y_test = loader.get_data_split()
sampler = RandomOverSampler(sampling_strategy="minority", random_state=2021)
X_train, y_train = sampler.fit_resample(X_train, y_train)
model = RandomForestClassifier(random_state=2021, n_jobs=-1)
model.fit(X_train, y_train)

case = X_test.iloc[[0]]
class_index = list(model.classes_).index(1)
explainer = shap.TreeExplainer(model)
explanation = explainer(case)
base = explanation.base_values[0, class_index]
values = explanation.values[0, :, class_index]
probability = model.predict_proba(case)[0, class_index]
assert np.isclose(base + values.sum(), probability)

output = project / "outputs" / "shap"
output.mkdir(parents=True, exist_ok=True)
shap.force_plot(base, values, case.iloc[0], matplotlib=True, show=False,
                figsize=(18, 4), contribution_threshold=0.05)
plt.savefig(output / "force_plot.png", dpi=180, bbox_inches="tight")
plt.close()
shap.save_html(str(output / "force_plot.html"),
               shap.force_plot(base, values, case.iloc[0]))
print(f"Case index: {case.index[0]}, actual class: {y_test.iloc[0]}")
print(f"Base: {base:.6f}, model probability: {probability:.6f}")
print("Top contributions:")
for i in np.argsort(np.abs(values))[::-1][:8]:
    print(f"  {case.columns[i]} = {case.iloc[0, i]}: {values[i]:+.6f}")
print(f"Saved plots to {output}")
