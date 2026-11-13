"""Compare teacher preprocessing with training-only median BMI imputation."""
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

project = Path(__file__).resolve().parent
data = pd.read_csv(project / "data/healthcare-dataset-stroke-data.csv")
X = data.drop(columns=["id", "stroke"])
y = data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2021
)
categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numeric = [column for column in X.columns if column not in categorical]
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
train_categories = encoder.fit_transform(X_train[categorical])
test_categories = encoder.transform(X_test[categorical])
category_names = encoder.get_feature_names_out(categorical)
missing_bmi = X_test["bmi"].isna().to_numpy()

for strategy in ["teacher_zero", "training_median"]:
    if strategy == "teacher_zero":
        train_numeric = X_train[numeric].fillna({"bmi": 0})
        test_numeric = X_test[numeric].fillna({"bmi": 0})
    else:
        imputer = SimpleImputer(strategy="median")
        train_numeric = pd.DataFrame(imputer.fit_transform(X_train[numeric]), columns=numeric)
        test_numeric = pd.DataFrame(imputer.transform(X_test[numeric]), columns=numeric)
        print("Training BMI median:", imputer.statistics_[numeric.index("bmi")])
    train = pd.DataFrame(np.column_stack([train_categories, train_numeric]),
                         columns=[*category_names, *numeric])
    test = pd.DataFrame(np.column_stack([test_categories, test_numeric]),
                        columns=[*category_names, *numeric])
    train, sampled_y = RandomOverSampler(
        sampling_strategy="minority", random_state=2021
    ).fit_resample(train, y_train.reset_index(drop=True))
    model = RandomForestClassifier(random_state=2021, n_jobs=-1)
    model.fit(train, sampled_y)
    predicted = model.predict(test)
    class_index = list(model.classes_).index(1)
    explanation = shap.TreeExplainer(model)(test)
    values = explanation.values[:, :, class_index]
    model_output = model.predict_proba(test)[:, class_index]
    residual = np.max(np.abs(explanation.base_values[:, class_index] + values.sum(axis=1) - model_output))
    bmi_shap = values[:, test.columns.get_loc("bmi")]
    print("\n", strategy)
    print("Accuracy:", accuracy_score(y_test, predicted))
    print("Precision:", precision_score(y_test, predicted, zero_division=0))
    print("Recall:", recall_score(y_test, predicted))
    print("Confusion matrix:\n", confusion_matrix(y_test, predicted))
    print("Missing BMI test cases:", missing_bmi.sum())
    print("Mean BMI SHAP for missing BMI:", bmi_shap[missing_bmi].mean())
    print("Mean absolute BMI SHAP overall:", np.abs(bmi_shap).mean())
    print("Max SHAP reconstruction error:", residual)
