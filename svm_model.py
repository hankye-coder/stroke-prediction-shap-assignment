"""Train and evaluate the alternative SVM on the existing data split."""
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

PROJECT = Path(__file__).resolve().parent
data = pd.read_csv(PROJECT / "data/healthcare-dataset-stroke-data.csv")
X = data.drop(columns=["id", "stroke"])
y = data["stroke"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=2021
)
categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numeric = [column for column in X.columns if column not in categorical]
preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
])
train_transformed = preprocessor.fit_transform(X_train)
test_transformed = preprocessor.transform(X_test)
train_sampled, labels_sampled = RandomOverSampler(
    sampling_strategy="minority", random_state=2021
).fit_resample(train_transformed, y_train)
model = SVC(kernel="rbf", probability=True, random_state=2021)
model.fit(train_sampled, labels_sampled)

if __name__ == "__main__":
    prediction = model.predict(test_transformed)
    print("Training and test sizes:", len(X_train), len(X_test))
    print("Oversampled training size:", len(train_sampled))
    print("Classes:", model.classes_)
    print("Accuracy:", accuracy_score(y_test, prediction))
    print("Confusion matrix:\n", confusion_matrix(y_test, prediction))
    print(classification_report(y_test, prediction, digits=4, zero_division=0))
    print("Case index:", X_test.index[0], "actual:", y_test.iloc[0])
    print("Case predicted class:", prediction[0])
    print("Case probability for class 1:", model.predict_proba(test_transformed[[0]])[0, list(model.classes_).index(1)])
