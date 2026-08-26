import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = "data/credit_risk.csv"
MODEL_PATH = "models/best_credit_risk_model.joblib"

df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Credit_Risk")
plt.title("Credit Risk Distribution")
plt.tight_layout()
plt.savefig("credit_risk_distribution.png")
plt.close()

X = df.drop("Credit_Risk", axis=1)
y = df["Credit_Risk"].map({"Good": 1, "Bad": 0})

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
}

results = []
best_model = None
best_name = None
best_f1 = -1

for name, classifier in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    result = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0)
    }
    results.append(result)

    if result["F1 Score"] > best_f1:
        best_f1 = result["F1 Score"]
        best_model = pipeline
        best_name = name

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False)
print("\nModel Comparison:")
print(results_df.to_string(index=False))

os.makedirs("models", exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"\nBest Model: {best_name}")
print(f"Model saved at: {MODEL_PATH}")
