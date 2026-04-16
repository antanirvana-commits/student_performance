from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "StudentPerformanceFactors.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_THRESHOLD = 75


def load_data():
    return pd.read_csv(DATA_PATH)


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Performance"] = (df["Exam_Score"] >= TARGET_THRESHOLD).astype(int)
    return df


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def main():
    df = load_data()
    df = create_target(df)

    X = df.drop(columns=["Exam_Score", "Performance"])
    y = df["Performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42
        )
    }

    all_metrics = {}
    best_model_name = None
    best_pipeline = None
    best_auc = -1

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator)
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test)
        all_metrics[model_name] = metrics

        current_auc = metrics["roc_auc"]
        if current_auc > best_auc:
            best_auc = current_auc
            best_model_name = model_name
            best_pipeline = pipeline

        print(f"\n=== {model_name} ===")
        for k, v in metrics.items():
            if k == "confusion_matrix":
                print(f"{k}: {v}")
            else:
                print(f"{k}: {v:.4f}")

    output_metrics = {
        "best_model": best_model_name,
        "models": all_metrics
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=4)

    joblib.dump(best_pipeline, MODEL_DIR / "best_model_pipeline.joblib")

    print("\nTraining selesai.")
    print(f"Best model: {best_model_name}")
    print(f"Metrics saved to: {ARTIFACTS_DIR / 'metrics.json'}")
    print(f"Best model saved to: {MODEL_DIR / 'best_model_pipeline.joblib'}")


if __name__ == "__main__":
    main()