from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from sklearn.linear_model import (
    LogisticRegression,
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "StudentPerformanceFactors.csv"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"

ARTIFACTS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_THRESHOLD = 75


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def create_classification_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Performance"] = (df["Exam_Score"] >= TARGET_THRESHOLD).astype(int)
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))
    else:
        metrics["roc_auc"] = 0.0

    return metrics


def train_classification_models(df: pd.DataFrame):
    X = df.drop(columns=["Exam_Score", "Performance"])
    y = df["Performance"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            random_state=42,
        ),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
        ),
    }

    all_metrics = {}
    best_model_name = None
    best_pipeline = None
    best_auc = -1.0

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_classifier(pipeline, X_test, y_test)
        all_metrics[model_name] = metrics

        current_auc = metrics.get("roc_auc", 0.0)
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

    return all_metrics, best_model_name, best_pipeline


def evaluate_regressor(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return {
        "r2_score": float(r2),
        "mae": float(mae),
        "rmse": float(rmse),
    }


def train_regression_models(df: pd.DataFrame):
    X = df.drop(columns=["Exam_Score", "Performance"])
    y = df["Exam_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = build_preprocessor(X)

    regression_models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.001, max_iter=5000),
        "ElasticNet Regression": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=500,
            max_depth=18,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.9,
            random_state=42,
        ),
        "Extra Trees Regressor": ExtraTreesRegressor(
            n_estimators=500,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1,
        ),
        "AdaBoost Regressor": AdaBoostRegressor(
            n_estimators=300,
            learning_rate=0.05,
            random_state=42,
        ),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_depth=8,
            random_state=42,
        ),
    }

    all_regression_metrics = {}
    best_regression_name = None
    best_regression_pipeline = None
    best_r2 = -999.0

    for model_name, estimator in regression_models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        metrics = evaluate_regressor(pipeline, X_test, y_test)
        all_regression_metrics[model_name] = metrics

        current_r2 = metrics.get("r2_score", -999.0)
        if current_r2 > best_r2:
            best_r2 = current_r2
            best_regression_name = model_name
            best_regression_pipeline = pipeline

        print(f"\n=== {model_name} ===")
        print(f"r2_score: {metrics['r2_score']:.4f}")
        print(f"mae: {metrics['mae']:.4f}")
        print(f"rmse: {metrics['rmse']:.4f}")

    # Polynomial Ridge Regression
    polynomial_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    polynomial_pipeline.fit(X_train, y_train)
    poly_metrics = evaluate_regressor(polynomial_pipeline, X_test, y_test)
    all_regression_metrics["Polynomial Ridge Regression"] = poly_metrics

    current_r2 = poly_metrics.get("r2_score", -999.0)
    if current_r2 > best_r2:
        best_r2 = current_r2
        best_regression_name = "Polynomial Ridge Regression"
        best_regression_pipeline = polynomial_pipeline

    print("\n=== Polynomial Ridge Regression ===")
    print(f"r2_score: {poly_metrics['r2_score']:.4f}")
    print(f"mae: {poly_metrics['mae']:.4f}")
    print(f"rmse: {poly_metrics['rmse']:.4f}")

    return all_regression_metrics, best_regression_name, best_regression_pipeline


def main():
    df = load_data()
    df = create_classification_target(df)

    classification_metrics, best_model_name, best_classification_pipeline = train_classification_models(df)
    regression_metrics, best_regression_name, best_regression_pipeline = train_regression_models(df)

    output_metrics = {
        "best_model": best_model_name,
        "models": classification_metrics,
        "best_regression_model": best_regression_name,
        "regression_models": regression_metrics,
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(output_metrics, f, indent=4)

    joblib.dump(best_classification_pipeline, MODEL_DIR / "best_model_pipeline.joblib")
    joblib.dump(best_regression_pipeline, MODEL_DIR / "regression_model.joblib")

    print("\nTraining selesai.")
    print(f"Best classification model: {best_model_name}")
    print(f"Best regression model: {best_regression_name}")
    print(f"Metrics saved to: {ARTIFACTS_DIR / 'metrics.json'}")
    print(f"Classification model saved to: {MODEL_DIR / 'best_model_pipeline.joblib'}")
    print(f"Regression model saved to: {MODEL_DIR / 'regression_model.joblib'}")


if __name__ == "__main__":
    main()