import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def build_features(df: pd.DataFrame, pass_threshold: int = 70):
    data = df.copy()
    data["Performance_Label"] = np.where(data["Exam_Score"] >= pass_threshold, "High Performance", "Low Performance")
    data["Performance_Target"] = np.where(data["Exam_Score"] >= pass_threshold, 1, 0)
    feature_cols = [c for c in data.columns if c not in ["Exam_Score", "Performance_Label", "Performance_Target"]]
    X = data[feature_cols]
    y = data["Performance_Target"]
    return data, X, y

def build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_cols),
    ])
    return preprocessor
