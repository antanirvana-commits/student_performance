from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "data/raw/StudentPerformanceFactors.csv"
PROCESSED_DATA_PATH = BASE_DIR / "data/processed/student_performance_processed.csv"
MODEL_PATH = BASE_DIR / "artifacts/models/best_model_pipeline.joblib"
PASS_THRESHOLD = 70
RANDOM_STATE = 42
