import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Load dataset
df = pd.read_csv("data/StudentPerformanceFactors.csv")

# Buat target (klasifikasi)
df['Performance'] = df['Exam_Score'].apply(lambda x: 1 if x >= 75 else 0)

# Drop kolom yang tidak perlu (sesuaikan jika beda)
X = df.drop(columns=['Exam_Score', 'Performance'])
y = df['Performance']

# Encoding (kalau ada data kategorikal)
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))

# Simpan model
joblib.dump(model, "models/model.pkl")
print("Model saved!")