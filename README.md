# Student Performance Prediction Dashboard

## Deskripsi Project
Project ini merupakan sistem prediksi performa siswa berbasis **Machine Learning** yang dibangun menggunakan dataset faktor performa siswa.  
Aplikasi ini memiliki fitur utama:

- eksplorasi dataset
- visualisasi EDA
- perbandingan performa beberapa algoritma machine learning
- prediksi performa siswa secara interaktif melalui Streamlit

Project ini dibuat untuk memenuhi tugas **Final Project Nalar Bootcamp** dengan ketentuan:
- berbasis Machine Learning / Deep Learning
- memiliki EDA
- memiliki visualisasi
- memiliki deployment menggunakan Streamlit
- memiliki evaluasi model menggunakan metrik klasifikasi

---

## Tujuan Project
Tujuan project ini adalah untuk memprediksi apakah seorang siswa memiliki **High Performance** atau **Low Performance** berdasarkan berbagai faktor akademik dan non-akademik, seperti:

- Hours Studied
- Attendance
- Sleep Hours
- Previous Scores
- Tutoring Sessions
- Physical Activity
- Motivation Level
- Internet Access
- Family Income
- Teacher Quality
- School Type
- Parental Involvement
- Access to Resources
- Extracurricular Activities
- Gender
- Peer Influence
- Parental Education Level
- Learning Disabilities

---

## Target Klasifikasi
Kolom target dibuat dari nilai `Exam_Score` dengan aturan berikut:

- `Performance = 1` jika `Exam_Score >= 75`
- `Performance = 0` jika `Exam_Score < 75`

Sehingga sistem akan memprediksi:
- **High Performance**
- **Low Performance**

---

## Algoritma yang Digunakan
Project ini menggunakan beberapa algoritma klasifikasi untuk dibandingkan performanya:

1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. Gradient Boosting  
5. Neural Network  

Model terbaik dipilih berdasarkan nilai evaluasi terbaik, terutama **ROC-AUC**.

---

## Fitur Aplikasi Streamlit
Aplikasi Streamlit memiliki beberapa halaman utama:

### 1. Home
Menampilkan ringkasan project dan preview dataset.

### 2. Dataset Overview
Menampilkan informasi dataset seperti:
- preview data
- shape dataset
- data types
- missing values
- descriptive statistics
- unique values per column

### 3. EDA
Menampilkan visualisasi eksplorasi data:
- class distribution
- exam score histogram
- correlation heatmap
- boxplot exam score by performance

### 4. Model Performance
Menampilkan performa setiap model:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### 5. Prediction
Menampilkan form interaktif untuk memprediksi performa siswa berdasarkan parameter input.

---

## Struktur Folder Project
```text
student_performance/
│
├── app/
│   └── streamlit_app.py
│
├── artifacts/
│   ├── metrics.json
│   └── models/
│       └── best_model_pipeline.joblib
│
├── data/
│   ├── raw/
│   │   └── StudentPerformanceFactors.csv
│   └── processed/
│
├── src/
│   └── train_model.py
│
├── requirements.txt
├── Dockerfile
└── README.md

menjalankan system 
cd C:\Users\LENOVO\Documents\student_performance
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m src.train_model
streamlit run app\streamlit_app.py