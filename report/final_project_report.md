# Final Project Report

## Judul
**Student Performance Risk Classifier**

## Background
Institusi pendidikan sering terlambat mengidentifikasi siswa yang berisiko memiliki performa akademik rendah. Dengan memanfaatkan data seperti jam belajar, kehadiran, nilai sebelumnya, sesi tutoring, dan akses sumber belajar, sekolah dapat mendeteksi potensi risiko lebih awal.

## Goal
Membangun sistem prediksi untuk mengklasifikasikan siswa ke dalam kategori **High Performance** atau **Low Performance** agar pihak sekolah dapat mengambil keputusan intervensi yang lebih cepat.

## Dataset
- Nama file: `StudentPerformanceFactors.csv`
- Jumlah baris: 6,607
- Jumlah kolom: 20

## Flow Project
1. Data loading
2. Data cleaning dan missing value handling
3. Exploratory Data Analysis
4. Feature engineering target klasifikasi
5. Training 4 model
6. Evaluasi model
7. Simpan model terbaik
8. Deployment ke Streamlit

## Hasil Evaluasi Utama
- Best model: **Logistic Regression**
- Accuracy: **0.979**
- Precision: **0.977**
- Recall: **0.935**
- ROC-AUC: **0.984**

## Impact
- Membantu guru dan sekolah mengenali siswa yang memerlukan dukungan tambahan.
- Menjadi dasar pembuatan dashboard monitoring performa akademik.
- Dapat dikembangkan untuk sistem rekomendasi intervensi belajar.

## Deployment
Aplikasi Streamlit tersedia pada folder `app/streamlit_app.py` dan siap di-deploy ke Streamlit Community Cloud.
