from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "StudentPerformanceFactors.csv"
CLASS_MODEL_PATH = BASE_DIR / "artifacts" / "models" / "best_model_pipeline.joblib"
REG_MODEL_PATH = BASE_DIR / "artifacts" / "models" / "regression_model.joblib"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"

st.set_page_config(
    page_title="Student Performance Prediction Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b1020 0%, #111827 100%);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1300px;
        }

        h1, h2, h3, h4, h5, h6 {
            color: #f8fafc !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        p, div, label, span, li {
            color: #e5e7eb;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        section[data-testid="stSidebar"] * {
            color: #f9fafb !important;
        }

        .hero-box {
            background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%);
            border-radius: 24px;
            padding: 32px 30px;
            margin-bottom: 22px;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }

        .hero-title {
            font-size: 2.4rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.45rem;
            line-height: 1.1;
        }

        .hero-subtitle {
            font-size: 1rem;
            color: #e0e7ff;
            line-height: 1.7;
            max-width: 850px;
        }

        .section-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 18px;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
            backdrop-filter: blur(10px);
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 700;
            color: #f8fafc;
            margin-bottom: 0.9rem;
        }

        .subtle-text {
            color: #cbd5e1;
            line-height: 1.7;
            font-size: 0.98rem;
        }

        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
        }

        div[data-testid="metric-container"] label {
            color: #cbd5e1 !important;
            font-weight: 600 !important;
        }

        div[data-testid="metric-container"] > div {
            color: #ffffff !important;
        }

        .stButton > button {
            width: 100%;
            background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 14px;
            padding: 0.78rem 1.2rem;
            font-weight: 700;
            font-size: 0.98rem;
            box-shadow: 0 10px 22px rgba(37, 99, 235, 0.25);
        }

        .stButton > button:hover {
            opacity: 0.94;
        }

        .stSelectbox label,
        .stSlider label,
        .stNumberInput label {
            font-weight: 600 !important;
            color: #e5e7eb !important;
        }

        .stDataFrame, div[data-testid="stDataFrame"] {
            border-radius: 16px;
            overflow: hidden;
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
        }

        .footer-note {
            text-align: center;
            color: #94a3b8;
            font-size: 0.92rem;
            margin-top: 28px;
            padding-top: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_classification_model():
    if CLASS_MODEL_PATH.exists():
        return joblib.load(CLASS_MODEL_PATH)
    return None


@st.cache_resource
def load_regression_model():
    if REG_MODEL_PATH.exists():
        return joblib.load(REG_MODEL_PATH)
    return None


@st.cache_data
def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["Performance"] = (result["Exam_Score"] >= 75).astype(int)
    return result


def normalize_metrics(metrics_data: dict) -> dict:
    if not metrics_data:
        return {
            "best_model": None,
            "models": {},
            "best_regression_model": None,
            "regression_models": {},
        }

    return {
        "best_model": metrics_data.get("best_model"),
        "models": metrics_data.get("models", {}),
        "best_regression_model": metrics_data.get("best_regression_model"),
        "regression_models": metrics_data.get("regression_models", {}),
    }


def get_sorted_unique_values(df: pd.DataFrame, column_name: str):
    values = df[column_name].dropna().astype(str).unique().tolist()
    return sorted(values)


def show_section_title(title: str):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def start_card():
    st.markdown('<div class="section-card">', unsafe_allow_html=True)


def end_card():
    st.markdown("</div>", unsafe_allow_html=True)


def plot_class_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    df["Performance"].value_counts().sort_index().plot(kind="bar", ax=ax)
    ax.set_title("Class Distribution")
    ax.set_xlabel("Performance (0 = Low, 1 = High)")
    ax.set_ylabel("Count")
    st.pyplot(fig)


def plot_exam_score_histogram(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df["Exam_Score"].hist(ax=ax, bins=20)
    ax.set_title("Exam Score Distribution")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)


def plot_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    ax.set_title("Correlation Heatmap")
    fig.colorbar(im)
    st.pyplot(fig)


def plot_boxplot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df.boxplot(column="Exam_Score", by="Performance", ax=ax)
    ax.set_title("Exam Score by Performance")
    ax.set_xlabel("Performance")
    ax.set_ylabel("Exam Score")
    plt.suptitle("")
    st.pyplot(fig)


df = load_data()
df = add_target(df)
classification_model = load_classification_model()
regression_model = load_regression_model()
raw_metrics = load_metrics()
metrics_bundle = normalize_metrics(raw_metrics)

st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Student Performance Prediction Dashboard</div>
        <div class="hero-subtitle">
            Dashboard interaktif untuk eksplorasi dataset, evaluasi model machine learning,
            dan prediksi performa siswa berdasarkan faktor akademik dan non-akademik.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Pilih halaman",
    [
        "Home",
        "Dataset Overview",
        "EDA",
        "Model Performance",
        "Regression Performance",
        "Prediction",
    ],
)

if menu == "Home":
    show_section_title("Project Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Jumlah Data", len(df))
    c2.metric("Jumlah Kolom", df.shape[1])
    c3.metric("Jumlah Kelas", df["Performance"].nunique())

    start_card()
    st.markdown(
        """
        <div class="subtle-text">
            Project ini dibangun untuk memprediksi performa siswa menggunakan beberapa algoritma
            machine learning berdasarkan faktor-faktor seperti jam belajar, kehadiran, kualitas guru,
            motivasi, dukungan keluarga, dan faktor pendukung lainnya. Sistem ini juga menyediakan
            analisis data, perbandingan performa model, evaluasi regresi, serta prediksi secara real-time.
        </div>
        """,
        unsafe_allow_html=True,
    )
    end_card()

    start_card()
    show_section_title("Preview Dataset")
    st.dataframe(df.head(10), use_container_width=True)
    end_card()

elif menu == "Dataset Overview":
    show_section_title("Dataset Overview")

    overview_option = st.selectbox(
        "Pilih informasi dataset",
        [
            "Preview Data",
            "Shape Dataset",
            "Data Types",
            "Missing Values",
            "Descriptive Statistics",
            "Unique Values per Column",
        ],
    )

    start_card()

    if overview_option == "Preview Data":
        rows_to_show = st.slider("Jumlah baris", 5, 50, 10)
        st.dataframe(df.head(rows_to_show), use_container_width=True)

    elif overview_option == "Shape Dataset":
        c1, c2 = st.columns(2)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])

    elif overview_option == "Data Types":
        dtypes_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
            }
        )
        st.dataframe(dtypes_df, use_container_width=True)

    elif overview_option == "Missing Values":
        missing_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Values": df.isnull().sum().values,
            }
        ).sort_values("Missing Values", ascending=False)
        st.dataframe(missing_df, use_container_width=True)

    elif overview_option == "Descriptive Statistics":
        st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    elif overview_option == "Unique Values per Column":
        unique_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Unique Values": [df[col].nunique() for col in df.columns],
            }
        ).sort_values("Unique Values", ascending=False)
        st.dataframe(unique_df, use_container_width=True)

    end_card()

elif menu == "EDA":
    show_section_title("Exploratory Data Analysis")

    eda_option = st.selectbox(
        "Pilih visualisasi",
        [
            "Class Distribution",
            "Exam Score Histogram",
            "Correlation Heatmap",
            "Boxplot Exam Score by Performance",
        ],
    )

    start_card()

    if eda_option == "Class Distribution":
        plot_class_distribution(df)
        st.markdown(
            '<div class="subtle-text">Visualisasi ini menunjukkan distribusi kelas antara performa rendah dan performa tinggi.</div>',
            unsafe_allow_html=True,
        )

    elif eda_option == "Exam Score Histogram":
        plot_exam_score_histogram(df)
        st.markdown(
            '<div class="subtle-text">Histogram digunakan untuk melihat persebaran nilai ujian siswa secara keseluruhan.</div>',
            unsafe_allow_html=True,
        )

    elif eda_option == "Correlation Heatmap":
        plot_correlation_heatmap(df)
        st.markdown(
            '<div class="subtle-text">Heatmap korelasi menunjukkan hubungan antar variabel numerik pada dataset.</div>',
            unsafe_allow_html=True,
        )

    elif eda_option == "Boxplot Exam Score by Performance":
        plot_boxplot(df)
        st.markdown(
            '<div class="subtle-text">Boxplot memperlihatkan perbandingan distribusi nilai ujian antara dua kelas performa.</div>',
            unsafe_allow_html=True,
        )

    end_card()

elif menu == "Model Performance":
    show_section_title("Classification Model Performance")

    models_dict = metrics_bundle.get("models", {})
    best_model_name = metrics_bundle.get("best_model")

    if not models_dict:
        st.warning("Metrics klasifikasi belum tersedia. Jalankan ulang training model terlebih dahulu.")
        st.code("python -m src.train_model")
    else:
        selected_model = st.selectbox("Pilih model klasifikasi", list(models_dict.keys()))
        selected_metrics = models_dict.get(selected_model, {})

        if best_model_name:
            st.success(f"Best Classification Model: {best_model_name}")

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{selected_metrics.get('accuracy', 0):.4f}")
        c2.metric("Precision", f"{selected_metrics.get('precision', 0):.4f}")
        c3.metric("Recall", f"{selected_metrics.get('recall', 0):.4f}")
        c4.metric("F1-Score", f"{selected_metrics.get('f1_score', 0):.4f}")
        c5.metric("ROC-AUC", f"{selected_metrics.get('roc_auc', 0):.4f}")

        start_card()
        show_section_title("Confusion Matrix")
        cm = selected_metrics.get("confusion_matrix")
        if cm:
            cm_df = pd.DataFrame(
                cm,
                index=["Actual 0", "Actual 1"],
                columns=["Pred 0", "Pred 1"],
            )
            st.dataframe(cm_df, use_container_width=True)
        else:
            st.info("Confusion matrix belum tersedia.")
        end_card()

        start_card()
        show_section_title("Model Summary")
        st.markdown(
            f"""
            <div class="subtle-text">
                Model <strong>{selected_model}</strong> dievaluasi menggunakan Accuracy, Precision,
                Recall, F1-Score, dan ROC-AUC untuk menilai performa klasifikasi secara menyeluruh.
            </div>
            """,
            unsafe_allow_html=True,
        )
        end_card()

elif menu == "Regression Performance":
    show_section_title("Regression Model Performance")

    reg_models = metrics_bundle.get("regression_models", {})
    best_reg_name = metrics_bundle.get("best_regression_model")

    if not reg_models:
        st.warning("Metrics regresi belum tersedia. Jalankan ulang training model terlebih dahulu.")
        st.code("python -m src.train_model")
    else:
        selected_reg_model = st.selectbox("Pilih model regresi", list(reg_models.keys()))
        selected_reg_metrics = reg_models.get(selected_reg_model, {})

        if best_reg_name:
            st.success(f"Best Regression Model: {best_reg_name}")

        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{selected_reg_metrics.get('r2_score', 0):.4f}")
        c2.metric("MAE", f"{selected_reg_metrics.get('mae', 0):.4f}")
        c3.metric("RMSE", f"{selected_reg_metrics.get('rmse', 0):.4f}")

        start_card()
        show_section_title("Regression Summary")
        st.markdown(
            f"""
            <div class="subtle-text">
                Model <strong>{selected_reg_model}</strong> dievaluasi menggunakan R² Score, MAE, dan RMSE.
                Target penugasan adalah memperoleh R² Score minimal 0.8 dengan error sekecil mungkin.
            </div>
            """,
            unsafe_allow_html=True,
        )
        end_card()

elif menu == "Prediction":
    show_section_title("Prediction")

    if classification_model is None or regression_model is None:
        st.error("Model belum tersedia. Jalankan training model terlebih dahulu.")
        st.code("python -m src.train_model")
    else:
        prediction_type = st.selectbox(
            "Pilih jenis prediksi",
            ["Classification Prediction", "Regression Prediction"],
        )

        start_card()
        st.markdown(
            """
            <div class="subtle-text">
                Masukkan parameter siswa di bawah ini untuk melakukan prediksi.
            </div>
            """,
            unsafe_allow_html=True,
        )
        end_card()

        motivation_options = get_sorted_unique_values(df, "Motivation_Level")
        internet_options = get_sorted_unique_values(df, "Internet_Access")
        family_income_options = get_sorted_unique_values(df, "Family_Income")
        teacher_quality_options = get_sorted_unique_values(df, "Teacher_Quality")
        school_type_options = get_sorted_unique_values(df, "School_Type")
        parental_involvement_options = get_sorted_unique_values(df, "Parental_Involvement")
        access_resources_options = get_sorted_unique_values(df, "Access_to_Resources")
        extracurricular_options = get_sorted_unique_values(df, "Extracurricular_Activities")
        gender_options = get_sorted_unique_values(df, "Gender")
        peer_influence_options = get_sorted_unique_values(df, "Peer_Influence")
        parental_education_options = get_sorted_unique_values(df, "Parental_Education_Level")
        learning_disabilities_options = get_sorted_unique_values(df, "Learning_Disabilities")

        col1, col2 = st.columns(2)

        with col1:
            hours_studied = st.slider("Hours Studied", 0, 24, 5)
            attendance = st.slider("Attendance (%)", 0, 100, 75)
            sleep_hours = st.slider("Sleep Hours", 0, 12, 7)
            previous_scores = st.slider("Previous Scores", 0, 100, 70)
            tutoring_sessions = st.slider("Tutoring Sessions", 0, 20, 2)
            physical_activity = st.slider("Physical Activity", 0, 10, 3)
            distance_from_home = st.slider("Distance from Home", 0, 50, 10)

        with col2:
            motivation_level = st.selectbox("Motivation Level", motivation_options)
            internet_access = st.selectbox("Internet Access", internet_options)
            family_income = st.selectbox("Family Income", family_income_options)
            teacher_quality = st.selectbox("Teacher Quality", teacher_quality_options)
            school_type = st.selectbox("School Type", school_type_options)
            parental_involvement = st.selectbox("Parental Involvement", parental_involvement_options)
            access_to_resources = st.selectbox("Access to Resources", access_resources_options)
            extracurricular_activities = st.selectbox("Extracurricular Activities", extracurricular_options)
            gender = st.selectbox("Gender", gender_options)
            peer_influence = st.selectbox("Peer Influence", peer_influence_options)
            parental_education_level = st.selectbox("Parental Education Level", parental_education_options)
            learning_disabilities = st.selectbox("Learning Disabilities", learning_disabilities_options)

        input_df = pd.DataFrame(
            [
                {
                    "Hours_Studied": hours_studied,
                    "Attendance": attendance,
                    "Sleep_Hours": sleep_hours,
                    "Previous_Scores": previous_scores,
                    "Tutoring_Sessions": tutoring_sessions,
                    "Physical_Activity": physical_activity,
                    "Distance_from_Home": distance_from_home,
                    "Motivation_Level": motivation_level,
                    "Internet_Access": internet_access,
                    "Family_Income": family_income,
                    "Teacher_Quality": teacher_quality,
                    "School_Type": school_type,
                    "Parental_Involvement": parental_involvement,
                    "Access_to_Resources": access_to_resources,
                    "Extracurricular_Activities": extracurricular_activities,
                    "Gender": gender,
                    "Peer_Influence": peer_influence,
                    "Parental_Education_Level": parental_education_level,
                    "Learning_Disabilities": learning_disabilities,
                }
            ]
        )

        start_card()
        show_section_title("Input Preview")
        st.dataframe(input_df, use_container_width=True)
        end_card()

        if st.button("Run Prediction"):
            try:
                start_card()
                show_section_title("Prediction Result")

                if prediction_type == "Classification Prediction":
                    pred = int(classification_model.predict(input_df)[0])

                    probability = None
                    if hasattr(classification_model, "predict_proba"):
                        probability = float(classification_model.predict_proba(input_df)[0][1])

                    if probability is not None:
                        st.progress(min(max(probability, 0.0), 1.0))
                        st.write(f"Probability High Performance: **{probability:.4f}**")

                    st.write(f"Predicted Class: **{pred}**")

                    if pred == 1:
                        st.success("Prediction: High Performance")
                    else:
                        st.error("Prediction: Low Performance")

                else:
                    predicted_score = float(regression_model.predict(input_df)[0])
                    st.metric("Predicted Exam Score", f"{predicted_score:.2f}")

                    if predicted_score >= 75:
                        st.success("Estimated Category: High Performance")
                    else:
                        st.error("Estimated Category: Low Performance")

                end_card()

            except Exception as e:
                st.error(f"Terjadi error saat prediksi: {e}")

st.markdown(
    '<div class="footer-note">Built with Streamlit for Final Project Deployment</div>',
    unsafe_allow_html=True,
)