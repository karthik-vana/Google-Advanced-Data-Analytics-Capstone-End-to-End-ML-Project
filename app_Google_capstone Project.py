# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Try to import xgboost; if not available, we'll skip XGBoost model
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("Employee Attrition Predictor — Streamlit")
st.markdown(
    """
Train & predict Employee Attrition using Random Forest, Logistic Regression, and XGBoost.
- Drop your dataset as `data.csv` in the same folder, or the app will use a public IBM HR dataset as fallback.
- The app trains models on load and provides an interactive prediction form.
"""
)
def load_data():
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload your employee attrition dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ Dataset uploaded successfully!")
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
            st.stop()
    else:
        # Fallback dataset (IBM HR Analytics)
        fallback_url = (
            "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
        )
        try:
            df = pd.read_csv(fallback_url)
            st.sidebar.info("Using fallback IBM HR dataset (no file uploaded).")
            return df
        except Exception as e:
            st.sidebar.error("❌ Failed to load fallback dataset. Please upload your CSV file.")
            st.stop()


def prepare_features(df):
    # Minimal cleaning: drop columns that are IDs or almost-unique
    df = df.copy()
    # Standard IBM dataset column for target: 'Attrition' (Yes/No)
    # Detect the target column automatically
possible_targets = ["Attrition", "attrition", "left", "Left", "employee_left", "EmployeeLeft", "Attrition_Flag"]
target_col = None
for col in df.columns:
    if col.strip() in possible_targets:
        target_col = col
        break

if target_col is None:
    st.error(
        "❌ Could not find target column. Please ensure your dataset has one of these columns: "
        "`Attrition`, `left`, `Attrition_Flag`, or similar."
    )
    st.stop()

st.sidebar.success(f"Detected target column: **{target_col}**")

# Convert target to binary 1/0
y = df[target_col].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true', 'left'] else 0)

    # Example set of features to use — adaptable to dataset columns available
    numeric_features = [
        f for f in [
            "Age", "DistanceFromHome", "MonthlyIncome", "PercentSalaryHike",
            "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
            "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked",
            "TrainingTimesLastYear"
        ] if f in df.columns
    ]

    categorical_features = [
        f for f in [
            "BusinessTravel", "Department", "EducationField", "Gender",
            "JobRole", "MaritalStatus", "OverTime"
        ] if f in df.columns
    ]

    # Reduce cardinality for any category with many levels? (skip for now)

    # Fill missing values (simple approach)
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())
    df[categorical_features] = df[categorical_features].fillna("Unknown")

    X = df[numeric_features + categorical_features]
    y = df['Attrition'].apply(lambda x: 1 if str(x).strip().lower() == 'yes' else 0)

    return X, y, numeric_features, categorical_features

@st.cache_data(show_spinner=False)
def build_preprocessor(numeric_features, categorical_features):
    num_pipe = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features)
    ], remainder='drop')

    return preprocessor

@st.cache_resource(show_spinner=False)
def train_models(X, y, numeric_features, categorical_features, random_state=42):
    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Pipelines
    rf_pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=random_state, n_jobs=-1))
    ])

    lr_pipe = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=1000))
    ])

    models = {"Random Forest": rf_pipe, "Logistic Regression": lr_pipe}

    if XGBOOST_AVAILABLE:
        xgb_pipe = Pipeline(steps=[
            ("pre", preprocessor),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", verbosity=0))
        ])
        models["XGBoost"] = xgb_pipe

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=random_state)

    metrics = {}
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        acc = accuracy_score(y_test, preds)
        metrics[name] = {
            "model": pipe,
            "accuracy": acc,
            "classification_report": classification_report(y_test, preds, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, preds)
        }
        # Save each trained model
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipe, f"models/{name.replace(' ', '_').lower()}_pipeline.joblib")

    return metrics, X_train, X_test, y_train, y_test

def show_metrics(metrics):
    st.header("Model performance on test set")
    for name, info in metrics.items():
        st.subheader(name)
        st.write(f"Accuracy: **{info['accuracy']:.4f}**")
        cr_df = pd.DataFrame(info["classification_report"]).transpose()
        st.dataframe(cr_df.style.format(precision=3))
        cm = info["confusion_matrix"]
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} — Confusion Matrix")
        st.pyplot(fig)
        plt.close(fig)

def build_input_form(numeric_features, categorical_features, example_df=None):
    st.sidebar.header("Prediction input")
    inputs = {}
    # numeric inputs with sensible ranges from example_df if provided
    for col in numeric_features:
        if example_df is not None and col in example_df.columns:
            col_min = int(example_df[col].min())
            col_max = int(example_df[col].max())
            col_mean = float(example_df[col].median())
            inputs[col] = st.sidebar.slider(col, min_value=col_min, max_value=col_max, value=int(col_mean))
        else:
            inputs[col] = st.sidebar.number_input(col, value=0)

    for col in categorical_features:
        choices = None
        if example_df is not None and col in example_df.columns:
            choices = sorted(example_df[col].dropna().unique().tolist())
        choice = st.sidebar.selectbox(col, options=choices if choices else ["Unknown"])
        inputs[col] = choice

    return pd.DataFrame([inputs])

def predict_and_display(model_pipeline, input_df):
    proba = model_pipeline.predict_proba(input_df)[0][1] if hasattr(model_pipeline, "predict_proba") else None
    pred = model_pipeline.predict(input_df)[0]
    label = "Yes (Attrition)" if pred == 1 else "No (No Attrition)"
    st.markdown("### Prediction")
    st.write(f"**Predicted class:** {label}")
    if proba is not None:
        st.write(f"**Probability of attrition:** {proba*100:.2f}%")

def main():
    df = load_data()
    st.subheader("Data preview")
    st.dataframe(df.head())

    X, y, numeric_features, categorical_features = prepare_features(df)

    st.sidebar.header("Training options")
    st.sidebar.write(f"Detected numeric features: {numeric_features}")
    st.sidebar.write(f"Detected categorical features: {categorical_features}")

    # Train models
    with st.spinner("Training models... this runs once and caches."):
        metrics, X_train, X_test, y_train, y_test = train_models(X, y, numeric_features, categorical_features)

    # Show metrics & plots
    show_metrics(metrics)

    # Model selection for prediction
    available_models = list(metrics.keys())
    st.sidebar.header("Prediction model")
    model_choice = st.sidebar.selectbox("Choose model", available_models)

    # Build input form using train data info
    example_df = pd.concat([X_train, X_test]).reset_index(drop=True)
    input_df = build_input_form(numeric_features, categorical_features, example_df=example_df)

    if st.sidebar.button("Predict"):
        # Need to use same pipeline preprocessing: each pipeline expects original feature columns order
        model_pipe = metrics[model_choice]["model"]
        # predict expects same columns as X (with numeric+categorical cols)
        # ensure input_df columns match order
        input_df = input_df[numeric_features + categorical_features]
        predict_and_display(model_pipe, input_df)

    # Allow saving/loading models
    st.sidebar.markdown("---")
    if st.sidebar.button("Download trained pipelines (zip)"):
        st.info("Trained pipelines are saved in the `models/` folder on the server instance (or locally when running). Use `joblib.load` to load them in production.")

    st.write("### Notes")
    st.markdown(
        """
- If you want to use your own dataset: place it as `data.csv` in the same folder as `app.py`. The dataset must include the target column `Attrition` with values `Yes`/`No`.
- Trained pipelines are saved as `models/<model_name>_pipeline.joblib`.
- For production, you may want to train offline and deploy only the saved pipeline to reduce startup time.
"""
    )

if __name__ == "__main__":
    main()


# --- Footer ---
st.markdown("---")
st.markdown("""
✅ Created by **Vana Karthik**  
🎓 [Verified Google Capstone Certificate](https://coursera.org/verify/Z9F0WG7HRN9W)  
💻 Repository: [GitHub](https://github.com/<your-username>/Google-Advanced-Data-Analytics-Capstone)
""")
