import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="📊")

MODEL_FILES = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "SVM": "models/svm.joblib"
}

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

st.title("Bank Marketing Classification App")
st.write(
    "Upload a test CSV containing the 19 input features and the target column `y`. "
    "The selected saved model will evaluate the data without retraining."
)

selected_model_name = st.selectbox("Choose a model", list(MODEL_FILES.keys()))
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    if "y" not in test_data.columns:
        st.error("The uploaded CSV must contain the target column named 'y'.")
    else:
        model = load_model(MODEL_FILES[selected_model_name])

        X_uploaded = test_data.drop(columns="y")
        y_actual = test_data["y"]

        expected_columns = list(model.feature_names_in_)
        missing_columns = set(expected_columns) - set(X_uploaded.columns)

        if missing_columns:
            st.error(f"Missing required columns: {sorted(missing_columns)}")
        else:
            X_uploaded = X_uploaded[expected_columns]
            y_pred = model.predict(X_uploaded)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_uploaded)[:, 1]
            else:
                y_score = model.decision_function(X_uploaded)

            metrics = pd.DataFrame([{
                "Accuracy": accuracy_score(y_actual, y_pred),
                "AUC": roc_auc_score(y_actual, y_score),
                "Precision": precision_score(y_actual, y_pred, zero_division=0),
                "Recall": recall_score(y_actual, y_pred, zero_division=0),
                "F1-score": f1_score(y_actual, y_pred, zero_division=0),
                "MCC": matthews_corrcoef(y_actual, y_pred)
            }])

            st.subheader(f"Evaluation: {selected_model_name}")
            st.dataframe(metrics.round(4), use_container_width=True)

            matrix = confusion_matrix(y_actual, y_pred)

            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(
                matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=["No", "Yes"],
                yticklabels=["No", "Yes"],
                ax=ax
            )
            ax.set_xlabel("Predicted label")
            ax.set_ylabel("Actual label")
            ax.set_title(f"Confusion Matrix - {selected_model_name}")
            st.pyplot(fig)

            report = classification_report(
                y_actual,
                y_pred,
                target_names=["No", "Yes"],
                output_dict=True
            )

            st.subheader("Classification Report")
            st.dataframe(pd.DataFrame(report).transpose().round(4))
