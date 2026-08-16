import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, roc_auc_score,
                             confusion_matrix, classification_report)

st.set_page_config(page_title="Credit Default Prediction", layout="wide")


st.title("Credit Card Default Prediction")
st.markdown(""" Name- Shruti Basavaraj Desai
ID- 2025AC05530 

Given a credit card client's demographic profile, credit limit, six months of
repayment status, six months of bill statement amounts and six months of payment
amounts, predict whether that client will default on their payment next month.
This is a binary classification problem framed from the lender's point of view.
The practical use is early risk flagging: a client predicted to default can be
routed to collections outreach, a credit limit review, or a restructured payment
plan before the missed payment actually happens.""")


MODEL_FILES = {
    'Logistic Regression': 'model/saved_models/logistic_regression.pkl',
    'Decision Tree': 'model/saved_models/decision_tree.pkl',
    'kNN': 'model/saved_models/knn.pkl',
    'Naive Bayes': 'model/saved_models/naive_bayes.pkl',
    'Random Forest': 'model/saved_models/random_forest.pkl'
}

selected_model_name = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))

model = joblib.load(MODEL_FILES[selected_model_name])
feature_columns = joblib.load('model/saved_models/feature_columns.pkl')

st.sidebar.write("Currently loaded:", selected_model_name)

uploaded_file = st.file_uploader("Upload test data (CSV)", type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    st.subheader("Data Preview")
    st.dataframe(df.head())
    TARGET = 'default_payment_next_month'

    if TARGET not in df.columns:
        st.error(f"Uploaded file must contain the column '{TARGET}'")
        st.stop()

    y_true = df[TARGET]
    X = df.drop(columns=[TARGET])
    X = X[feature_columns]
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    prec = precision_score(y_true, y_pred)
    recall=recall_score(y_true, y_pred)
    mcc =matthews_corrcoef(y_true, y_pred)
    fc= f1_score(y_true, y_pred)
    st.subheader("Evaluation Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6) 
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{prec:.4f}")
    col4.metric("Recall", f"{recall:.4f}")
    col6.metric("F1_Score", f"{fc:.4f}")
    col5.metric("MCC", f"{mcc:.4f}")
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))