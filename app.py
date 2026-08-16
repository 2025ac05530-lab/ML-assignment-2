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
