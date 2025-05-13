# Full Streamlit App with Model Loading, Prediction, SHAP Explainability, and Accuracy

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load model and sample training data for SHAP (assume saved during training phase)
model = joblib.load("best_model.pkl")
X_train = pd.read_csv("train_features.csv")
y_train = pd.read_csv("train_targets.csv")

st.set_page_config(page_title="Student Grade Predictor", layout="wide")
st.title("Student Grade Predictor (11th & 12th)")

# Input Form
st.sidebar.header("Student Profile")
name = st.sidebar.text_input("Student Name")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
effort = st.sidebar.selectbox("Effort", ["Low", "Middle", "High"])

grades_input = {}
with st.expander("Enter Grades from 6th to 10th", expanded=True):
    for grade in [6, 7, 8, 9, 10]:
        st.subheader(f"Grade {grade} Marks")
        for subject in ['Math', 'English', 'Science', 'History', 'Computer']:
            key = f"{subject}_Grade_{grade}"
            grades_input[key] = st.slider(f"{subject} (Grade {grade})", 0, 100, 75)

# Predict
if st.button("Predict 11th & 12th Marks"):
    input_df = pd.DataFrame([grades_input])

    # Add encoded categorical variables
    input_df['Gender_Female'] = 1 if gender == 'Female' else 0
    input_df['Gender_Male'] = 1 if gender == 'Male' else 0
    input_df['Socioeconomic_Status_Low'] = 1 if effort == 'Low' else 0
    input_df['Socioeconomic_Status_Middle'] = 1 if effort == 'Middle' else 0
    input_df['Socioeconomic_Status_High'] = 1 if effort == 'High' else 0

    # Ensure same column order
    input_df = input_df[X_train.columns]

    # Predict all subjects' 11th & 12th grade marks (assuming multioutput model)
    predictions = model.predict(input_df)[0]

    subjects = ['Math', 'English', 'Science', 'History', 'Computer']
    st.subheader("🎯 Predicted Marks:")
    for i, subject in enumerate(subjects):
        st.write(f"**{subject} Grade 11**: {predictions[i]:.2f}")
        st.write(f"**{subject} Grade 12**: {predictions[i+len(subjects)]:.2f}")

    # Model Evaluation on training data (optional)
    y_pred = model.predict(X_train)
    # rmse = mean_squared_error(y_train, y_pred, squared=False)
    rmse = np.sqrt(mean_squared_error(y_train, y_pred))  # Manual RMSE

    mae = mean_absolute_error(y_train, y_pred)
    st.info(f"Model RMSE on Train Data: {rmse:.2f}")
    st.info(f"Model MAE on Train Data: {mae:.2f}")

    # SHAP Explainability
    st.subheader("🔍 Feature Importance using SHAP")
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(input_df)

    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    st.pyplot(fig)

    st.caption("This SHAP plot shows how each input feature contributed to the predictions.")
