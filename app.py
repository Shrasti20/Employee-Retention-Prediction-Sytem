# ======================================
# STREAMLIT APP
# Employee Retention Prediction
# ======================================

import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL & PREPROCESSORS
# ===============================
model = joblib.load("final_xgboost_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="Employee Retention Predictor", layout="centered")
st.title("👨‍💼 Employee Retention Prediction App")
st.markdown("Predict whether an employee will **leave or stay** in the company.")

# ===============================
# INPUT FORM
# ===============================
st.subheader("📋 Employee Details")

# ---- Numerical Inputs ----
city_development_index = st.number_input("City Development Index", 0.0, 1.0, 0.5)
training_hours = st.number_input("Training Hours", 0, 500, 50)

# ---- Categorical Inputs (Dropdowns) ----
gender = st.selectbox("Gender", label_encoders["gender"].classes_)
relevent_experience = st.selectbox("Relevant Experience", label_encoders["relevent_experience"].classes_)
enrolled_university = st.selectbox("Enrolled University", label_encoders["enrolled_university"].classes_)
education_level = st.selectbox("Education Level", label_encoders["education_level"].classes_)
major_discipline = st.selectbox("Major Discipline", label_encoders["major_discipline"].classes_)
experience = st.selectbox("Experience", label_encoders["experience"].classes_)
company_size = st.selectbox("Company Size", label_encoders["company_size"].classes_)
company_type = st.selectbox("Company Type", label_encoders["company_type"].classes_)
last_new_job = st.selectbox("Last New Job", label_encoders["last_new_job"].classes_)
city = st.selectbox("City", label_encoders["city"].classes_)

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("🔮 Predict Retention"):

    # Create dataframe in SAME ORDER as training
    input_df = pd.DataFrame({
        "city": [city],
        "city_development_index": [city_development_index],
        "gender": [gender],
        "relevent_experience": [relevent_experience],
        "enrolled_university": [enrolled_university],
        "education_level": [education_level],
        "major_discipline": [major_discipline],
        "experience": [experience],
        "company_size": [company_size],
        "company_type": [company_type],
        "last_new_job": [last_new_job],
        "training_hours": [training_hours]
    })

    # Encode categorical columns
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])

    # Scale numerical columns
    numerical_cols = ["city_development_index", "training_hours"]
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Output
    # st.subheader("📊 Prediction Result")


    st.code(int(prediction))

   
