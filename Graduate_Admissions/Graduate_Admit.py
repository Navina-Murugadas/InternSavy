import streamlit as st
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('model.pkl','rb'))

st.title('PREDICTION OF GRADUATE ADMISSIONS FROM AN INDIAN PERSPECTIVE')

# Collect user input for prediction
gre = st.slider("GRE Score", 260, 340, 200)
toefl = st.slider("TOEFL Score", 90, 120, 90)
university_rating = st.slider("University Rating", 1, 5, 1)
sop = st.slider("Statement of Purpose (SOP)", 1.0, 5.0, 1.5)
lor = st.slider("Letter of Recommendation (LOR)", 1.0, 5.0, 1.5)
cgpa = st.number_input('CGPA')
research = st.radio("Research", ("Yes", "No"))
research = 1 if research == "Yes" else 0

def predict_admission_chance(gre, toefl, university_rating, sop, lor, cgpa, research):
    user_data = np.array([gre, toefl, university_rating, sop, lor, cgpa, research]).reshape(1, -1)
    prediction = model.predict(user_data)

    if prediction[0] >= 0.8:
        return "HIGH CHANCES OF ADMIT"
    elif 0.6 <= prediction[0] < 0.8:
        return "MODERATE CHANCES OF ADMIT"
    else:
        return "LOW CHANCES OF ADMIT"

if st.button('PREDICT'):
    chance_of_admit = predict_admission_chance(gre, toefl, university_rating, sop, lor, cgpa, research)
    st.success(f"Chance of Admission: {chance_of_admit}")

st.write("NOTE: This app predicts the chance of admission based on a LINEAR REGRESSION model.")
