# streamlit_app.py

import streamlit as st
import numpy as np
import joblib

# Load the trained model with probability enabled
svm_model = joblib.load('svm_cancer_model.pkl')

# Define the feature names in the same order as used during training
feature_names = ['age', 'gender', 'air_pollution', 'alcohol_use', 'dust_allergy',
                 'occupational_hazards', 'genetic_risk', 'chronic_lung_disease',
                 'balanced_diet', 'obesity', 'smoking', 'passive_smoker', 
                 'chest_pain', 'coughing_of_blood', 'fatigue', 'weight_loss',
                 'shortness_of_breath', 'wheezing', 'swallowing_difficulty',
                 'clubbing_of_finger_nails', 'frequent_cold', 'dry_cough', 'snoring']

# Create a mapping for the Gender input
gender_mapping = {"Male": 1, "Female": 0}

# Title of the Streamlit app
st.title("Cancer Level Prediction")

# Input fields for the user to provide values for all features
st.header("Input Patient Data")

age = st.number_input("Age", min_value=0, max_value=100, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
air_pollution = st.slider("Air Pollution Level (0-10)", min_value=0, max_value=10, value=5)
alcohol_use = st.slider("Alcohol Use (0-10)", min_value=0, max_value=10, value=3)
dust_allergy = st.slider("Dust Allergy (0-10)", min_value=0, max_value=10, value=2)
occupational_hazards = st.slider("Occupational Hazards (0-10)", min_value=0, max_value=10, value=4)
genetic_risk = st.slider("Genetic Risk (0-10)", min_value=0, max_value=10, value=7)
chronic_lung_disease = st.slider("Chronic Lung Disease (0-10)", min_value=0, max_value=10, value=6)
balanced_diet = st.slider("Balanced Diet (0-10)", min_value=0, max_value=10, value=8)
obesity = st.slider("Obesity (0-10)", min_value=0, max_value=10, value=5)
smoking = st.slider("Smoking (0-10)", min_value=0, max_value=10, value=6)
passive_smoker = st.slider("Passive Smoker (0-10)", min_value=0, max_value=10, value=4)
chest_pain = st.slider("Chest Pain (0-10)", min_value=0, max_value=10, value=5)
coughing_of_blood = st.slider("Coughing of Blood (0-10)", min_value=0, max_value=10, value=1)
fatigue = st.slider("Fatigue (0-10)", min_value=0, max_value=10, value=6)
weight_loss = st.slider("Weight Loss (0-10)", min_value=0, max_value=10, value=7)
shortness_of_breath = st.slider("Shortness of Breath (0-10)", min_value=0, max_value=10, value=8)
wheezing = st.slider("Wheezing (0-10)", min_value=0, max_value=10, value=4)
swallowing_difficulty = st.slider("Swallowing Difficulty (0-10)", min_value=0, max_value=10, value=3)
clubbing_of_finger_nails = st.slider("Clubbing of Finger Nails (0-10)", min_value=0, max_value=10, value=1)
frequent_cold = st.slider("Frequent Cold (0-10)", min_value=0, max_value=10, value=2)
dry_cough = st.slider("Dry Cough (0-10)", min_value=0, max_value=10, value=3)
snoring = st.slider("Snoring (0-10)", min_value=0, max_value=10, value=2)

# Prepare the input data for prediction
input_data = np.array([[age, gender_mapping[gender], air_pollution, alcohol_use, dust_allergy,
                        occupational_hazards, genetic_risk, chronic_lung_disease, balanced_diet,
                        obesity, smoking, passive_smoker, chest_pain, coughing_of_blood, fatigue,
                        weight_loss, shortness_of_breath, wheezing, swallowing_difficulty, 
                        clubbing_of_finger_nails, frequent_cold, dry_cough, snoring]])

# Button for prediction
if st.button("Predict Cancer Level"):
    # Perform prediction
    prediction = svm_model.predict(input_data)
    prediction_proba = svm_model.predict_proba(input_data)

    # Output the result based on the model's prediction
    cancer_level_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
    result = cancer_level_mapping[prediction[0]]
    st.success(f"The predicted cancer level is: {result}")

    # Display probability rates for each class (Low, Medium, High)
    st.write("### Probability rates:")
    st.write(f"Low: {prediction_proba[0][0]:.2f}")
    st.write(f"Medium: {prediction_proba[0][1]:.2f}")
    st.write(f"High: {prediction_proba[0][2]:.2f}")

# Sidebar information about the app
st.sidebar.title("About")
st.sidebar.info("""
    This app predicts the cancer level (Low, Medium, High) based on patient data.
    The model is built using Support Vector Machines (SVM) trained on a dataset
    of cancer patients.
""")
