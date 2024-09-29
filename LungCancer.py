# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the trained SVM model and scaler
# svm_model = joblib.load('svm_lung_cancer_model.joblib')
# scaler = joblib.load('scaler.joblib')

# # Define the app title
# st.title("Lung Cancer Prediction App")

# # Sidebar for user inputs
# st.sidebar.header("Input Features")

# def user_input_features():
#     gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
#     age = st.sidebar.number_input('Enter your age:', min_value=1, max_value=100, value=25)
#     smoking = st.sidebar.selectbox('Smoking Habit', ['No', 'Yes'])
#     yellow_fingers = st.sidebar.selectbox('Yellow Fingers', ['No', 'Yes'])
#     anxiety = st.sidebar.selectbox('Anxiety', ['No', 'Yes'])
#     peer_pressure = st.sidebar.selectbox('Peer Pressure', ['No', 'Yes'])
#     chronic_disease = st.sidebar.selectbox('Chronic Disease', ['No', 'Yes'])
#     fatigue = st.sidebar.selectbox('Fatigue', ['No', 'Yes'])
#     allergy = st.sidebar.selectbox('Allergy', ['No', 'Yes'])
#     wheezing = st.sidebar.selectbox('Wheezing', ['No', 'Yes'])
#     alcohol_consuming = st.sidebar.selectbox('Alcohol Consuming', ['No', 'Yes'])
#     coughing = st.sidebar.selectbox('Coughing', ['No', 'Yes'])
#     shortness_of_breath = st.sidebar.selectbox('Shortness of Breath', ['No', 'Yes'])
#     swallowing_difficulty = st.sidebar.selectbox('Swallowing Difficulty', ['No', 'Yes'])
#     chest_pain = st.sidebar.selectbox('Chest Pain', ['No', 'Yes'])

#     # Create a dictionary with user inputs
#     data = {
#         'GENDER': 1 if gender == 'Male' else 0,
#         'AGE': age,
#         'SMOKING': 1 if smoking == 'Yes' else 0,
#         'YELLOW_FINGERS': 1 if yellow_fingers == 'Yes' else 0,
#         'ANXIETY': 1 if anxiety == 'Yes' else 0,
#         'PEER_PRESSURE': 1 if peer_pressure == 'Yes' else 0,
#         'CHRONIC DISEASE': 1 if chronic_disease == 'Yes' else 0,
#         'FATIGUE ': 1 if fatigue == 'Yes' else 0,  # Include trailing space
#         'ALLERGY ': 1 if allergy == 'Yes' else 0,  # Include trailing space
#         'WHEEZING': 1 if wheezing == 'Yes' else 0,
#         'ALCOHOL CONSUMING': 1 if alcohol_consuming == 'Yes' else 0,  # Match exact name
#         'COUGHING': 1 if coughing == 'Yes' else 0,
#         'SHORTNESS OF BREATH': 1 if shortness_of_breath == 'Yes' else 0,
#         'SWALLOWING DIFFICULTY': 1 if swallowing_difficulty == 'Yes' else 0,
#         'CHEST PAIN': 1 if chest_pain == 'Yes' else 0
#     }

#     features = pd.DataFrame(data, index=[0])
#     return features


# # Get user input
# input_df = user_input_features()

# # Display user input
# st.subheader("User Input Features")
# st.write(input_df)

# # # Scale the user input data
# # input_scaled = scaler.transform(input_df)

# # Make prediction using decision function
# decision_score = svm_model.decision_function(input_df)

# # Set the threshold
# threshold = 0.0 # Adjust if necessary

# # Make prediction based on decision scores and the defined threshold
# if decision_score[0] > threshold:
#     prediction = 1  # Predict Positive for Lung Cancer
# else:
#     prediction = 0  # Predict Negative for Lung Cancer

# # Display prediction result
# st.subheader("Prediction Result")

# # Calculate probability from the decision score
# probability = 1 / (1 + np.exp(-decision_score[0]))  # Sigmoid function to get probability

# # Display the result and probability score
# result = "Positive for Lung Cancer" if prediction == 1 else "Negative for Lung Cancer"
# st.write(result)
# st.write(f"Probability score: {probability:.2f}")

# # Visualization Section for feature values
# st.subheader("Feature Values of User Input")
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(x=input_df.columns, y=input_df.iloc[0], ax=ax, color='blue')
# ax.set_title('Feature Values of User Input')
# plt.xticks(rotation=45)
# st.pyplot(fig)

# # Footer
# st.write("This is a simple SVM-based lung cancer prediction app. The model's accuracy may vary depending on the dataset used for training.")










# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the trained SVM model and scaler
# svm_model = joblib.load('svm_lung_cancer_model_best.joblib')
# scaler = joblib.load('scaler.joblib')

# # Define the app title
# st.title("Lung Cancer Prediction App")

# # Sidebar for user inputs
# st.sidebar.header("Input Features")

# # Function to take user input from the sidebar
# def user_input_features():
#     gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
#     age = st.sidebar.number_input('Enter your age:', min_value=1, max_value=100, value=25)
#     smoking = st.sidebar.selectbox('Smoking Habit', ['No', 'Yes'])
#     yellow_fingers = st.sidebar.selectbox('Yellow Fingers', ['No', 'Yes'])
#     anxiety = st.sidebar.selectbox('Anxiety', ['No', 'Yes'])
#     peer_pressure = st.sidebar.selectbox('Peer Pressure', ['No', 'Yes'])
#     chronic_disease = st.sidebar.selectbox('Chronic Disease', ['No', 'Yes'])
#     fatigue = st.sidebar.selectbox('Fatigue', ['No', 'Yes'])
#     allergy = st.sidebar.selectbox('Allergy', ['No', 'Yes'])
#     wheezing = st.sidebar.selectbox('Wheezing', ['No', 'Yes'])
#     alcohol_consuming = st.sidebar.selectbox('Alcohol Consuming', ['No', 'Yes'])
#     coughing = st.sidebar.selectbox('Coughing', ['No', 'Yes'])
#     shortness_of_breath = st.sidebar.selectbox('Shortness of Breath', ['No', 'Yes'])
#     swallowing_difficulty = st.sidebar.selectbox('Swallowing Difficulty', ['No', 'Yes'])
#     chest_pain = st.sidebar.selectbox('Chest Pain', ['No', 'Yes'])

#     # Create a dictionary with user inputs
#     data = {
#         'GENDER': 1 if gender == 'Male' else 0,
#         'AGE': age,
#         'SMOKING': 1 if smoking == 'Yes' else 0,
#         'YELLOW_FINGERS': 1 if yellow_fingers == 'Yes' else 0,
#         'ANXIETY': 1 if anxiety == 'Yes' else 0,
#         'PEER_PRESSURE': 1 if peer_pressure == 'Yes' else 0,
#         'CHRONIC DISEASE': 1 if chronic_disease == 'Yes' else 0,
#         'FATIGUE ': 1 if fatigue == 'Yes' else 0,  # Include trailing space
#         'ALLERGY ': 1 if allergy == 'Yes' else 0,  # Include trailing space
#         'WHEEZING': 1 if wheezing == 'Yes' else 0,
#         'ALCOHOL CONSUMING': 1 if alcohol_consuming == 'Yes' else 0,  # Match exact name
#         'COUGHING': 1 if coughing == 'Yes' else 0,
#         'SHORTNESS OF BREATH': 1 if shortness_of_breath == 'Yes' else 0,
#         'SWALLOWING DIFFICULTY': 1 if swallowing_difficulty == 'Yes' else 0,
#         'CHEST PAIN': 1 if chest_pain == 'Yes' else 0
#     }

#     features = pd.DataFrame(data, index=[0])
#     return features


# # Get user input
# input_df = user_input_features()

# # Display user input
# st.subheader("User Input Features")
# st.write(input_df)

# # Scale the user input data
# input_scaled = scaler.transform(input_df)

# # Make prediction using predict_proba method to get probabilities
# probabilities = svm_model.predict_proba(input_scaled)

# # Use the positive class probability (lung cancer = 1)
# positive_class_prob = probabilities[0][1]

# # Set the threshold based on classification report
# threshold = 0.0  # Adjust if necessary to match the classification report

# # Make prediction based on probability and the defined threshold
# if positive_class_prob > threshold:
#     prediction = 1  # Predict Positive for Lung Cancer
# else:
#     prediction = 0  # Predict Negative for Lung Cancer

# # Display prediction result
# st.subheader("Prediction Result")

# # Display the result and probability score
# result = "Positive for Lung Cancer" if prediction == 1 else "Negative for Lung Cancer"
# st.write(result)
# st.write(f"Probability score: {positive_class_prob:.2f}")

# # Classification report details
# st.subheader("Model Performance Details")
# st.write("""
# ### Classification Report:
# - Precision for No Cancer (0): 0.71
# - Recall for No Cancer (0): 0.62
# - F1-Score for No Cancer (0): 0.67
# - Precision for Cancer (1): 0.95
# - Recall for Cancer (1): 0.96
# - F1-Score for Cancer (1): 0.95
# - **Overall Accuracy: 92%**
# """)

# # Visualization Section for feature values
# st.subheader("Feature Values of User Input")
# fig, ax = plt.subplots(figsize=(10, 5))
# sns.barplot(x=input_df.columns, y=input_df.iloc[0], ax=ax, color='blue')
# ax.set_title('Feature Values of User Input')
# plt.xticks(rotation=45)
# st.pyplot(fig)

# # Footer
# st.write("This is a simple SVM-based lung cancer prediction app. The model is trained on a balanced dataset with SMOTE oversampling.")












# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # Load the trained model and scaler
# model = joblib.load('best_lung_cancer_model.joblib')
# scaler = joblib.load('scaler.joblib')

# # Streamlit app
# st.title("Lung Cancer Prediction App")

# # Input fields for features
# st.header("Input Features")
# age = st.number_input("Age", min_value=0)
# gender = st.selectbox("Gender", ["Male", "Female"])
# smoking = st.selectbox("Smoking", ["Yes", "No"])
# yellow_fingers = st.selectbox("Yellow Fingers", ["Yes", "No"])
# anxiety = st.selectbox("Anxiety", ["Yes", "No"])
# peer_pressure = st.selectbox("Peer Pressure", ["Yes", "No"])
# chronic_disease = st.selectbox("Chronic Disease", ["Yes", "No"])
# fatigue = st.selectbox("Fatigue", ["Yes", "No"])
# allergy = st.selectbox("Allergy", ["Yes", "No"])
# wheezing = st.selectbox("Wheezing", ["Yes", "No"])
# alcohol_consuming = st.selectbox("Alcohol Consuming", ["Yes", "No"])
# coughing = st.selectbox("Coughing", ["Yes", "No"])
# shortness_of_breath = st.selectbox("Shortness of Breath", ["Yes", "No"])
# swallowing_difficulty = st.selectbox("Swallowing Difficulty", ["Yes", "No"])
# chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

# # Convert categorical inputs to numerical
# gender = 1 if gender == "Male" else 0
# smoking = 1 if smoking == "Yes" else 0
# yellow_fingers = 1 if yellow_fingers == "Yes" else 0
# anxiety = 1 if anxiety == "Yes" else 0
# peer_pressure = 1 if peer_pressure == "Yes" else 0
# chronic_disease = 1 if chronic_disease == "Yes" else 0
# fatigue = 1 if fatigue == "Yes" else 0
# allergy = 1 if allergy == "Yes" else 0
# wheezing = 1 if wheezing == "Yes" else 0
# alcohol_consuming = 1 if alcohol_consuming == "Yes" else 0
# coughing = 1 if coughing == "Yes" else 0
# shortness_of_breath = 1 if shortness_of_breath == "Yes" else 0
# swallowing_difficulty = 1 if swallowing_difficulty == "Yes" else 0
# chest_pain = 1 if chest_pain == "Yes" else 0

# # Create a DataFrame for the input
# input_data = pd.DataFrame([[age, gender, smoking, yellow_fingers, anxiety, peer_pressure,
#                              chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
#                              coughing, shortness_of_breath, swallowing_difficulty, chest_pain]],
#                            columns=['AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY',
#                                     'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE', 'ALLERGY',
#                                     'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
#                                     'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN'])

# # Standardize the input data
# input_data_scaled = scaler.transform(input_data)

# # Make prediction
# if st.button("Predict"):
#     prediction_proba = model.predict_proba(input_data_scaled)[:, 1]  # Get probability of lung cancer
#     prediction = model.predict(input_data_scaled)  # Get predicted class

#     st.write(f"Probability of Lung Cancer: {prediction_proba[0]:.2f}")
#     if prediction[0] == 1:
#         st.success("The model predicts: Lung Cancer (Positive)")
#     else:
#         st.success("The model predicts: No Lung Cancer (Negative)")























# import streamlit as st
# import pandas as pd
# import joblib
# from sklearn.preprocessing import StandardScaler

# # Load the trained model and scaler
# model = joblib.load('svm_model.pkl')
# scaler = StandardScaler()

# # Streamlit app
# st.title("Lung Cancer Prediction App")
# st.header("Enter the patient's information to predict lung cancer risk")

# # Input fields for user input data
# # age = st.number_input("Age", min_value=1, max_value=120, step=1)
# smokes = st.number_input("Smokes (number of cigarettes per day)", min_value=0, max_value=100, step=1)
# # areaQ = st.number_input("enter AREAQ)", min_value=0.0, max_value=10.0, step=0.1)
# alkhol = st.number_input("Alkhol (alcohol consumption per day)", min_value=0.0, max_value=10.0, step=0.2)

# # When the user clicks on the Predict button
# if st.button("Predict"):
#     # Create a DataFrame from the input
#     input_data = pd.DataFrame([[smokes, alkhol]], columns=[ 'Smokes','Alkhol'])

#     # Standardize the input data (assuming scaling was used during training)
#     input_data_scaled = scaler.fit_transform(input_data)  # Use fit_transform for simplicity, ideally load a scaler

#     # Make prediction using the trained model
#     prediction = model.predict(input_data_scaled)[0]
#     prediction_proba = model.predict_proba(input_data_scaled)[0]

#     # Display prediction
#     if prediction == 1:
#         st.write(f"Prediction: High risk of lung cancer (Probability: {prediction_proba[1]:.2f})")
#     else:
#         st.write(f"Prediction: Low risk of lung cancer (Probability: {prediction_proba[0]:.2f})")

 
 
 
 
 
 
 
 
# import streamlit as st
# import numpy as np
# import joblib
# from sklearn.preprocessing import StandardScaler

# # Load the trained model and scaler
# svm_model = joblib.load('svm_lung_cancer_model.pkl')
# # scaler = joblib.load('scaler.pkl')  # Make sure this scaler was saved during model training

# # Streamlit app title and description
# st.title("Lung Cancer Prediction App")
# st.write("Enter your details to predict the risk of lung cancer.")

# # Function to make predictions
# def predict_lung_cancer(input_data):
#     # Reshape input data to match the expected shape (1, n_features)
#     input_data_reshaped = np.array(input_data).reshape(1, -1)
    
#     # Scale the input data
#     # input_data_scaled = scaler.transform(input_data_reshaped)
    
#     # Predict using the SVM model
#     prediction = svm_model.predict(input_data_reshaped)
#     return prediction[0]

# # Collect user input data
# gender = st.selectbox("Gender", ("Male", "Female"))
# age = st.slider("Age", 21, 87, 30)
# smoking = st.selectbox("Smoking", ("Yes", "No"))
# yellow_fingers = st.selectbox("Yellow Fingers", ("Yes", "No"))
# anxiety = st.selectbox("Anxiety", ("Yes", "No"))
# peer_pressure = st.selectbox("Peer Pressure", ("Yes", "No"))
# chronic_disease = st.selectbox("Chronic Disease", ("Yes", "No"))
# fatigue = st.selectbox("Fatigue", ("Yes", "No"))
# allergy = st.selectbox("Allergy", ("Yes", "No"))
# wheezing = st.selectbox("Wheezing", ("Yes", "No"))
# alcohol_consuming = st.selectbox("Alcohol Consuming", ("Yes", "No"))
# coughing = st.selectbox("Coughing", ("Yes", "No"))
# shortness_of_breath = st.selectbox("Shortness of Breath", ("Yes", "No"))
# swallowing_difficulty = st.selectbox("Swallowing Difficulty", ("Yes", "No"))
# chest_pain = st.selectbox("Chest Pain", ("Yes", "No"))

# # Convert categorical inputs to numerical format
# gender = 1 if gender == "Male" else 0
# smoking = 2 if smoking == "Yes" else 1
# yellow_fingers = 2 if yellow_fingers == "Yes" else 1
# anxiety = 2 if anxiety == "Yes" else 1
# peer_pressure = 2 if peer_pressure == "Yes" else 1
# chronic_disease = 2 if chronic_disease == "Yes" else 1
# fatigue = 2 if fatigue == "Yes" else 1
# allergy = 2 if allergy == "Yes" else 1
# wheezing = 2 if wheezing == "Yes" else 1
# alcohol_consuming = 2 if alcohol_consuming == "Yes" else 1
# coughing = 2 if coughing == "Yes" else 1
# shortness_of_breath = 2 if shortness_of_breath == "Yes" else 1
# swallowing_difficulty = 2 if swallowing_difficulty == "Yes" else 1
# chest_pain = 2 if chest_pain == "Yes" else 1

# # Combine all inputs into a single array
# user_input = [gender, age, smoking, yellow_fingers, anxiety, peer_pressure,
#               chronic_disease, fatigue, allergy, wheezing, alcohol_consuming,
#               coughing, shortness_of_breath, swallowing_difficulty, chest_pain]

# # When the user clicks the "Predict" button
# if st.button("Predict"):
#     # Make the prediction
#     result = predict_lung_cancer(user_input)
    
#     # Display the result
#     if result == 1:
#         st.error("Prediction: Positive for Lung Cancer")
#     else:
#         st.success("Prediction: Negative for Lung Cancer")


















































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
