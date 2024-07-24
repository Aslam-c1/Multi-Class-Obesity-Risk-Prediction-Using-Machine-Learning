import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

# Function to load the model
def load_trained_model(model_path):
    return load_model(model_path)

# Load the model
model = load_trained_model('/home/samsapi0l/Documents/infosys-internship-project/SET2/obesity/obesity_model')

# Define the dropdown options
dropdown_options = {
    'Gender': ['Male', 'Female'],
    'family_history_with_overweight': ['yes', 'no'],
    'frequent_consumption_of_high_caloric_food': ['yes', 'no'],
    'consumption_of_food_between_meals': ['Sometimes', 'Frequently', 'no', 'Always'],
    'SMOKE': ['no', 'yes'],
    'calories_consumption_monitoring': ['no', 'yes'],
    'consumption_of_alcohol': ['Sometimes', 'no', 'Frequently'],
    'transportation_used': ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'],
    'obesity_level': ['Overweight_Level_II', 'Normal_Weight', 'Insufficient_Weight', 'Obesity_Type_III', 'Obesity_Type_II', 'Overweight_Level_I', 'Obesity_Type_I']
}

# Create the Streamlit form
with st.form("prediction_form"):
    st.header("Obesity Risk Prediction Form")

    # Create fields in the specified order
    gender = st.selectbox('Gender', dropdown_options['Gender'])
    age = st.number_input('Age', value=24.443011)
    height = st.number_input('Height (in meters)', value=1.699998)
    weight = st.number_input('Weight (in kg)', value=81.66995)
    family_history = st.selectbox('Family History with Overweight', dropdown_options['family_history_with_overweight'])
    high_caloric_food = st.selectbox('Frequent Consumption of High Caloric Food', dropdown_options['frequent_consumption_of_high_caloric_food'])
    veg_consumption = st.number_input('Frequency of Consumption of Vegetables', value=2.0)
    main_meals = st.number_input('Number of Main Meals', value=2.983297)
    food_between_meals = st.selectbox('Consumption of Food Between Meals', dropdown_options['consumption_of_food_between_meals'])
    smoke = st.selectbox('Do you smoke?', dropdown_options['SMOKE'])
    water_consumption = st.number_input('Daily Consumption of Water (in liters)', value=2.763573)
    calorie_monitoring = st.selectbox('Calories Consumption Monitoring', dropdown_options['calories_consumption_monitoring'])
    physical_activity = st.number_input('Physical Activity Frequency', value=0.0)
    tech_time = st.number_input('Time Using Technology Devices (in hours)', value=0.976473)
    alcohol_consumption = st.selectbox('Consumption of Alcohol', dropdown_options['consumption_of_alcohol'])
    transportation = st.selectbox('Transportation Used', dropdown_options['transportation_used'])

    # Form submission button
    submit = st.form_submit_button("Predict")

# Perform prediction on form submission
if submit:
    input_data = {
        'Gender': gender,
        'Age': age,
        'Height': height,
        'Weight': weight,
        'family_history_with_overweight': family_history,
        'frequent_consumption_of_high_caloric_food': high_caloric_food,
        'frequency_of_consumption_of_vegetables': veg_consumption,
        'number_of_main_meals': main_meals,
        'consumption_of_food_between_meals': food_between_meals,
        'SMOKE': smoke,
        'consumption_of_water_daily': water_consumption,
        'calories_consumption_monitoring': calorie_monitoring,
        'physical_activity_frequency': physical_activity,
        'time_using_technology_devices': tech_time,
        'consumption_of_alcohol': alcohol_consumption,
        'transportation_used': transportation
    }

    input_df = pd.DataFrame([input_data])
    predictions = predict_model(model, data=input_df)
    # st.write(predictions)
    prediction_result = predictions[['prediction_label', 'prediction_score']].iloc[0]
    st.write('Prediction Label:', prediction_result['prediction_label'])
    st.write('Prediction Score:', prediction_result['prediction_score'])