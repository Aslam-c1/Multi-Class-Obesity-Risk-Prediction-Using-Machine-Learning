import streamlit as st         #type:ignore
import pandas as pd
from pycaret.classification import load_model

# Load the trained model
model_path = r'C:\Users\ACER\OneDrive\Desktop\Infosys Internship\Local system\new_final_model_week3_Infosys_Springboard (1)'
model = load_model(model_path)

# Define the function to predict obesity level
def predict_obesity(input_data):
    input_df = pd.DataFrame([input_data])
    prediction_proba = model.predict_proba(input_df)
    probabilities = {obesity_labels[i]: prediction_proba[0][i] for i in range(len(prediction_proba[0]))}
    predicted_category = max(probabilities, key=probabilities.get)
    return predicted_category, probabilities

# Mapping for obesity level labels
obesity_labels = {
    0: "Insufficient_Weight",
    1: "Normal_Weight",
    2: "Overweight_Level_I",
    3: "Overweight_Level_II",
    4: "Obesity_Type_I",
    5: "Obesity_Type_II",
    6: "Obesity_Type_III"
}

# Streamlit app
st.title("Obesity Level Prediction")

# Function to assign color based on probability value
def get_probability_color(probability):
    if probability >= 0.9:
        return 'darkgreen', 'white'  # dark green with white text
    elif probability >= 0.7:
        return 'green', 'white'  # green with white text
    elif probability >= 0.5:
        return 'lightgreen', 'black'  # light green with black text
    elif probability >= 0.3:
        return 'orange', 'black'  # orange with black text
    elif probability >= 0.1:
        return 'darkorange', 'black'  # dark orange with black text
    else:
        return 'red', 'white'  # red with white text

# Create form for user input
with st.form(key='obesity_form'):
    # Layout the form inputs in two columns
    col1, col2 = st.columns(2)

    # First column inputs
    with col1:
        age = st.number_input('Age (in years)', min_value=0.0, max_value=100.0, step=0.1)
        height = st.number_input('Height (in m)', min_value=0.0, max_value=3.0, step=0.01)
        weight = st.number_input('Weight (in kg)', min_value=0.0, max_value=300.0, step=0.1)
        fcvc = st.number_input('Frequency of Consumption of Vegetables', min_value=0.0, max_value=10.0, step=0.1)
        ch20 = st.number_input('Consumption of Water Daily (in litres)', min_value=0.0, max_value=10.0, step=0.1)
        ncp = st.number_input('Number of Main Meals', min_value=0.0, max_value=10.0, step=0.1)
        faf = st.number_input('Physical Activity Frequency', min_value=0.0, max_value=10.0, step=0.1)
        tue = st.number_input('Time Using Technology Devices (in hrs)', min_value=0.0, max_value=10.0, step=0.1)
    
    st.write()
    st.write()

    # Second column inputs
    with col2:
        gender = st.selectbox('Gender', ['Male', 'Female'])
        family_history_with_overweight = st.selectbox('Family History with Overweight', ['Yes', 'No'])
        frequent_consumption_of_high_caloric_food = st.selectbox('Frequent Consumption of High Caloric Food', ['Yes', 'No'])
        smoke = st.selectbox('Smoke', ['Yes', 'No'])
        consumption_of_alcohol = st.selectbox('Consumption of Alcohol', ['Sometimes', 'Frequently', 'No'])
        consumption_of_food_between_meals = st.selectbox('Consumption of Food Between Meals', ['Sometimes', 'Frequently', 'Always', 'No'])
        calories_consumption_monitoring = st.selectbox('Calories Consumption Monitoring', ['Yes', 'No'])
        transportation_used = st.selectbox('Transportation Used', ['Public_Transportation', 'Automobile', 'Walking', 'Motorbike', 'Bike'])

    submit_button = st.form_submit_button(label='Predict', help='Click to predict obesity level')

# Predict and display the result
if submit_button:
    # Collect input data into a dictionary
    input_data = {
        # Columns expected by the model
        'Gender': gender,
        'family_history_with_overweight': family_history_with_overweight,
        'frequent_consumption_of_high_caloric_food': frequent_consumption_of_high_caloric_food,
        'consumption_of_food_between_meals': consumption_of_food_between_meals,
        'SMOKE': smoke,
        'calories_consumption_monitoring': calories_consumption_monitoring,
        'consumption_of_alcohol': consumption_of_alcohol,
        'transportation_used': transportation_used,
        
        # Additional columns expected by the model
        'Age': age,
        'Height': height,
        'Weight': weight,
        'frequency_of_consumption_of_vegetables': fcvc,
        'number_of_main_meals': ncp,
        'consumption_of_water_daily': ch20,
        'physical_activity_frequency': faf,
        'time_using_technology_devices': tue
    }

    # Perform prediction
    predicted_category, probabilities = predict_obesity(input_data)
    
    # Display prediction result
    st.subheader(f'Predicted Obesity Level: {predicted_category}')
    st.subheader('Prediction Probabilities:')
    prob_df = pd.DataFrame(probabilities.items(), columns=['Obesity Level', 'Probability'])
    
    # Apply color highlighting based on probability value
    prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f'{x:.4f}')

    styled_prob_df = prob_df.style.applymap(lambda x: f'background-color: {get_probability_color(float(x))[0]}; color: {get_probability_color(float(x))[1]}', subset=['Probability'])
    st.dataframe(styled_prob_df)
