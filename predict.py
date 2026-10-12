import pandas as pd
import joblib

def predict_calories(inputs):
    # Load the saved model and label encoders
    model = joblib.load("calories_model.joblib")
    label_encoders = joblib.load("label_encoders.joblib")

    # Convert inputs to a DataFrame
    input_df = pd.DataFrame([inputs], columns=[
        'Age', 'Gender', 'Weight (kg)', 'Height (m)', 'Max_BPM', 'Avg_BPM',
        'Resting_BPM', 'Workout_Type', 'Fat_Percentage', 'Water_Intake (liters)',
        'Workout_Frequency (days/week)', 'Experience_Level', 'BMI', 'Diet_Type',
        'Sleep_Duration (hours)', 'Session_Duration (minutes)'
    ])

    # Encode categorical variables using the saved label encoders
    for col in ['Gender', 'Workout_Type', 'Diet_Type']:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Make prediction
    prediction = model.predict(input_df)
    return prediction[0]