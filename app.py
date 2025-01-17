import streamlit as st
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# Load the trained model and necessary preprocessing
@st.cache_data
def load_model():
    
     data = pd.read_csv('Food_Delivery_Times.csv')
    
    # Preprocess categorical columns
     label_encoder = LabelEncoder()
     data['Traffic_Level'] = label_encoder.fit_transform(data['Traffic_Level'])
     data['Time_of_Day'] = label_encoder.fit_transform(data['Time_of_Day'])
     data['Vehicle_Type'] = label_encoder.fit_transform(data['Vehicle_Type'])
     data['Weather']=label_encoder.fit_transform(data['Weather'])

    # Select features and target variable
     X = data[['Distance_km', 'Traffic_Level','Weather','Preparation_Time_min','Vehicle_Type','Time_of_Day']]
     y = data['Delivery_Time_min']

     # Split the data into training and testing sets
     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

     # Train the Linear Regression model
     model = LinearRegression()
     model.fit(X_train, y_train)

     return model

def predict_delivery_time(Distance_km, Traffic_Level,Weather,Preparation_Time_min,Vehicle_Type,Time_of_Day):
    # Preprocess the input features (label encoding for categorical variables)
    label_encoder = LabelEncoder()
    Traffic_Level_encoded = label_encoder.fit_transform([Traffic_Level])[0]
    Time_of_Day_encoded= label_encoder.fit_transform([Time_of_Day])[0]
    Vehicle_Type_encoded= label_encoder.fit_transform([Vehicle_Type])[0]
    Weather_encoded=label_encoder.fit_transform([Weather])[0]

    # Prepare the input for prediction
    input_data = pd.DataFrame([[Distance_km,Traffic_Level_encoded,Time_of_Day_encoded,Vehicle_Type_encoded,Weather_encoded,Preparation_Time_min]], columns=['Distance_km', 'Traffic_Level','Weather','Preparation_Time_min','Vehicle_Type','Time_of_Day'])

    # Predict delivery time using the model
    prediction = model.predict(input_data)
    return prediction[0]

# Set up Streamlit layout
st.title("Food Delivery Time Prediction")

# Input fields for the user
Distance_km= st.number_input('Distance_km', min_value=5.0, max_value=50.0, value=5.0)
Traffic_Level = st.selectbox('Traffic_Level', ['Low', 'Medium', 'High'])
Time_of_Day=st.selectbox('Time_of_Day',['Morning','Afternoon','Evening','Night'])
Vehicle_Type=st.selectbox('Vehicle_Type',['Bike','Scooter','Car'])
Weather=st.selectbox('Weather',['Windy','Clear','Foggy','Rainy','Snowy'])
Preparation_Time_min=st.number_input('Preparation_Time_min', min_value=5.0, max_value=30.0, value=5.0)

# Load model
model = load_model()

# When the user clicks the button, show the prediction
if st.button('Predict Delivery Time'):
    prediction = predict_delivery_time(Distance_km, Traffic_Level,Weather,Preparation_Time_min,Vehicle_Type,Time_of_Day)
    st.write(f"Predicted Delivery Time: {prediction:.2f} minutes")

