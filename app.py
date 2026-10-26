
import streamlit as st
import joblib

# Load the trained model and flower names
model = joblib.load("iris_model.pkl")
flower_names = joblib.load("flower_names.pkl")

st.title("Iris Flower Prediction")
st.write("Enter the flower measurements to predict its type.")

# Get input measurements
sepal_length = st.number_input(
    "Sepal Length (cm)", min_value=0.0, value=5.1
)
sepal_width = st.number_input(
    "Sepal Width (cm)", min_value=0.0, value=3.5
)
petal_length = st.number_input(
    "Petal Length (cm)", min_value=0.0, value=1.4
)
petal_width = st.number_input(
    "Petal Width (cm)", min_value=0.0, value=0.2
)

if st.button("Predict"):
    measurements = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    prediction = model.predict(measurements)[0]
    flower = flower_names[prediction]

    st.success(f"Predicted Flower: {flower.title()}")
