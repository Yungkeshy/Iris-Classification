import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

st.title("Iris Flower Classification")

# Load the trained model and scaler
model = tf.keras.models.load_model('Iris-Classification/iris_model.h5')
scaler = StandardScaler()
scaler.mean_ = np.array([4.8, 3.0, 1.4, 0.1])  # Mean values from the training set
scaler.scale_ = np.array([3.4, 2.4, 3.4, 1.8])  # Standard deviations from the training set

# User input form
st.sidebar.header("Input Features:")
sepal_length = st.sidebar.slider("Sepal Length", 4.0, 8.0, 5.0)
sepal_width = st.sidebar.slider("Sepal Width", 2.0, 4.5, 3.0)
petal_length = st.sidebar.slider("Petal Length", 1.0, 7.0, 4.0)
petal_width = st.sidebar.slider("Petal Width", 0.1, 2.5, 1.0)

# Make a prediction
input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
predicted_prob = model.predict(input_data)
predicted_class = np.argmax(predicted_prob)

# Map the predicted class back to the original label
class_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
predicted_label = class_mapping[predicted_class]

st.write(f"Predicted Iris species: **{predicted_label}**")

# Display the input features
st.subheader("Input Features:")
input_features = pd.DataFrame({
    "Sepal Length": [sepal_length],
    "Sepal Width": [sepal_width],
    "Petal Length": [petal_length],
    "Petal Width": [petal_width]
})
st.table(input_features)
