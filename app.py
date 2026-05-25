# ---------------------------------------------------
# Titanic Survival Prediction System
# Streamlit + TensorFlow Deployment
# ---------------------------------------------------

import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------

st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# ---------------------------------------------------
# Load Trained Model
# ---------------------------------------------------
import keras

model = keras.models.load_model("titanic_ann_model.keras")

# ---------------------------------------------------
# Header Section
# ---------------------------------------------------

st.title("🚢 Titanic Survival Prediction System")

st.subheader(
    "Deep Learning Based Passenger Survival Prediction"
)

# ---------------------------------------------------
# Project Description
# ---------------------------------------------------

st.markdown("""
## 📌 Project Description

This application predicts whether a passenger would survive
during the Titanic disaster using an Artificial Neural Network (ANN).

### Technologies Used
- Deep Learning
- TensorFlow/Keras
- Streamlit Deployment

The model analyzes passenger information such as:
- Passenger Class
- Age
- Fare

and predicts survival probability.
""")

# ---------------------------------------------------
# Input Section
# ---------------------------------------------------

st.markdown("## 🎯 Passenger Input Form")

col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox(
        "Passenger Class",
        [1, 2, 3]
    )

with col2:
    age = st.slider(
        "Age",
        1,
        80,
        24
    )

with col3:
    fare = st.number_input(
        "Fare",
        min_value=0.0,
        max_value=600.0,
        value=120.0
    )

# ---------------------------------------------------
# Data Preprocessing
# ---------------------------------------------------

# Training Data Ranges
# Replace with actual dataset min/max if needed

pclass_min, pclass_max = 1, 3
age_min, age_max = 1, 80
fare_min, fare_max = 0, 600

# Min-Max Normalization

pclass_norm = (pclass - pclass_min) / (pclass_max - pclass_min)

age_norm = (age - age_min) / (age_max - age_min)

fare_norm = (fare - fare_min) / (fare_max - fare_min)

# Prepare Input

input_data = np.array([
    [pclass_norm, age_norm, fare_norm]
])

# ---------------------------------------------------
# Prediction Button
# ---------------------------------------------------

if st.button("Predict Survival"):

    # Prediction
    prediction = model.predict(input_data)

    probability = prediction[0][0]

    # Prediction Logic
    if probability > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    # ---------------------------------------------------
    # Output Section
    # ---------------------------------------------------

    st.markdown("## 📊 Prediction Result")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            label="Prediction",
            value=result
        )

    with c2:
        st.metric(
            label="Survival Probability",
            value=f"{probability*100:.2f}%"
        )

    with c3:
        confidence = max(probability, 1 - probability)

        st.metric(
            label="Confidence Score",
            value=f"{confidence*100:.2f}%"
        )

    # ---------------------------------------------------
    # Visualization
    # ---------------------------------------------------

    st.markdown("## 📈 Survival Probability Visualization")

    survive_prob = probability
    nonsurvive_prob = 1 - probability

    chart_data = pd.DataFrame({
        "Category": ["Survived", "Not Survived"],
        "Probability": [survive_prob, nonsurvive_prob]
    })

    st.bar_chart(
        chart_data.set_index("Category")
    )

    # Pie Chart

fig, ax = plt.subplots(figsize=(4, 4))

ax.pie(
    [survive_prob, nonsurvive_prob],
    labels=["Survived", "Not Survived"],
    autopct="%1.1f%%"
)

ax.axis("equal")

st.pyplot(fig, use_container_width=False)