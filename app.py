import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide"
)

# --------------------------------
# LOAD MODEL
# --------------------------------
model = tf.keras.models.load_model("titanic_ann_model.keras")

# --------------------------------
# HEADER SECTION
# --------------------------------
st.markdown("""
# 🚢 Titanic Survival Prediction System
### Deep Learning Based Passenger Survival Prediction
""")

st.image(
    "https://images.unsplash.com/photo-1569253519107-6bfc0d8d935d",
    width=700
)

# --------------------------------
# PROJECT DESCRIPTION
# --------------------------------
st.markdown("""
## 📌 Project Description

This application predicts whether a passenger would survive during the Titanic disaster using an Artificial Neural Network (ANN).

### Technologies Used
- Deep Learning
- TensorFlow/Keras
- Streamlit
- ANN Model Deployment

The system takes passenger details as input and predicts survival probability.
""")

st.divider()

# --------------------------------
# INPUT SECTION
# --------------------------------
st.markdown("## 🎯 Passenger Details")

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
        value=120.0
    )

# --------------------------------
# NORMALIZATION FUNCTION
# --------------------------------
def normalize_inputs(pclass, age, fare):

    pclass_norm = (pclass - 1) / 2
    age_norm = age / 100
    fare_norm = fare / 150

    return np.array([[pclass_norm, age_norm, fare_norm]])

# --------------------------------
# PREDICTION BUTTON
# --------------------------------
if st.button("Predict Survival"):

    # preprocess
    input_data = normalize_inputs(pclass, age, fare)

    # prediction
    prediction = model.predict(input_data)

    survival_prob = float(prediction[0][0])
    nonsurvival_prob = 1 - survival_prob

    # decision
    if survival_prob > 0.5:
        result = "✅ Survived"
    else:
        result = "❌ Not Survived"

    # --------------------------------
    # OUTPUT SECTION
    # --------------------------------
    st.markdown("## 📊 Prediction Output")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Prediction",
            result
        )

    with col2:
        st.metric(
            "Survival Probability",
            f"{survival_prob * 100:.2f}%"
        )

    with col3:
        st.metric(
            "Confidence Score",
            f"{max(survival_prob, nonsurvival_prob) * 100:.2f}%"
        )

    st.divider()

    # --------------------------------
    # BAR CHART
    # --------------------------------
    st.markdown("## 📈 Probability Visualization")

    labels = ["Survived", "Not Survived"]
    values = [survival_prob, nonsurvival_prob]

    fig, ax = plt.subplots()

    ax.bar(labels, values)

    ax.set_ylabel("Probability")
    ax.set_title("Prediction Probability")

    st.pyplot(fig)

    # --------------------------------
    # PIE CHART
    # --------------------------------
    fig2, ax2 = plt.subplots()

    ax2.pie(
        values,
        labels=labels,
        autopct='%1.1f%%'
    )

    ax2.set_title("Survival Probability Distribution")

    st.pyplot(fig2)

# --------------------------------
# FOOTER
# --------------------------------
st.divider()

st.markdown("""
### 🚀 Developed using Streamlit + TensorFlow
""")