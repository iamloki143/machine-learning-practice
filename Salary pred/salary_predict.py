
import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("/home/lokesh/Downloads/Salary pred/Model_ml.sav", "rb"))
st.set_page_config()

st.markdown("""
<style>
/* Main background */
.stApp {
    background-color: #0e1117;
}

/* Heading */
h1 {
    color: #ff334f;
    font-size: 42px;
}

/* Buttons */
.stButton > button {
    background-color: #ff334f;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
}

/* Input fields */
.stNumberInput input {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)
st.title("Salary Prediction App")
st.write("Predict an estimated salary using your trained ML model.")

st.divider()

experience = st.number_input(
    "Years of Experience",
    min_value=0.0,
    max_value=50.0,
    value=1.0,
    step=0.5
)

if st.button("Predict Salary"):
    input_data = np.array([[experience]])

    prediction = model.predict(input_data)
    predict_salary= prediction[0][0]

    st.success(
        f"Predicted Salary: ₹{float(predict_salary):,.2f}"
    )


