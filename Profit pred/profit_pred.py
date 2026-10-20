
import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("/home/lokesh/Downloads/profit_ml.sav", "rb"))
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
st.title("Startup Profit App")
st.write("Predict an estimated profit using your trained ML model.")

st.divider()

RD_spend = st.number_input(
    "Enter R&D Spend Amount:",
    min_value=0.0,
    max_value=99999999.0,
    value=1000.0,
    step=500.0
)
Adm_spend = st.number_input(
    "Enter Administration Amount:",
    min_value=0.0,
    max_value=99999999.0,
    value=1000.0,
    step=500.0
)
Mark_spend = st.number_input(
    "Enter Marketing Spend Amount:",
    min_value=0.0,
    max_value=99999999.0,
    value=1000.0,
    step=500.0
)

if st.button("Predict Profit"):
    input_data = pd.DataFrame([{
        'R&D Spend': RD_spend,
        'Administration': Adm_spend,
        'Marketing Spend': Mark_spend
    }])

    prediction = model.predict(input_data)

    predict_profit= prediction[0]

    st.success(
        f"Predicted Salary: ₹{float(predict_profit):,.2f}"
    )


