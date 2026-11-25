
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="Indian Currency Recognition",
    page_icon="💵"
)

st.title("💵 Indian Currency Recognition")
st.write("Upload an Indian currency note to identify its denomination.")

MODEL_PATH = Path("currency_note_cnn.keras")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

class_names = ["₹10", "₹20", "₹50", "₹100",
               "₹200", "₹500", "₹2000"]

uploaded_file = st.file_uploader(
    "Upload Currency Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Currency",
             use_container_width=True)

    if st.button("Recognize Currency"):

        resized_image = image.resize((128, 128))

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        image_array = image_array / 255.0

        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array, verbose=0)

        predicted_index = np.argmax(predictions[0])

        predicted_currency = class_names[predicted_index]

        confidence = predictions[0][predicted_index] * 100

        st.success("Recognition Completed!")

        st.metric(
            "Predicted Currency",
            predicted_currency
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Class Probabilities")

        for i, currency in enumerate(class_names):
            st.write(currency)
            st.progress(float(predictions[0][i]))
