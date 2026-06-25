import json
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path

# Load Model
@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "saved_model.h5"
    return tf.keras.models.load_model(model_path)

# Load Class Names
def load_class_names():
    class_names_path = Path(__file__).resolve().parent / "class_names.json"

    if class_names_path.exists():
        with open(class_names_path, "r", encoding="utf-8") as f:
            return json.load(f)

    return ["FAKE", "REAL"]

# Preprocess Image
def preprocess_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(128, 128))
    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Main App
def main():

    st.set_page_config(
        page_title="AI Image Detection",
        page_icon="🖼️",
        layout="centered"
    )

    st.title("🖼️ AI Generated Image Detection")
    st.write(
        "Upload an image and the model will predict whether it is REAL or FAKE."
    )

    # Load model and classes
    import tensorflow as tf
    model = tf.keras.models.load_model("saved_model.h5", compile=False)

    class_names = load_class_names()

    uploaded_file = st.file_uploader(
        "Choose an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        st.image(
            uploaded_file,
            caption="Uploaded Image",
            use_container_width=True
        )

        with st.spinner("Analyzing Image..."):

            img_array = preprocess_image(uploaded_file)
        
            prediction = model.predict(img_array, verbose=0)[0]

            st.write("Raw Prediction:", prediction)

            predicted_idx = int(np.argmax(prediction))

            predicted_class = class_names[predicted_idx]

            confidence = float(prediction[predicted_idx]) * 100

            st.success("Prediction Completed")

            st.subheader("Prediction Result")

            st.write(f"**Predicted Class:** {predicted_class}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            if len(prediction) >= 2:
                st.write(f"**FAKE Score:** {prediction[0] * 100:.2f}%")
                st.write(f"**REAL Score:** {prediction[1] * 100:.2f}%")

if __name__ == "__main__":
    main()