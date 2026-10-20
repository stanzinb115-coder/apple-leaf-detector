import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

class_names = ["Apple scab", "Black rot", "Cedar apple rust", "Healthy"]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("apple_leaf_model.keras")

model = load_model()

st.title("Apple Leaf Disease Detector")
st.write("Upload a photo of an apple leaf and the model will guess what it is.")
st.caption("Works best on a close-up photo of a single leaf.")

file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

if file is not None:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    arr = np.array(img.resize((224, 224)), dtype="float32")
    arr = np.expand_dims(arr, 0)

    probs = model.predict(arr)[0]
    best = int(np.argmax(probs))

    st.subheader(f"{class_names[best]} ({probs[best] * 100:.0f}%)")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {probs[i] * 100:.1f}%")
