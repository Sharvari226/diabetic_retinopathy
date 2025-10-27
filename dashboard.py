import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Lambda

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model("dr_multitask_v2_final", compile=False)
    return model


model = load_trained_model()
st.sidebar.success("✅ Model Loaded Successfully")

