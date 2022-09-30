import streamlit as st
import pandas as pd
from PIL import Image
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
"""
# Deep Learning Classifier
"""
print(os.getcwd())
model = tf.keras.models.load_model(Path("model.h5"))
uploaded_file = st.file_uploader("Choose a image to be upload")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image.resize((224,224)))
    img_array = np.expand_dims(img_array, axis=0)
    result = model.predict(img_array)
    argmax_index = np.argmax(result, axis=1)
    if argmax_index[0] == 0:
        st.image(image, caption="Predicted as : Cat")
        print("Predicted as : Cat")
    else:
        st.image(image, caption="Predicted as : Dog")
        print("Predicted as : Dog")
    