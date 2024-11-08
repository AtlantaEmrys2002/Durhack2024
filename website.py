# -*- coding: utf-8 -*-
"""Website.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jq0b7wV57jWDFQIPksIa_PS_UEH7EDhW
"""

# !pip install streamlit

import streamlit as st
import cv2
import keras
import tensorflow as tf
import numpy as np
import tempfile

# Page Layout (Using Streamlit)

st.title("Do I need an umbrella? ☂️")

st.header('''☁️ Welcome to the future of cloud computing...''')

st.subheader('''How does it work?''')

'''Behind this app is a custom-built deep learning classifier which takes any picture of clouds and tells you 
whether it is going to rain or not.'''

'''Never again will you be caught out in a rain storm and return home resembling a piece of seaweed!'''

'''All you have to is upload a cloudy image and make sure you have an umbrella handy!'''

# To build this model, run the Cloud Classifier jupyter notebook

model_loaded = keras.models.load_model('./trained_model.keras')

# Upload file using streamlit

uploaded_image = st.file_uploader("Choose a cloudy image: ", type=['jpg'])

if uploaded_image is not None:

    # Load image from image uploader to temporary file that can be handled by opencv-python
    bytes_data = uploaded_image.getvalue()
    fp = tempfile.NamedTemporaryFile()
    fp.write(bytes_data)
    st.image(bytes_data)
    img = cv2.imread(fp.name)
    fp.close()

    # Process image to be accepted by deep learning model
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resize = tf.image.resize(img, (256, 256))
    np.expand_dims(resize, 0)

    # Input image to deep learning model
    test_prediction = model_loaded.predict(np.expand_dims(resize / 255, 0))

    # Process result
    if test_prediction > 0.5:
        st.subheader("It's going to rain!")
    else:
        st.subheader("It's not going to rain!")
