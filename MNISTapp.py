import os 
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

model = load_model('MNISTmodel.keras')

st.title('Hand Written Digit Classification')
st.header('Liam Frank')
st.write('Shallow Convolutional Neural Network trained on MNIST dataset')
st.write("\n" * 4)

col1, col2 = st.columns(2)
SIZE=200

with col1:
    st.subheader('Draw a Digit 0-9:')
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        drawing_mode="freedraw",
        key='canvas'
    )

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    
    with col2:
        st.subheader('Pixelated Model Input:')
        st.image(rescaled)
        
st.markdown("""
    <style>
    div.stButton > button {
        width: 200px;
        height: 50px;
        font-size: 20px;
        font-weight: bold;
        color: white;
        border: none;
        border-radius: 8px;
        cursor: pointer;
        transition: color 0.3s ease; background-color 0.3s ease;
    }
    
    div.stButton > button:hover {
        background-color: white;
        color: black;
    }
    
    </style>
    """, unsafe_allow_html=True)

if st.button('Predict'):
    with st.spinner("Making prediction..."):
        test_x = img[:, :, 0]
        test_x = test_x / 255.0
        val = model.predict(test_x.reshape(1, 28, 28))
        st.metric("Result:", np.argmax(val[0]))

