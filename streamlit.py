## to run the program type streamlit run streamlit.py on terminal


import tensorflow as tf
from tensorflow import keras
import keras
import numpy as np 
import os
import streamlit as st
import PIL
from PIL import Image
import cv2
import time

from keras.models import load_model
import tensorflow.keras.backend as K

st.markdown("<h1 style='text-align:center;'>Image Colorization</h1>",unsafe_allow_html=True)
st.markdown("<h3 style='text-align:center;'> Built with Tensorflow2 & Keras</h1>",unsafe_allow_html=True)

st.text('2. Click the button below to colorize your selected image.')

gray = np.load('/users/benzhao/downloads/gray_scale.npy')

st.sidebar.title('1. Choose from 300 images')

i=st.sidebar.number_input(label='Enter a value:',min_value=1,value=1,step=1)

def batch_prep(gray_img, batch_size=100):
    img=np.zeros((batch_size,224,224,3))
    for i in range(0,3):
        img[:batch_size,:,:,i]=gray_img[:batch_size]
    return img
    
img_in=batch_prep(gray,batch_size=300)

st.sidebar.image(gray[i])

model=tf.keras.models.load_model('modelfinal.h5')
prediction=model.predict(img_in)

start_analyze_file=st.button('Colorize')
if start_analyze_file==True:
    
    with st.spinner(text='colorizing'):
        time.sleep(1)
        
    model=tf.keras.models.load_model('modelfinal.h5')
    prediction=model.predict(img_in)
    st.cache(func=None, persist=False, allow_output_mutation=True, show_spinner=True, suppress_st_warning=False, hash_funcs=None, max_entries=None, ttl=None)   
    st.success('Finished!')
    st.image(prediction[i].astype('uint8'),clamp=True)
    
    
