import streamlit as st
from PIL import Image
import cv2
import numpy as np
import io

def mask_lesion(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(gray) 
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
    else:
        masked_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return masked_image

st.title('Skin Lesion Detector')
uploaded_file = st.file_uploader("Drop an image file here or click to upload", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Mask Lesion'):
        masked_image = mask_lesion(image)
        st.image(masked_image, caption='Masked Lesion Image', use_column_width=True)
