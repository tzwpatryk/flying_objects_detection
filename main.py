import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2

st.set_page_config(page_title='Flying objects detection using YOLOV8', layout='wide')

model = YOLO("runs/detect/train/weights/last.pt")

with st.header("Upload image"):
    image = st.file_uploader("JPEG file:")

if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(img_bytes, 1)

    st.image(opencv_image)

    results = model(opencv_image)
    
