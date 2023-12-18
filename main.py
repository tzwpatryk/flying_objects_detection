import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image

def show_results(results, revert=True):
    im_array = results[0].plot()
    if revert:
        im = Image.fromarray(im_array[..., ::-1])
    else:
        im = Image.fromarray(im_array)
    return im


st.set_page_config(page_title='Flying objects detection using YOLOV8', layout='wide')

model = YOLO("runs/detect/train/weights/last.pt")

with st.header("Upload image"):
    image = st.file_uploader("JPEG file:")

if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(img_bytes, 1)

    results = model(opencv_image)
    img = show_results(results)
    st.image(img)

with st.header("Enable webcam"):
    webcam = st.checkbox("Run")

FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while webcam:
    _, frame = camera.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model(frame)
    img = show_results(results, revert=False)

    FRAME_WINDOW.image(img)
else:
    st.write('Stopped')