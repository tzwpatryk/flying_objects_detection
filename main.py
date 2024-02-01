from numpy import ndarray
import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import av
from turn import get_ice_servers
import tempfile

def show_results(results, revert=True):
    im_array = results[0].plot()
    if revert:
        im = Image.fromarray(im_array[..., ::-1])
    else:
        im = Image.fromarray(im_array)
    return im

def video_frame_callback(frame):
    frame = frame.to_ndarray(format="bgr24")
    results = model(frame, conf=conf)
    im_array = results[0].plot()
    img = av.VideoFrame.from_ndarray(im_array, format="bgr24")

    return img

model = YOLO("runs/detect/train/weights/last.pt")

conf = st.slider("Confidence value", min_value=0.1, max_value=0.9, value=0.3)

st.header("Prediction on image")
image = st.file_uploader("Upload image")

if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(img_bytes, 1)

    results = model(opencv_image, conf=conf)
    img = show_results(results)
    st.image(img)

st.header("Prediction on video")
uploaded_video = st.file_uploader("Upload video", type="mp4")

stframe = st.empty()

if uploaded_video is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success and frame_count%10==0:
            results = model(frame, conf=conf)
            annotated_frame = results[0].plot()
            stframe.image(annotated_frame)
        frame_count += 1
    cap.release()

st.header("Prediction on your webcam")
webrtc_ctx = webrtc_streamer(key="object-detection-cam",
                            rtc_configuration={"iceServers": get_ice_servers()},
                            media_stream_constraints={"video": True, "audio": False},
                            video_frame_callback=video_frame_callback)