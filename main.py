import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, ClientSettings
import av

def show_results(results, revert=True):
    im_array = results[0].plot()
    if revert:
        im = Image.fromarray(im_array[..., ::-1])
    else:
        im = Image.fromarray(im_array)
    return im

def video_frame_callback(frame):
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.to_ndarray(format="bgr24")
    results = model(frame, conf=0.3)
    # img = show_results(results, revert=False)
    im_array = results[0].plot()
    img = av.VideoFrame.from_ndarray(im_array, format="bgr24")

    return img

WEBRTC_CLIENT_SETTINGS = ClientSettings(
    # rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    )


model = YOLO("runs/detect/train/weights/last.pt")

with st.header("Upload image"):
    image = st.file_uploader("JPEG file:")

if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(img_bytes, 1)

    results = model(opencv_image)
    img = show_results(results)
    st.image(img)

webrtc_ctx = webrtc_streamer(key="snapshot", 
                             client_settings=WEBRTC_CLIENT_SETTINGS, 
                             video_frame_callback=video_frame_callback)