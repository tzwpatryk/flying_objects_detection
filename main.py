from numpy import ndarray
import numpy as np
import streamlit as st
import numpy as np
from ultralytics import YOLO
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer
import av
import tempfile

# function that converts model prediction on an image to PIL Image
def show_results(results, revert=True):
    im_array = results[0].plot()
    if revert:
        im = Image.fromarray(im_array[..., ::-1])
    else:
        im = Image.fromarray(im_array)
    return im

# callback for a webcam that makes a prediction for each frame and returns image with boxes
def video_frame_callback(frame):
    frame = frame.to_ndarray(format="bgr24")

    results = model(frame, conf=conf)
    im_array = results[0].plot()

    img = av.VideoFrame.from_ndarray(im_array, format="bgr24")
    return img

# loading model with the latest weights
model = YOLO("runs/detect/train/weights/last.pt")

# adding streamlit components
conf = st.slider("Confidence value", min_value=0.1, max_value=0.9, value=0.5)
st.header("Prediction on image")
image = st.file_uploader("Upload image")

# predictions on images
if image is not None:
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(img_bytes, 1)

    results = model(opencv_image, conf=conf)
    img = show_results(results)

    st.image(img)

st.header("Prediction on video")
uploaded_video = st.file_uploader("Upload video", type="mp4")

stframe = st.empty()

# predictions on video
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

# predictions on webcam
st.header("Prediction on your webcam")
webrtc_ctx = webrtc_streamer(key="object-detection-cam",
                            media_stream_constraints={"video": True, "audio": False},
                            video_frame_callback=video_frame_callback)