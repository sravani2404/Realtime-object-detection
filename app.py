"""
Real-Time Object Detection Dashboard (v3)
-------------------------------------------
"""

import threading
import time
from collections import deque

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
from ultralytics import YOLO


# Page setup + styling

st.set_page_config(page_title="Real-Time Object Detection Dashboard", page_icon="🎯", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header {visibility: hidden;}
    .hero {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%);
        padding: 28px 32px; border-radius: 16px; color: white; margin-bottom: 24px;
    }
    .hero h1 { margin: 0; font-size: 28px; font-weight: 700; }
    .hero p { margin: 6px 0 0 0; font-size: 15px; opacity: 0.9; }
    div[data-testid="stMetric"] {
        background: #ffffff08; border: 1px solid #ffffff15; border-radius: 12px; padding: 10px 14px;
    }
    .badge {
        display: inline-block; background: #22c55e33; color: #22c55e; border: 1px solid #22c55e55;
        padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; margin-right: 6px;
    }
    </style>
    <div class="hero">
        <h1>🎯 Real-Time Object Detection Dashboard</h1>
        <p>Live webcam inference powered by YOLOv8 — streamed to your browser, analyzed in real time.</p>
        <span class="badge">● Live</span><span class="badge">YOLOv8n</span><span class="badge">80 COCO classes</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)



# Video processor: this OBJECT is what actually survives between reruns.

class YOLOProcessor(VideoProcessorBase):
    def __init__(self) -> None:
        self.conf_threshold = 0.4
        self.class_filter: list[str] = []
        self.log: deque = deque(maxlen=300)
        self.lock = threading.Lock()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.resize(img, (480, 360))  # smaller frame = faster inference, less lag

        results = model.predict(img, conf=self.conf_threshold, verbose=False)[0]
        annotated = results.plot()

        detected_classes = []
        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            if self.class_filter and cls_name not in self.class_filter:
                continue
            detected_classes.append(cls_name)

        with self.lock:
            self.log.append(
                {"timestamp": time.time(), "objects": detected_classes, "count": len(detected_classes)}
            )

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")



# Sidebar controls

conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)
class_filter = st.sidebar.multiselect("Filter to specific classes (optional)", list(model.names.values()))
st.sidebar.markdown("---")
st.sidebar.caption("Model: YOLOv8n · Classes: 80 (COCO)")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📷 Live Feed")
    ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=YOLOProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Push the latest sidebar settings into the running processor every rerun
if ctx.video_processor:
    ctx.video_processor.conf_threshold = conf_threshold
    ctx.video_processor.class_filter = class_filter

with col2:
    st.markdown("### 📊 Live Analytics")
    st_autorefresh(interval=1000, key="analytics_refresh")

    if ctx.video_processor:
        with ctx.video_processor.lock:
            log_snapshot = list(ctx.video_processor.log)
    else:
        log_snapshot = []

    if log_snapshot:
        df = pd.DataFrame(log_snapshot)
        latest_count = df.iloc[-1]["count"]
        avg_count = round(df["count"].mean(), 1)

        m1, m2 = st.columns(2)
        m1.metric("Objects right now", latest_count)
        m2.metric("Avg per frame", avg_count)

        all_objects = [obj for row in df["objects"] for obj in row]
        if all_objects:
            st.markdown("**Object frequency**")
            st.bar_chart(pd.Series(all_objects).value_counts())

        st.markdown("**Detections per frame (recent history)**")
        st.line_chart(df.set_index("timestamp")["count"])
    else:
        st.info("Start the webcam on the left — analytics will appear here within a second or two.")

st.markdown("---")
st.markdown("### 🖼️ Or test with a single image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    results = model.predict(img, conf=conf_threshold, verbose=False)[0]
    annotated = results.plot()
    st.image(annotated[:, :, ::-1], caption="Detection result", use_container_width=True)
    if len(results.boxes):
        st.write(pd.Series([model.names[int(b.cls[0])] for b in results.boxes]).value_counts())