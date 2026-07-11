"""
Real-Time Object Detection Dashboard (v4 — snapshot edition)
---------------------------------------------------------------
Why this version exists: continuous live video (via streamlit-webrtc) requires
a WebRTC connection, which Streamlit Community Cloud's network is known to
block without a paid TURN server. Rather than ship a demo link that hangs on
"Connecting..." for visitors, this version uses st.camera_input — a simple
"take a photo" button that works over plain HTTPS, no WebRTC required. It
runs the exact same YOLOv8 model on each captured photo.

The true continuous-video version still works great locally (see the README)
and that's what the demo GIF shows — this is just what's deployed publicly,
chosen specifically for reliability.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import hashlib
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Page setup + styling
# ---------------------------------------------------------------------------
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
        <p>Snapshot-based inference powered by YOLOv8 — take a photo, get instant detections.</p>
        <span class="badge">YOLOv8n</span><span class="badge">80 COCO classes</span>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")


model = load_model()

# ---------------------------------------------------------------------------
# Session state: safe to use here since everything runs on the main thread
# (no background WebRTC thread involved in this version)
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "last_hash" not in st.session_state:
    st.session_state.last_hash = None

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Controls")
conf_threshold = st.sidebar.slider("Confidence threshold", 0.1, 1.0, 0.4, 0.05)
class_filter = st.sidebar.multiselect("Filter to specific classes (optional)", list(model.names.values()))
st.sidebar.markdown("---")
st.sidebar.caption("Model: YOLOv8n · Classes: 80 (COCO)")
if st.sidebar.button("🗑️ Clear history"):
    st.session_state.history = []
    st.session_state.last_hash = None


def run_detection(image_bgr: np.ndarray):
    results = model.predict(image_bgr, conf=conf_threshold, verbose=False)[0]
    annotated = results.plot()
    detected_classes = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        if class_filter and cls_name not in class_filter:
            continue
        detected_classes.append(cls_name)
    return annotated, detected_classes


col1, col2 = st.columns([2, 1])

with col1:
    tab1, tab2 = st.tabs(["📷 Take a Photo", "🖼️ Upload an Image"])

    with tab1:
        camera_file = st.camera_input("Click below to capture a photo")
        if camera_file is not None:
            file_bytes = np.asarray(bytearray(camera_file.getvalue()), dtype=np.uint8)
            img_hash = hashlib.md5(file_bytes).hexdigest()
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, detected_classes = run_detection(img)
            st.image(annotated[:, :, ::-1], caption="Detection result", use_container_width=True)

            # Only log a new history entry if this is an actually new capture
            if img_hash != st.session_state.last_hash:
                st.session_state.history.append(
                    {"timestamp": time.time(), "objects": detected_classes, "count": len(detected_classes)}
                )
                st.session_state.last_hash = img_hash

    with tab2:
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            annotated, detected_classes = run_detection(img)
            st.image(annotated[:, :, ::-1], caption="Detection result", use_container_width=True)

with col2:
    st.markdown("### 📊 Detection History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        latest_count = df.iloc[-1]["count"]
        avg_count = round(df["count"].mean(), 1)

        m1, m2 = st.columns(2)
        m1.metric("Objects in last shot", latest_count)
        m2.metric("Avg per shot", avg_count)

        all_objects = [obj for row in df["objects"] for obj in row]
        if all_objects:
            st.markdown("**Object frequency (this session)**")
            st.bar_chart(pd.Series(all_objects).value_counts())

        st.markdown("**Objects per snapshot over time**")
        st.line_chart(df.reset_index()["count"])
    else:
        st.info("Take a photo or upload an image to see detection history here.")