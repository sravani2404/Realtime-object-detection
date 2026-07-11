# 🎯 Real-Time Object Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.11-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Deployed](https://img.shields.io/badge/Deployed-Streamlit%20Community%20Cloud-ff4b4b)

A computer vision app that detects objects in real time using **YOLOv8**. Take a photo
or upload an image and get instant bounding boxes, class labels, and a running detection
history — deployed for free with no GPU required.

**🔗 Live demo:** _paste your `https://your-app-name.streamlit.app` link here_

---

## Two Versions in This Repo

- **`app.py`** — snapshot-based capture (this is what's deployed live, link above).
  Uses `st.camera_input` to take a single photo over plain HTTPS. Chosen for the public
  demo specifically because continuous WebRTC video isn't reliably supported on free
  hosting without a paid TURN server (see *Architecture Decisions* below).
- **`app_live_local.py`** — the original continuous live-webcam version, using
  `streamlit-webrtc`. Fully functional, but only run locally:
  ```
  streamlit run app_live_local.py
  ```

## Features

- Real-time YOLOv8 inference on a captured photo or uploaded image
- Adjustable confidence threshold and per-class filtering in the sidebar
- Session-long detection history: object frequency chart + detections-per-shot trend
- Two versions in one repo, so both a reliable public demo and a full live-video build exist

## Tech Stack

`Python` · `YOLOv8 (Ultralytics)` · `OpenCV` · `Streamlit` · `streamlit-webrtc` (local version only) · `Pandas`

## Run Locally

```bash
git clone <your-repo-url>
cd realtime-object-detection
python3 -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py                # snapshot version
# or
streamlit run app_live_local.py     # continuous live-video version
```

## Deploy for Free (Streamlit Community Cloud)

1. Push this repo to GitHub (see commands below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Click **"Create app"** → **"Deploy a public app from GitHub"**.
4. Select this repository, branch `main`, and main file path `app.py`.
5. Click **"Advanced settings"** and set the Python version to **3.11** (important —
   newer Python versions can break some of these packages).
6. Click **Deploy**. First build takes a few minutes.
7. Note the live URL (`https://your-app-name.streamlit.app`) and add it to the top of
   this README.

**Note:** the free tier spins down after inactivity — the first visit after idle time
can take 30-60 seconds to wake back up.

## Architecture Decisions

**Why isn't the live version deployed?** `streamlit-webrtc` requires a working WebRTC
connection between the visitor's browser and the server. Streamlit Community Cloud's
network is documented (by the `streamlit-webrtc` library itself) to block these
connections without a paid TURN relay service. Rather than deploy a public demo link
that hangs on "Connecting..." for visitors, the public version uses `st.camera_input`,
which works over standard HTTPS with no special networking required. The continuous
live-video version is kept in the repo for local use, since it's fully functional —
just not reliably deployable on free infrastructure.

## What This Demonstrates (for interviews)

- Real-time computer vision inference pipeline, not just batch/offline prediction
- Understanding of networking constraints (WebRTC/NAT traversal) affecting deployment choices, and making a deliberate reliability trade-off instead of shipping something fragile
- End-to-end ownership: model → app → cloud deployment, not just a notebook
- Basic MLOps: reproducible environment (`requirements.txt`, `packages.txt`), cached model loading

## Possible Extensions

- Swap in a custom-trained YOLO model fine-tuned on a niche class
- Add a proper paid/self-hosted TURN server to make the live version deployable too
- Add object tracking (ByteTrack/DeepSORT) for persistent IDs across frames