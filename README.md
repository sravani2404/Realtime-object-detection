# 🎯 Real-Time Object Detection Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![Deployed](https://img.shields.io/badge/Deployed-HuggingFace%20Spaces-yellow)

A browser-based real-time object detection app. Point your webcam at anything and see
live bounding boxes, class labels, and a running analytics dashboard (object frequency,
detections-per-frame trend) — all running on free infrastructure with no GPU required.

**🔗 Live demo:** _add your Hugging Face Spaces link here after deployment_

![demo](docs/demo.gif)
<!-- Record a 5-10s screen capture of the live app and drop it in a /docs folder as demo.gif -->

---

## Features

- Real-time video inference in-browser via WebRTC (`streamlit-webrtc`)
- YOLOv8n (nano) model — fast enough to run live on a free CPU instance
- Adjustable confidence threshold and per-class filtering, live in the sidebar
- Live analytics panel: object frequency bar chart + detections-per-frame trend line
- Fallback single-image upload mode (useful if a reviewer's browser blocks webcam access)

## Tech Stack

`Python` · `YOLOv8 (Ultralytics)` · `OpenCV` · `Streamlit` · `streamlit-webrtc` · `Pandas`

## Architecture

```
Browser webcam ──(WebRTC)──> streamlit-webrtc ──> YOLOv8 inference ──> annotated frame ──> back to browser
                                                        │
                                                        └──> detection log ──> live charts (Streamlit)
```

## Run Locally

```bash
git clone <your-repo-url>
cd realtime-object-detection
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`), click "Start" on the
video widget, and allow camera access.

## Deploy for Free (Hugging Face Spaces)

1. Create a free account at [huggingface.co](https://huggingface.co).
2. Click **New Space** → SDK: **Streamlit** → Hardware: **CPU basic (free)**.
3. Push this folder's contents to the Space's git repo:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```
4. The Space auto-builds using `requirements.txt` and `packages.txt`. First build takes
   a few minutes (downloading the YOLOv8n weights). After that, it's live at
   `https://huggingface.co/spaces/<your-username>/<space-name>`.
5. Put that link at the top of this README and on your resume/LinkedIn.

## What This Demonstrates (for interviews)

- Real-time computer vision inference pipeline, not just batch/offline prediction
- Trade-off reasoning: chose YOLOv8n over larger variants specifically for free-tier CPU latency
- End-to-end ownership: model → app → cloud deployment, not just a notebook
- Basic MLOps: reproducible environment (`requirements.txt`, `packages.txt`), cached model loading

## Possible Extensions

- Swap in a custom-trained YOLO model (e.g., fine-tuned on a niche class you care about)
- Add object tracking (ByteTrack/DeepSORT) for persistent IDs across frames
- Add a FastAPI backend to serve predictions to multiple frontends
