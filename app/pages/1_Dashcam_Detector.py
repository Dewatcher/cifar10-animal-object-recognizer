import os, cv2, subprocess, tempfile
import streamlit as st
from ultralytics import YOLO
import imageio_ffmpeg

st.set_page_config(page_title="Dashcam Hazard Detector", page_icon="🚘", layout="wide")
st.title("🚘 Dashcam Hazard Detector — Project Extension")
st.caption("Upgrade: single-image classification → multi-object detection on video. Keep clips ≤10s (free cloud CPU).")

BASE = os.path.dirname(os.path.abspath(__file__))
ROAD_CLASSES = [0, 1, 2, 3, 5, 7, 9, 11]

ADAS_HAZARD_MAP = {
    'person':        ('Vulnerable Road User (Emergency Warning)', 'error'),
    'car':           ('Vehicle Hazard (Proximity Tracking)', 'warning'),
    'motorcycle':    ('Vehicle Hazard (Proximity Tracking)', 'warning'),
    'bicycle':       ('Vehicle Hazard (Proximity Tracking)', 'warning'),
    'bus':           ('Vehicle Hazard (Proximity Tracking)', 'warning'),
    'truck':         ('Vehicle Hazard (Proximity Tracking)', 'warning'),
    'traffic light': ('Contextual Asset (Low-Risk Reference)', 'info'),
    'stop sign':     ('Contextual Asset (Low-Risk Reference)', 'info'),
}

@st.cache_resource
def load_model():
    return YOLO(os.path.join(BASE, "yolov8n.pt"))

model = load_model()
video_file = st.file_uploader("Choose a dashcam clip...", type=["mp4", "mov", "avi"])

if video_file is not None:
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_in.write(video_file.read()); tmp_in.close()

    cap = cv2.VideoCapture(tmp_in.name)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    raw_path = os.path.join(tempfile.gettempdir(), "annotated_raw.mp4")
    out = cv2.VideoWriter(raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    progress, hazards, counts, i = st.progress(0.0, "Detecting..."), {}, {}, 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        res = model(frame, classes=ROAD_CLASSES, imgsz=416, verbose=False)[0]
        for box in res.boxes:
            name, conf = model.names[int(box.cls)], float(box.conf)
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            counts[name] = counts.get(name, 0) + 1
            if y2 > 0.85 * h or (y2 - y1) > 0.40 * h:          # proximity heuristic
                hazards[name] = max(hazards.get(name, 0.0), conf)
        out.write(res.plot())                                   # boxes + labels drawn
        i += 1
        if total: progress.progress(min(i / total, 1.0))
    cap.release(); out.release(); progress.empty()

    # Re-encode to H.264 so browsers can actually play it
    final_path = raw_path.replace("_raw", "")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", raw_path,
                    "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "faststart", final_path], check=True, capture_output=True)

    st.subheader("Inference & Hazard Result")
    col1, col2 = st.columns(2)
    with col1: st.video(tmp_in.name); st.caption("Original footage")
    with col2: st.video(final_path);  st.caption("AI detection overlay")

    st.subheader("Hazard Alerts")
    if hazards:
        for name, conf in hazards.items():
            text, kind = ADAS_HAZARD_MAP.get(name, ('Unclassified Hazard', 'warning'))
            if kind == 'error':   st.error(f"🚨 **RED ALERT:** {text} — {name.upper()} @ {conf:.0%}")
            elif kind == 'warning': st.warning(f"⚠️ **YELLOW ALERT:** {text} — {name.upper()} @ {conf:.0%}")
            else: st.info(f"ℹ️ **NOTICE:** {text} — {name.upper()}")
    else:
        st.success("✅ **CLEAR:** No proximity hazards detected in this clip.")
