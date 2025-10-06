import streamlit as st
import tempfile
import os
import json
import yaml
from cover_drive_analysis_realtime import analyze_video

st.set_page_config(page_title="Cover Drive Analyzer", layout="wide")

st.title(" Cover Drive Analyzer")

# ---------------- Sidebar Controls ----------------
st.sidebar.header(" Settings")

handedness = st.sidebar.selectbox("Handedness", ["right", "left"], index=0)

st.sidebar.subheader("Pose Settings")
model_complexity = st.sidebar.slider("Model Complexity", 0, 2, 1)
min_det_conf = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5, 0.05)
min_track_conf = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5, 0.05)

st.sidebar.subheader("Metrics Thresholds")
elbow_good_min = st.sidebar.slider("Elbow Min (°)", 60, 150, 105)
elbow_good_max = st.sidebar.slider("Elbow Max (°)", 90, 170, 140)
spine_lean_max = st.sidebar.slider("Spine Lean Max (°)", 5, 45, 18)
head_over_toe_max = st.sidebar.slider("Head-Toe Max X (px)", 10, 100, 35)
foot_dir_max = st.sidebar.slider("Foot Dir Max (°)", 5, 60, 25)

st.sidebar.subheader("Scoring Weights")
w_footwork = st.sidebar.slider("Weight: Footwork", 0.0, 2.0, 1.0, 0.1)
w_head = st.sidebar.slider("Weight: Head Position", 0.0, 2.0, 1.0, 0.1)
w_swing = st.sidebar.slider("Weight: Swing Control", 0.0, 2.0, 1.0, 0.1)
w_balance = st.sidebar.slider("Weight: Balance", 0.0, 2.0, 1.0, 0.1)
w_follow = st.sidebar.slider("Weight: Follow Through", 0.0, 2.0, 1.0, 0.1)

st.sidebar.subheader("Grade Bands")
beginner_band = st.sidebar.text_input("Beginner (e.g. 0-40)", "0-40")
intermediate_band = st.sidebar.text_input("Intermediate (e.g. 40-70)", "40-70")
advanced_band = st.sidebar.text_input("Advanced (e.g. 70-100)", "70-100")

uploaded_file = st.file_uploader("Upload a cricket batting video", type=["mp4", "avi", "mov", "mkv"])

# ---------------- Processing ----------------
if uploaded_file:
    st.video(uploaded_file)

    if st.button("Run Analysis"):
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, uploaded_file.name)
            with open(input_path, "wb") as f:
                f.write(uploaded_file.read())

            # Build config dict
            cfg = {
                "pose": {
                    "model_complexity": model_complexity,
                    "min_detection_confidence": min_det_conf,
                    "min_tracking_confidence": min_track_conf,
                    "enable_segmentation": False,
                    "smooth_landmarks": True
                },
                "metrics": {
                    "elbow_good_min": elbow_good_min,
                    "elbow_good_max": elbow_good_max,
                    "spine_lean_max": spine_lean_max,
                    "head_over_toe_max_x_px": head_over_toe_max,
                    "foot_dir_max": foot_dir_max,
                },
                "smoothing": {"win": 5},
                "video": {
                    "width": 960,
                    "height": 540,
                    "target_fps": 25,
                    "fourcc": "mp4v"
                },
                "scoring": {
                    "weights": {
                        "footwork": w_footwork,
                        "head_position": w_head,
                        "swing_control": w_swing,
                        "balance": w_balance,
                        "follow_through": w_follow,
                    },
                    "grade_bands": {
                        "beginner": beginner_band,
                        "intermediate": intermediate_band,
                        "advanced": advanced_band,
                    }
                }
            }

            # Save config to temp file
            cfg_path = os.path.join(tmpdir, "config.yaml")
            with open(cfg_path, "w") as f:
                yaml.safe_dump(cfg, f)

            st.info(" Processing video... Please wait.")
            result = analyze_video(input_path, cfg_path=cfg_path, output_dir=tmpdir, handedness=handedness)
            st.success(" Processing complete!")

            # Show annotated video
            out_video = os.path.join(tmpdir, "annotated_video.mp4")
            if os.path.exists(out_video):
                st.video(out_video)
                with open(out_video, "rb") as f:
                    st.download_button(" Download Annotated Video", f, file_name="annotated_video.mp4")

            # Show evaluation JSON
            eval_path = os.path.join(tmpdir, "evaluation.json")
            if os.path.exists(eval_path):
                with open(eval_path, "r") as f:
                    eval_data = json.load(f)
                st.json(eval_data)
                with open(eval_path, "rb") as f:
                    st.download_button(" Download Evaluation JSON", f, file_name="evaluation.json")

            # Metrics CSV
            csv_path = os.path.join(tmpdir, "metrics.csv")
            if os.path.exists(csv_path):
                with open(csv_path, "rb") as f:
                    st.download_button("Download Metrics CSV", f, file_name="metrics.csv")

            # Charts
            st.subheader(" Charts")
            for chart_name in ["elbow_angle.png", "spine_lean.png"]:
                chart_path = os.path.join(tmpdir, chart_name)
                if os.path.exists(chart_path):
                    st.image(chart_path, caption=chart_name)
