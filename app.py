# # # import gradio as gr
# # # import tempfile
# # # import os
# # # import yaml
# # # import json
# # # import pandas as pd
# # # from cover_drive_analysis_realtime import analyze_video


# # # def run_analysis(
# # #     video,
# # #     handedness,
# # #     model_complexity,
# # #     min_det_conf,
# # #     min_track_conf,
# # #     elbow_good_min,
# # #     elbow_good_max,
# # #     spine_lean_max,
# # #     head_over_toe_max,
# # #     foot_dir_max,
# # #     w_footwork,
# # #     w_head,
# # #     w_swing,
# # #     w_balance,
# # #     w_follow,
# # #     beginner_band,
# # #     intermediate_band,
# # #     advanced_band,
# # # ):
# # #     with tempfile.TemporaryDirectory() as tmpdir:
# # #         # Save input video
# # #         input_path = os.path.join(tmpdir, "input.mp4")
# # #         with open(input_path, "wb") as f:
# # #             f.write(video.read())

# # #         # Build config dict
# # #         cfg = {
# # #             "pose": {
# # #                 "model_complexity": model_complexity,
# # #                 "min_detection_confidence": min_det_conf,
# # #                 "min_tracking_confidence": min_track_conf,
# # #                 "enable_segmentation": False,
# # #                 "smooth_landmarks": True,
# # #             },
# # #             "metrics": {
# # #                 "elbow_good_min": elbow_good_min,
# # #                 "elbow_good_max": elbow_good_max,
# # #                 "spine_lean_max": spine_lean_max,
# # #                 "head_over_toe_max_x_px": head_over_toe_max,
# # #                 "foot_dir_max": foot_dir_max,
# # #             },
# # #             "smoothing": {"win": 5},
# # #             "video": {"width": 960, "height": 540, "target_fps": 25, "fourcc": "mp4v"},
# # #             "scoring": {
# # #                 "weights": {
# # #                     "footwork": w_footwork,
# # #                     "head_position": w_head,
# # #                     "swing_control": w_swing,
# # #                     "balance": w_balance,
# # #                     "follow_through": w_follow,
# # #                 },
# # #                 "grade_bands": {
# # #                     "beginner": beginner_band,
# # #                     "intermediate": intermediate_band,
# # #                     "advanced": advanced_band,
# # #                 },
# # #             },
# # #         }

# # #         cfg_path = os.path.join(tmpdir, "config.yaml")
# # #         with open(cfg_path, "w") as f:
# # #             yaml.safe_dump(cfg, f)

# # #         # Run analysis
# # #         result = analyze_video(
# # #             input_path, cfg_path=cfg_path, output_dir=tmpdir, handedness=handedness
# # #         )

# # #         # Collect outputs
# # #         out_video = os.path.join(tmpdir, "annotated_video.mp4")
# # #         eval_json = os.path.join(tmpdir, "evaluation.json")
# # #         csv_path = os.path.join(tmpdir, "metrics.csv")
# # #         elbow_chart = os.path.join(tmpdir, "elbow_angle.png")
# # #         spine_chart = os.path.join(tmpdir, "spine_lean.png")

# # #         eval_data = {}
# # #         if os.path.exists(eval_json):
# # #             with open(eval_json, "r") as f:
# # #                 eval_data = json.load(f)

# # #         metrics_df = None
# # #         if os.path.exists(csv_path):
# # #             metrics_df = pd.read_csv(csv_path)

# # #         return (
# # #             out_video if os.path.exists(out_video) else None,  # video
# # #             eval_data,  # JSON dict
# # #             csv_path if os.path.exists(csv_path) else None,  # download CSV
# # #             metrics_df,  # show CSV as dataframe
# # #             elbow_chart if os.path.exists(elbow_chart) else None,
# # #             spine_chart if os.path.exists(spine_chart) else None,
# # #         )


# # # # Gradio interface
# # # iface = gr.Interface(
# # #     fn=run_analysis,
# # #     inputs=[
# # #         gr.File(type="filepath", file_types=[".mp4", ".avi", ".mov", ".mkv"], label="Upload Video"),
# # #         gr.Radio(["right", "left"], value="right", label="Handedness"),
# # #         gr.Slider(0, 2, 1, step=1, label="Model Complexity"),
# # #         gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Min Detection Confidence"),
# # #         gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Min Tracking Confidence"),
# # #         gr.Slider(60, 150, 105, step=1, label="Elbow Min (°)"),
# # #         gr.Slider(90, 170, 140, step=1, label="Elbow Max (°)"),
# # #         gr.Slider(5, 45, 18, step=1, label="Spine Lean Max (°)"),
# # #         gr.Slider(10, 100, 35, step=1, label="Head-Toe Max X (px)"),
# # #         gr.Slider(5, 60, 25, step=1, label="Foot Dir Max (°)"),
# # #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Footwork"),
# # #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Head Position"),
# # #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Swing Control"),
# # #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Balance"),
# # #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Follow Through"),
# # #         gr.Textbox("0-40", label="Beginner Band"),
# # #         gr.Textbox("40-70", label="Intermediate Band"),
# # #         gr.Textbox("70-100", label="Advanced Band"),
# # #     ],
# # #     outputs=[
# # #         gr.Video(label="Annotated Video"),
# # #         gr.JSON(label="Evaluation JSON"),
# # #         gr.File(label="Download Metrics CSV"),
# # #         gr.Dataframe(label="Metrics Table"),
# # #         gr.Image(label="Elbow Angle Chart"),
# # #         gr.Image(label="Spine Lean Chart"),
# # #     ],
# # #     title="Cover Drive Analyzer",
# # #     description="Upload a batting video to get annotated feedback, metrics, and scores.",
# # # )

# # # if __name__ == "__main__":
# # #     iface.launch()



# # import gradio as gr
# # import tempfile
# # import os
# # import yaml
# # import json
# # import pandas as pd
# # import shutil
# # from cover_drive_analysis_realtime import analyze_video


# # def run_analysis(
# #     video_path,
# #     handedness,
# #     model_complexity,
# #     min_det_conf,
# #     min_track_conf,
# #     elbow_good_min,
# #     elbow_good_max,
# #     spine_lean_max,
# #     head_over_toe_max,
# #     foot_dir_max,
# #     w_footwork,
# #     w_head,
# #     w_swing,
# #     w_balance,
# #     w_follow,
# #     beginner_band,
# #     intermediate_band,
# #     advanced_band,
# # ):
# #     with tempfile.TemporaryDirectory() as tmpdir:
# #         # Copy uploaded video to tmpdir
# #         tmpdir = tempfile.mkdtemp()  # stays after function ends
# #         input_path = os.path.join(tmpdir, "input.mp4")
# #         shutil.copy(video_path, input_path)

# #         # Build config dict
# #         cfg = {
# #             "pose": {
# #                 "model_complexity": model_complexity,
# #                 "min_detection_confidence": min_det_conf,
# #                 "min_tracking_confidence": min_track_conf,
# #                 "enable_segmentation": False,
# #                 "smooth_landmarks": True,
# #             },
# #             "metrics": {
# #                 "elbow_good_min": elbow_good_min,
# #                 "elbow_good_max": elbow_good_max,
# #                 "spine_lean_max": spine_lean_max,
# #                 "head_over_toe_max_x_px": head_over_toe_max,
# #                 "foot_dir_max": foot_dir_max,
# #             },
# #             "smoothing": {"win": 5},
# #             "video": {"width": 960, "height": 540, "target_fps": 25, "fourcc": "mp4v"},
# #             "scoring": {
# #                 "weights": {
# #                     "footwork": w_footwork,
# #                     "head_position": w_head,
# #                     "swing_control": w_swing,
# #                     "balance": w_balance,
# #                     "follow_through": w_follow,
# #                 },
# #                 "grade_bands": {
# #                     "beginner": beginner_band,
# #                     "intermediate": intermediate_band,
# #                     "advanced": advanced_band,
# #                 },
# #             },
# #         }

# #         cfg_path = os.path.join(tmpdir, "config.yaml")
# #         with open(cfg_path, "w") as f:
# #             yaml.safe_dump(cfg, f)

# #         # Run analysis
# #         result = analyze_video(
# #             input_path, cfg_path=cfg_path, output_dir=tmpdir, handedness=handedness
# #         )

# #         # Collect outputs
# #         out_video = os.path.join(tmpdir, "annotated_video.mp4")
# #         eval_json = os.path.join(tmpdir, "evaluation.json")
# #         csv_path = os.path.join(tmpdir, "metrics.csv")
# #         elbow_chart = os.path.join(tmpdir, "elbow_angle.png")
# #         spine_chart = os.path.join(tmpdir, "spine_lean.png")

# #         eval_data = {}
# #         if os.path.exists(eval_json):
# #             with open(eval_json, "r") as f:
# #                 eval_data = json.load(f)

# #         metrics_df = None
# #         if os.path.exists(csv_path):
# #             metrics_df = pd.read_csv(csv_path)

# #         return (
# #             out_video if os.path.exists(out_video) else None,  # video
# #             eval_data,  # JSON dict
# #             csv_path if os.path.exists(csv_path) else None,  # download CSV
# #             metrics_df,  # show CSV table
# #             elbow_chart if os.path.exists(elbow_chart) else None,
# #             spine_chart if os.path.exists(spine_chart) else None,
# #         )


# # # Gradio interface
# # iface = gr.Interface(
# #     fn=run_analysis,
# #     inputs=[
# #         gr.File(type="filepath", file_types=[".mp4", ".avi", ".mov", ".mkv"], label="Upload Video"),
# #         gr.Radio(["right", "left"], value="right", label="Handedness"),
# #         gr.Slider(0, 2, 1, step=1, label="Model Complexity"),
# #         gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Min Detection Confidence"),
# #         gr.Slider(0.0, 1.0, 0.5, step=0.05, label="Min Tracking Confidence"),
# #         gr.Slider(60, 150, 105, step=1, label="Elbow Min (°)"),
# #         gr.Slider(90, 170, 140, step=1, label="Elbow Max (°)"),
# #         gr.Slider(5, 45, 18, step=1, label="Spine Lean Max (°)"),
# #         gr.Slider(10, 100, 35, step=1, label="Head-Toe Max X (px)"),
# #         gr.Slider(5, 60, 25, step=1, label="Foot Dir Max (°)"),
# #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Footwork"),
# #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Head Position"),
# #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Swing Control"),
# #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Balance"),
# #         gr.Slider(0.0, 2.0, 1.0, step=0.1, label="Weight: Follow Through"),
# #         gr.Textbox("0-40", label="Beginner Band"),
# #         gr.Textbox("40-70", label="Intermediate Band"),
# #         gr.Textbox("70-100", label="Advanced Band"),
# #     ],
# #     outputs=[
# #         gr.Video(label="Annotated Video"),
# #         gr.JSON(label="Evaluation JSON"),
# #         gr.File(label="Download Metrics CSV"),
# #         gr.Dataframe(label="Metrics Table"),
# #         gr.Image(label="Elbow Angle Chart"),
# #         gr.Image(label="Spine Lean Chart"),
# #     ],
# #     title="🏏 Cover Drive Analyzer",
# #     description="Upload a batting video to get annotated feedback, metrics, and scores.",
# # )

# # if __name__ == "__main__":
# #     iface.launch()


# import gradio as gr
# import os
# import tempfile
# import shutil
# import json
# import pandas as pd
# from pathlib import Path
# import yaml

# # Import the analysis functions from the original code
# from cover_drive_analysis_realtime import analyze_video, load_yaml

# def create_default_config():
#     """Create a default configuration for the analysis"""
#     config = {
#         "pose": {
#             "model_complexity": 1,
#             "enable_segmentation": False,
#             "smooth_landmarks": True,
#             "min_detection_confidence": 0.5,
#             "min_tracking_confidence": 0.5
#         },
#         "metrics": {
#             "elbow_good_min": 105,
#             "elbow_good_max": 140,
#             "spine_lean_max": 18,
#             "head_over_toe_max_x_px": 35,
#             "foot_dir_max": 25
#         },
#         "smoothing": {
#             "win": 5
#         },
#         "video": {
#             "width": 960,
#             "height": 540,
#             "target_fps": 25,
#             "fourcc": "mp4v"
#         },
#         "scoring": {
#             "weights": {
#                 "footwork": 0.2,
#                 "head_position": 0.2,
#                 "swing_control": 0.25,
#                 "balance": 0.15,
#                 "follow_through": 0.2
#             },
#             "grade_bands": {
#                 "beginner": "1.0-5.0",
#                 "intermediate": "5.1-7.5",
#                 "advanced": "7.6-10.0"
#             }
#         }
#     }
#     return config

# def process_cricket_video(video_file, handedness, progress=gr.Progress()):
#     """Process the uploaded cricket video and return analysis results"""
    
#     if video_file is None:
#         return None, None, None, None, "Please upload a video file first."
    
#     progress(0, desc="Setting up analysis...")
    
#     # Create temporary directories
#     temp_dir = tempfile.mkdtemp()
#     output_dir = os.path.join(temp_dir, "output")
    
#     try:
#         # Create config file
#         config = create_default_config()
#         config_path = os.path.join(temp_dir, "config.yaml")
#         with open(config_path, 'w') as f:
#             yaml.dump(config, f)
        
#         progress(0.1, desc="Starting video analysis...")
        
#         # Run the analysis
#         result = analyze_video(
#             input_path=video_file,
#             cfg_path=config_path,
#             output_dir=output_dir,
#             handedness=handedness
#         )
        
#         progress(0.8, desc="Generating outputs...")
        
#         # Prepare outputs
#         annotated_video = os.path.join(output_dir, "annotated_video.mp4")
#         evaluation_json = os.path.join(output_dir, "evaluation.json")
#         metrics_csv = os.path.join(output_dir, "metrics.csv")
        
#         # Create analysis summary
#         summary = create_analysis_summary(result)
        
#         # Load metrics for display
#         metrics_df = None
#         if os.path.exists(metrics_csv):
#             metrics_df = pd.read_csv(metrics_csv)
        
#         progress(1.0, desc="Analysis complete!")
        
#         return (
#             annotated_video if os.path.exists(annotated_video) else None,
#             evaluation_json if os.path.exists(evaluation_json) else None,
#             metrics_csv if os.path.exists(metrics_csv) else None,
#             metrics_df,
#             summary
#         )
        
#     except Exception as e:
#         return None, None, None, None, f"Error processing video: {str(e)}"
    
#     finally:
#         # Clean up temporary files (optional - comment out for debugging)
#         # shutil.rmtree(temp_dir, ignore_errors=True)
#         pass

# def create_analysis_summary(result):
#     """Create a readable summary of the analysis results"""
#     if not result:
#         return "No analysis results available."
    
#     summary = f"""
# # Cricket Cover Drive Analysis Results

# ## Overall Performance
# - **Score**: {result.get('overall', 'N/A')}/10
# - **Grade**: {result.get('grade', 'N/A').title()}
# - **Processing FPS**: {result.get('avg_fps', 'N/A')}

# ## Detailed Scores
# """
    
#     scores = result.get('scores', {})
#     comments = result.get('comments', {})
    
#     for metric, score in scores.items():
#         metric_name = metric.replace('_', ' ').title()
#         comment = comments.get(metric, 'No specific feedback available.')
#         summary += f"- **{metric_name}**: {score:.1f}/10\n  - *{comment}*\n\n"
    
#     # Add phases information if available
#     phases = result.get('phases', {})
#     if phases:
#         summary += "\n## Swing Phases Detected\n"
#         for phase, frame_range in phases.items():
#             if frame_range and len(frame_range) == 2:
#                 summary += f"- **{phase.replace('_', ' ').title()}**: Frames {frame_range[0]} - {frame_range[1]}\n"
    
#     return summary

# def create_gradio_interface():
#     """Create the Gradio interface"""
    
#     with gr.Blocks(title="Cricket Cover Drive Analysis", theme=gr.themes.Soft()) as demo:
#         gr.Markdown("""
#         # 🏏 Cricket Cover Drive Analysis
        
#         Upload a cricket video to analyze your cover drive technique. The system will:
#         - Track your pose and movements
#         - Analyze key metrics like elbow angle, spine lean, head position, and footwork
#         - Provide scores and feedback on your technique
#         - Generate an annotated video showing your form
        
#         **Supported formats**: MP4, AVI, MOV
#         """)
        
#         with gr.Row():
#             with gr.Column(scale=1):
#                 # Input section
#                 gr.Markdown("## Upload & Settings")
                
#                 video_input = gr.File(
#                     label="Upload Cricket Video",
#                     file_types=["video"],
#                     file_count="single"
#                 )
                
#                 handedness = gr.Radio(
#                     choices=["right", "left"],
#                     value="right",
#                     label="Batting Hand",
#                     info="Select your dominant batting hand"
#                 )
                
#                 analyze_btn = gr.Button(
#                     "🔍 Analyze Video",
#                     variant="primary",
#                     size="lg"
#                 )
            
#             with gr.Column(scale=2):
#                 # Output section
#                 gr.Markdown("## Analysis Results")
                
#                 analysis_summary = gr.Markdown(
#                     value="Upload a video and click 'Analyze Video' to see results here.",
#                     label="Summary"
#                 )
        
#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown("## Annotated Video")
#                 annotated_video = gr.Video(
#                     label="Processed Video with Annotations",
#                     show_download_button=True
#                 )
            
#             with gr.Column():
#                 gr.Markdown("## Metrics Over Time")
#                 metrics_plot = gr.Dataframe(
#                     label="Detailed Metrics",
#                     interactive=False
#                 )
        
#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown("## Download Results")
                
#                 evaluation_file = gr.File(
#                     label="📊 Evaluation Report (JSON)",
#                     visible=False
#                 )
                
#                 metrics_file = gr.File(
#                     label="📈 Metrics Data (CSV)",
#                     visible=False
#                 )
        
#         # Event handlers
#         analyze_btn.click(
#             fn=process_cricket_video,
#             inputs=[video_input, handedness],
#             outputs=[
#                 annotated_video,
#                 evaluation_file,
#                 metrics_file,
#                 metrics_plot,
#                 analysis_summary
#             ],
#             show_progress=True
#         ).then(
#             fn=lambda x, y: (gr.File(visible=x is not None), gr.File(visible=y is not None)),
#             inputs=[evaluation_file, metrics_file],
#             outputs=[evaluation_file, metrics_file]
#         )
        
#         # Examples section
#         gr.Markdown("""
#         ## 📋 Tips for Best Results
        
#         1. **Video Quality**: Use clear, well-lit videos
#         2. **Camera Angle**: Side-on view works best for cover drive analysis
#         3. **Full Body**: Ensure the full body is visible throughout the shot
#         4. **Stable Camera**: Minimize camera movement for better tracking
#         5. **Video Length**: 3-10 seconds of the cover drive motion is ideal
        
#         ## 📊 Metrics Explained
        
#         - **Footwork**: Alignment of front foot with the crease
#         - **Head Position**: How well head is positioned over front foot
#         - **Swing Control**: Smoothness of elbow movement through the shot
#         - **Balance**: Spine angle and overall body balance
#         - **Follow Through**: Completion and control of the shot
#         """)
    
#     return demo

# # Additional utility functions for the Gradio app
# def validate_video_file(file_path):
#     """Validate if the uploaded file is a valid video"""
#     valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
#     file_ext = Path(file_path).suffix.lower()
#     return file_ext in valid_extensions

# if __name__ == "__main__":
#     # Create and launch the Gradio interface
#     demo = create_gradio_interface()
#     demo.launch(
#         server_name="0.0.0.0",
#         server_port=7860,
#         share=True,  # Set to True to create a public link
#         debug=True
#     )
import gradio as gr
import os
import tempfile
import shutil
import json
import pandas as pd
import numpy as np
import cv2
import math
import time
import yaml
import csv
from pathlib import Path
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

try:
    import mediapipe as mp
except ImportError:
    mp = None
    print("Warning: mediapipe not installed. Please install with: pip install mediapipe")

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
    print("Warning: matplotlib not installed. Charts will not be generated.")

# Core analysis functions (fixed versions)
def angle_abc(a, b, c):
    """Calculate angle between three points"""
    if a is None or b is None or c is None: 
        return None
    try:
        ab, cb = a-b, c-b
        nab, ncb = np.linalg.norm(ab), np.linalg.norm(cb)
        if nab < 1e-6 or ncb < 1e-6: 
            return None
        cosang = np.dot(ab, cb) / (nab*ncb)
        cosang = np.clip(cosang, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosang)))
    except:
        return None

def line_to_vertical_angle(pt_top, pt_bottom):
    """Angle between line and vertical"""
    if pt_top is None or pt_bottom is None: 
        return None
    try:
        dx, dy = pt_bottom[0]-pt_top[0], pt_bottom[1]-pt_top[1]
        if abs(dy) < 1e-6 and abs(dx) < 1e-6: 
            return None
        angle_line = math.degrees(math.atan2(dy, dx))
        return float(abs(90.0 - angle_line))
    except:
        return None

def foot_direction(ankle, toe):
    """Calculate foot direction angle"""
    if ankle is None or toe is None: 
        return None
    try:
        v = toe - ankle
        if np.linalg.norm(v) < 1e-6: 
            return None
        ang = math.degrees(math.atan2(v[1], v[0]))
        return float(abs(180.0 - ang))
    except:
        return None

def projected_x_dist(p1, p2):
    """X-axis distance between points"""
    if p1 is None or p2 is None: 
        return None
    try:
        return float(abs(p1[0]-p2[0]))
    except:
        return None

def moving_average(arr: List[float], win: int) -> List[float]:
    """Calculate moving average"""
    if win <= 1 or len(arr) <= 1:
        return arr
    out = []
    q = deque(maxlen=win)
    for v in arr:
        q.append(v)
        out.append(sum(q)/len(q))
    return out

class PoseEstimator:
    def __init__(self, cfg: dict):
        if mp is None:
            raise RuntimeError("mediapipe not available. Install mediapipe first.")
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            model_complexity=cfg["pose"].get("model_complexity", 1),
            enable_segmentation=cfg["pose"].get("enable_segmentation", False),
            smooth_landmarks=cfg["pose"].get("smooth_landmarks", True),
            min_detection_confidence=cfg["pose"].get("min_detection_confidence", 0.5),
            min_tracking_confidence=cfg["pose"].get("min_tracking_confidence", 0.5),
        )
        self.idx = self.mp_pose.PoseLandmark

    def _get_xy(self, lm, w, h):
        """Convert landmark to image coordinates"""
        try:
            return np.array([lm.x*w, lm.y*h, lm.z*w], dtype=np.float32)
        except:
            return None

    def extract(self, frame_bgr) -> Dict[str, Optional[np.ndarray]]:
        """Extract pose landmarks from frame"""
        try:
            h, w = frame_bgr.shape[:2]
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self.pose.process(rgb)
            
            if not res.pose_landmarks:
                return defaultdict(lambda: None)
            
            lms = res.pose_landmarks.landmark
            get = lambda landmark: self._get_xy(lms[landmark], w, h)
            
            J = defaultdict(lambda: None)
            try:
                J["head"] = get(self.idx.NOSE)
                J["l_shoulder"] = get(self.idx.LEFT_SHOULDER)
                J["r_shoulder"] = get(self.idx.RIGHT_SHOULDER)
                J["l_elbow"] = get(self.idx.LEFT_ELBOW)
                J["r_elbow"] = get(self.idx.RIGHT_ELBOW)
                J["l_wrist"] = get(self.idx.LEFT_WRIST)
                J["r_wrist"] = get(self.idx.RIGHT_WRIST)
                J["l_hip"] = get(self.idx.LEFT_HIP)
                J["r_hip"] = get(self.idx.RIGHT_HIP)
                J["l_knee"] = get(self.idx.LEFT_KNEE)
                J["r_knee"] = get(self.idx.RIGHT_KNEE)
                J["l_ankle"] = get(self.idx.LEFT_ANKLE)
                J["r_ankle"] = get(self.idx.RIGHT_ANKLE)
                J["l_foot_index"] = get(self.idx.LEFT_FOOT_INDEX)
                J["r_foot_index"] = get(self.idx.RIGHT_FOOT_INDEX)
            except Exception as e:
                print(f"Warning: Error extracting some landmarks: {e}")
            
            return J
        except Exception as e:
            print(f"Error in pose extraction: {e}")
            return defaultdict(lambda: None)

class MetricTracker:
    def __init__(self, cfg: dict):
        m = cfg["metrics"]
        self.th = {
            "elbow_good_min": m.get("elbow_good_min", 105),
            "elbow_good_max": m.get("elbow_good_max", 140),
            "spine_lean_max": m.get("spine_lean_max", 18),
            "head_over_toe_max_x_px": m.get("head_over_toe_max_x_px", 35),
            "foot_dir_max": m.get("foot_dir_max", 25),
        }
        self.win = cfg["smoothing"].get("win", 5)
        
        self.logs = {
            "elbow_angle": [],
            "spine_lean": [],
            "head_toe_x": [],
            "foot_dir": [],
            "time": [],
        }

    def compute(self, J: Dict[str, Optional[np.ndarray]], handedness: str = "right") -> Dict[str, Optional[float]]:
        """Compute metrics from joint positions"""
        try:
            # Choose side based on handedness
            if handedness == "right":
                shoulder = J.get("l_shoulder")
                elbow = J.get("l_elbow") 
                wrist = J.get("l_wrist")
                hip_top = J.get("l_shoulder")
                hip_bottom = J.get("l_hip")
                ankle = J.get("l_ankle")
                toe = J.get("l_foot_index")
            else:
                shoulder = J.get("r_shoulder")
                elbow = J.get("r_elbow")
                wrist = J.get("r_wrist") 
                hip_top = J.get("r_shoulder")
                hip_bottom = J.get("r_hip")
                ankle = J.get("r_ankle")
                toe = J.get("r_foot_index")

            head = J.get("head")

            elbow_angle = angle_abc(shoulder, elbow, wrist)
            spine_lean = line_to_vertical_angle(hip_top, hip_bottom)
            head_toe_x = projected_x_dist(head, toe)
            foot_dir_ang = foot_direction(ankle, toe)

            return {
                "elbow_angle": elbow_angle,
                "spine_lean": spine_lean,
                "head_toe_x": head_toe_x,
                "foot_dir": foot_dir_ang,
            }
        except Exception as e:
            print(f"Error computing metrics: {e}")
            return {
                "elbow_angle": None,
                "spine_lean": None, 
                "head_toe_x": None,
                "foot_dir": None,
            }

    def log(self, t: float, metrics: Dict[str, Optional[float]]):
        """Log metrics with timestamp"""
        for k in ["elbow_angle","spine_lean","head_toe_x","foot_dir"]:
            self.logs[k].append(metrics.get(k, None))
        self.logs["time"].append(t)

    def smoothed(self) -> Dict[str, List[Optional[float]]]:
        """Apply smoothing to logged metrics"""
        out = {}
        for k, arr in self.logs.items():
            if k == "time":
                out[k] = arr
                continue
                
            # Replace None with previous valid value
            clean = []
            last = None
            for v in arr:
                if v is None:
                    clean.append(last if last is not None else None)
                else:
                    clean.append(v)
                    last = v

            # Apply moving average
            nums = [x for x in clean if x is not None]
            if not nums:
                out[k] = clean
                continue
                
            # Forward fill then smooth
            ff = []
            last = nums[0]
            for v in clean:
                if v is None:
                    ff.append(last)
                else:
                    last = v
                    ff.append(v)
            out[k] = moving_average(ff, self.win)
        return out

class MetricLogger:
    def __init__(self):
        self.rows = []

    def log(self, t, metrics):
        row = {"time": t}
        row.update(metrics)
        self.rows.append(row)

    def to_csv(self, path):
        if not self.rows: 
            return
        keys = self.rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(r)

    def to_charts(self, out_dir):
        if plt is None or not self.rows: 
            return
        times = [r["time"] for r in self.rows]
        for key in ["elbow_angle", "spine_lean"]:
            vals = [r.get(key) for r in self.rows]
            plt.figure()
            plt.plot(times, vals, label=key)
            plt.xlabel("Time (s)")
            plt.ylabel(key)
            plt.title(key + " over time")
            plt.legend()
            plt.savefig(os.path.join(out_dir, f"{key}.png"))
            plt.close()

# Drawing functions
GREEN = (40, 200, 40)
RED = (40, 40, 220)
WHITE = (245, 245, 245)
YELLOW = (30, 220, 220)

def draw_pose(frame, J: Dict[str, Optional[np.ndarray]]):
    """Draw pose skeleton on frame"""
    try:
        pairs = [
            ("l_shoulder","l_elbow"), ("l_elbow","l_wrist"),
            ("r_shoulder","r_elbow"), ("r_elbow","r_wrist"),
            ("l_shoulder","l_hip"), ("r_shoulder","r_hip"),
            ("l_hip","l_knee"), ("l_knee","l_ankle"),
            ("r_hip","r_knee"), ("r_knee","r_ankle"),
            ("l_ankle","l_foot_index"), ("r_ankle","r_foot_index"),
            ("l_shoulder","r_shoulder"), ("l_hip","r_hip")
        ]
        
        for a, b in pairs:
            pa, pb = J.get(a), J.get(b)
            if pa is not None and pb is not None:
                try:
                    pa_2d = pa[:2].astype(int)
                    pb_2d = pb[:2].astype(int)
                    cv2.line(frame, tuple(pa_2d), tuple(pb_2d), WHITE, 2)
                except:
                    continue
        
        for k, pt in J.items():
            if pt is not None:
                try:
                    pt_2d = pt[:2].astype(int)
                    cv2.circle(frame, tuple(pt_2d), 3, YELLOW, -1)
                except:
                    continue
    except Exception as e:
        print(f"Warning: Error drawing pose: {e}")

def put_metric_text(frame, x, y, label, value, unit="", good: Optional[bool]=None):
    """Display metric text on frame"""
    try:
        text = f"{label}: {value if value is not None else '—'}{unit}"
        color = GREEN if good is True else (RED if good is False else WHITE)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    except:
        pass

def cue(frame, x, y, ok: bool, msg_ok="Good", msg_bad="Check"):
    """Display coaching cue on frame"""
    try:
        msg = f" {msg_ok}" if ok else f" {msg_bad}"
        color = GREEN if ok else RED
        cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    except:
        pass

def evaluate(cfg: dict, tracks: Dict[str, List[Optional[float]]]) -> Dict[str, object]:
    """Evaluate performance and generate scores"""
    th = cfg["metrics"]

    def score_footwork(foot_dir):
        valid = [x for x in foot_dir if x is not None]
        if not valid: return 5.0, "Insufficient foot detection."
        dev = np.median([abs(x) for x in valid])
        s = np.interp(th["foot_dir_max"] - dev, [0, th["foot_dir_max"]], [4, 10])
        tip = f"Keep front foot aligned within ~{th['foot_dir_max']}°"
        return float(np.clip(s, 1, 10)), tip

    def score_head_position(head_toe_x):
        valid = [x for x in head_toe_x if x is not None]
        if not valid: return 5.0, "Insufficient head/toe visibility."
        dev = np.median(valid)
        s = np.interp(th["head_over_toe_max_x_px"] - dev, [0, th["head_over_toe_max_x_px"]], [4, 10])
        tip = "Bring head closer over the front foot at impact."
        return float(np.clip(s, 1, 10)), tip

    def score_swing_control(elbow_angle):
        valid = [x for x in elbow_angle if x is not None]
        if not valid: return 5.0, "Insufficient elbow visibility."
        var = np.var(valid)
        s = np.interp(30 - min(var, 30), [0, 30], [4, 10])
        tip = "Maintain smooth elbow trajectory through downswing."
        return float(np.clip(s, 1, 10)), tip

    def score_balance(spine_lean):
        valid = [x for x in spine_lean if x is not None]
        if not valid: return 5.0, "Insufficient torso visibility."
        dev = np.median([abs(x) for x in valid])
        s = np.interp(th["spine_lean_max"] - dev, [0, th["spine_lean_max"]], [4, 10])
        tip = "Keep spine closer to vertical for better balance."
        return float(np.clip(s, 1, 10)), tip

    def score_follow_through(elbow_angle):
        valid = [x for x in elbow_angle if x is not None]
        if len(valid) < 5: return 5.0, "Limited follow-through data."
        tail = valid[-max(5, len(valid)//5):]
        in_range = [th["elbow_good_min"] <= x <= th["elbow_good_max"] for x in tail]
        frac = sum(in_range)/len(tail)
        s = 4 + 6*frac
        tip = "Finish high with controlled elbow range."
        return float(np.clip(s, 1, 10)), tip

    scores = {}
    comments = {}
    s, c = score_footwork(tracks["foot_dir"]); scores["footwork"]=s; comments["footwork"]=c
    s, c = score_head_position(tracks["head_toe_x"]); scores["head_position"]=s; comments["head_position"]=c
    s, c = score_swing_control(tracks["elbow_angle"]); scores["swing_control"]=s; comments["swing_control"]=c
    s, c = score_balance(tracks["spine_lean"]); scores["balance"]=s; comments["balance"]=c
    s, c = score_follow_through(tracks["elbow_angle"]); scores["follow_through"]=s; comments["follow_through"]=c

    w = cfg["scoring"]["weights"]
    overall = sum(scores[k]*w[k] for k in w.keys())
    
    grade = "beginner"
    bands = cfg["scoring"]["grade_bands"]
    lo_b, hi_b = map(float, bands["beginner"].split('-'))
    lo_i, hi_i = map(float, bands["intermediate"].split('-'))
    lo_a, hi_a = map(float, bands["advanced"].split('-'))
    if overall >= lo_a: grade = "advanced"
    elif overall >= lo_i: grade = "intermediate"

    return {
        "scores": scores,
        "overall": round(float(overall), 2),
        "grade": grade,
        "comments": comments,
    }

def create_default_config():
    """Create a default configuration for the analysis"""
    config = {
        "pose": {
            "model_complexity": 1,
            "enable_segmentation": False,
            "smooth_landmarks": True,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5
        },
        "metrics": {
            "elbow_good_min": 105,
            "elbow_good_max": 140,
            "spine_lean_max": 18,
            "head_over_toe_max_x_px": 35,
            "foot_dir_max": 25
        },
        "smoothing": {
            "win": 5
        },
        "video": {
            "width": 1080,
            "height": 1920,
            "target_fps": 25,
            "fourcc": "mp4v"
        },
        "scoring": {
            "weights": {
                "footwork": 0.2,
                "head_position": 0.2,
                "swing_control": 0.25,
                "balance": 0.15,
                "follow_through": 0.2
            },
            "grade_bands": {
                "beginner": "1.0-5.0",
                "intermediate": "5.1-7.5",
                "advanced": "7.6-10.0"
            }
        }
    }
    return config

def analyze_video_safe(input_path: str, cfg: dict, output_dir: str = "./output", handedness: str = "right"):
    """Safe video analysis with comprehensive error handling"""
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Video IO
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {input_path}")
            return None

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
        
        if total_frames <= 0:
            print("Error: Video has no frames or cannot read frame count")
            cap.release()
            return None

        W = int(cfg["video"].get("width", 960))
        H = int(cfg["video"].get("height", 540))
        TGT_FPS = int(cfg["video"].get("target_fps", 25))
        fourcc = cv2.VideoWriter_fourcc(*cfg["video"].get("fourcc", "mp4v"))
        out_path = os.path.join(output_dir, "annotated_video.mp4")
        writer = cv2.VideoWriter(out_path, fourcc, TGT_FPS, (W, H))

        pose = PoseEstimator(cfg)
        tracker = MetricTracker(cfg)
        logger = MetricLogger()

        frame_interval = 1.0 / max(src_fps, 1e-6)
        t0 = time.time()
        frames_proc = 0
        successful_detections = 0

        print(f"Processing {total_frames} frames...")

        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            try:
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                
                # Extract pose
                J = pose.extract(frame)
                
                # Check if we got valid pose data
                valid_joints = sum(1 for v in J.values() if v is not None)
                if valid_joints > 5:  # Need at least some key joints
                    successful_detections += 1

                # Compute metrics
                m = tracker.compute(J, handedness=handedness)
                
                # Log metrics
                current_time = frames_proc * frame_interval
                tracker.log(current_time, m)
                logger.log(current_time, m)
                
                # Draw pose
                draw_pose(frame, J)
                
                # Display metrics
                th = tracker.th
                elbow = m.get("elbow_angle")
                spine = m.get("spine_lean")
                hk = m.get("head_toe_x")
                fd = m.get("foot_dir")

                good_elbow = (elbow is not None) and (th["elbow_good_min"] <= elbow <= th["elbow_good_max"])
                good_spine = (spine is not None) and (abs(spine) <= th["spine_lean_max"])
                good_head = (hk is not None) and (hk <= th["head_over_toe_max_x_px"])
                good_foot = (fd is not None) and (abs(fd) <= th["foot_dir_max"])

                put_metric_text(frame, 15, 30, "Elbow", None if elbow is None else round(elbow,1), "°", good_elbow)
                put_metric_text(frame, 15, 60, "Spine", None if spine is None else round(spine,1), "°", good_spine)
                put_metric_text(frame, 15, 90, "Head-Toe X", None if hk is None else int(hk), "px", good_head)
                put_metric_text(frame, 15, 120, "Foot Dir", None if fd is None else round(fd,1), "°", good_foot)

                cue(frame, 15, 200, good_elbow, "Good elbow", "Adjust elbow")
                cue(frame, 15, 230, good_head, "Head positioned", "Move head over foot")
                cue(frame, 15, 260, good_spine, "Good balance", "Reduce lean")

                writer.write(frame)
                frames_proc += 1
                
            except Exception as e:
                print(f"Warning: Error processing frame {i}: {e}")
                # Write blank frame to maintain video continuity
                blank_frame = np.zeros((H, W, 3), dtype=np.uint8)
                cv2.putText(blank_frame, f"Processing error", (50, H//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                writer.write(blank_frame)
                frames_proc += 1

        cap.release()
        writer.release()

        print(f"Successfully detected poses in {successful_detections}/{frames_proc} frames")

        if successful_detections < frames_proc * 0.1:
            print("Warning: Very low pose detection rate. Results may be unreliable.")
            print("Tips: Ensure good lighting, clear view of player, minimal background clutter")

        # Generate results
        tracks = tracker.smoothed()
        result = evaluate(cfg, tracks)

        # Save results
        eval_path = os.path.join(output_dir, "evaluation.json")
        with open(eval_path, "w") as f:
            json.dump(result, f, indent=2)

        csv_path = os.path.join(output_dir, "metrics.csv")
        logger.to_csv(csv_path)
        
        try:
            logger.to_charts(output_dir)
        except Exception as e:
            print(f"Warning: Could not generate charts: {e}")

        # Performance stats
        elapsed = time.time() - t0
        avg_fps = frames_proc / elapsed if elapsed > 0 else 0.0
        result["avg_fps"] = round(float(avg_fps), 2)
        result["successful_detections"] = successful_detections
        result["total_frames"] = frames_proc
        result["detection_rate"] = round(successful_detections / max(frames_proc, 1), 2)

        print(f"Analysis complete: {frames_proc} frames processed, {successful_detections} with pose detection")
        return result

    except Exception as e:
        print(f"Critical error in analyze_video_safe: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_cricket_video(video_file, handedness, progress=gr.Progress()):
    """Process the uploaded cricket video and return analysis results"""
    
    if video_file is None:
        return None, None, None, None, "Please upload a video file first."
    
    if mp is None:
        return None, None, None, None, "MediaPipe is not installed. Please install it with: pip install mediapipe"
    
    progress(0, desc="Setting up analysis...")
    
    # Create temporary directories
    temp_dir = tempfile.mkdtemp()
    output_dir = os.path.join(temp_dir, "output")
    
    try:
        # Validate video file
        if not os.path.exists(video_file):
            return None, None, None, None, "Video file not found."
        
        # Create config
        config = create_default_config()
        
        progress(0.1, desc="Starting video analysis...")
        
        # Run the analysis
        result = analyze_video_safe(
            input_path=video_file,
            cfg=config,
            output_dir=output_dir,
            handedness=handedness
        )
        
        if result is None:
            return None, None, None, None, "Failed to analyze video. Please check if the video shows a person clearly performing cricket movements."
        
        progress(0.8, desc="Generating outputs...")
        
        # Prepare outputs
        annotated_video = os.path.join(output_dir, "annotated_video.mp4")
        evaluation_json = os.path.join(output_dir, "evaluation.json")
        metrics_csv = os.path.join(output_dir, "metrics.csv")
        
        # Create analysis summary
        summary = create_analysis_summary(result)
        
        # Load metrics for display
        metrics_df = None
        if os.path.exists(metrics_csv):
            try:
                metrics_df = pd.read_csv(metrics_csv)
                metrics_df = metrics_df.round(2)
                metrics_df = metrics_df.fillna("N/A")
            except Exception as e:
                print(f"Warning: Could not load metrics CSV: {e}")
        
        progress(1.0, desc="Analysis complete!")
        
        return (
            annotated_video if os.path.exists(annotated_video) else None,
            evaluation_json if os.path.exists(evaluation_json) else None,
            metrics_csv if os.path.exists(metrics_csv) else None,
            metrics_df,
            summary
        )
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Full error details: {error_details}")
        return None, None, None, None, f"Error processing video: {str(e)}\n\nTip: Make sure the video clearly shows a person performing cricket movements."
    
    finally:
        # Clean up temporary files after some delay
        # Comment out the cleanup for debugging
        pass

def create_analysis_summary(result):
    """Create a readable summary of the analysis results"""
    if not result:
        return "No analysis results available."
    
    detection_rate = result.get('detection_rate', 0)
    detection_status = "✅ Good" if detection_rate > 0.7 else ("⚠️ Fair" if detection_rate > 0.3 else "❌ Poor")
    
    summary = f"""
# Cricket Cover Drive Analysis Results

## Detection Quality
- **Pose Detection Rate**: {detection_rate*100:.1f}% ({result.get('successful_detections', 0)}/{result.get('total_frames', 0)} frames) {detection_status}
- **Processing Speed**: {result.get('avg_fps', 'N/A')} FPS

## Overall Performance  
- **Score**: {result.get('overall', 'N/A')}/10
- **Grade**: {result.get('grade', 'N/A').title()}

## Detailed Scores
"""
    
    scores = result.get('scores', {})
    comments = result.get('comments', {})
    
    for metric, score in scores.items():
        metric_name = metric.replace('_', ' ').title()
        comment = comments.get(metric, 'No specific feedback available.')
        
        # Add score indicator
        if score >= 8:
            indicator = "🟢"
        elif score >= 6:
            indicator = "🟡"
        else:
            indicator = "🔴"
            
        summary += f"{indicator} **{metric_name}**: {score:.1f}/10\n   *{comment}*\n\n"
    
    # Add tips based on detection rate
    if detection_rate < 0.5:
        summary += """
## 💡 Tips for Better Detection
- Ensure good lighting and clear background
- Keep the full body visible throughout the shot
- Use a stable camera position
- Film from a side-on angle for best results
"""
    
    return summary

def create_gradio_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Cricket Cover Drive Analysis", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🏏 Cricket Cover Drive Analysis
        
        Upload a cricket video to analyze your cover drive technique. The system will:
        - Track your pose and movements using AI
        - Analyze key metrics like elbow angle, spine lean, head position, and footwork  
        - Provide scores and personalized feedback on your technique
        - Generate an annotated video showing your form with real-time metrics
        
        **Supported formats**: MP4, AVI, MOV, MKV
        
        **Requirements**: The video should clearly show a person performing a cover drive from a side angle.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 📤 Upload & Settings")
                
                video_input = gr.File(
                    label="Upload Cricket Video",
                    file_types=["video"],
                    file_count="single",
                    height=200
                )
                
                handedness = gr.Radio(
                    choices=["right", "left"],
                    value="right",
                    label="Batting Hand",
                    info="Select your dominant batting hand"
                )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Video",
                    variant="primary",
                    size="lg"
                )
                
                gr.Markdown("""
                ### ⚡ Processing Info
                - Analysis typically takes 30-60 seconds
                - Longer videos will take more time
                - Progress will be shown during processing
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("## 📊 Analysis Results")
                
                analysis_summary = gr.Markdown(
                    value="Upload a video and click 'Analyze Video' to see results here.",
                    label="Summary"
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 🎥 Annotated Video")
                annotated_video = gr.Video(
                    label="Processed Video with Pose Tracking & Metrics",
                    show_download_button=True,
                    height=400
                )
            
            with gr.Column():
                gr.Markdown("## 📈 Metrics Data")
                metrics_plot = gr.Dataframe(
                    label="Frame-by-Frame Measurements",
                    interactive=False,
            
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 💾 Download Results")
                
                with gr.Row():
                    evaluation_file = gr.File(
                        label="📊 Evaluation Report (JSON)",
                        visible=False
                    )
                    
                    metrics_file = gr.File(
                        label="📈 Metrics Data (CSV)",
                        visible=False
                    )
        
        # Event handlers
        analyze_btn.click(
            fn=process_cricket_video,
            inputs=[video_input, handedness],
            outputs=[
                annotated_video,
                evaluation_file,
                metrics_file,
                metrics_plot,
                analysis_summary
            ],
            show_progress=True
        ).then(
            fn=lambda x, y: (gr.File(visible=x is not None), gr.File(visible=y is not None)),
            inputs=[evaluation_file, metrics_file],
            outputs=[evaluation_file, metrics_file]
        )
        
        # Help section
        with gr.Accordion("📋 Tips for Best Results", open=False):
            gr.Markdown("""
            ### 🎯 Video Requirements
            1. **Clear View**: Full body should be visible throughout the shot
            2. **Side Angle**: Camera positioned to the side of the batsman works best
            3. **Good Lighting**: Well-lit environment for better pose detection
            4. **Stable Camera**: Minimize camera shake and movement
            5. **Video Length**: 3-10 seconds covering the complete cover drive motion
            6. **Background**: Plain background helps with pose detection
            
            ### 📊 Metrics Explained
            - **Footwork**: Alignment and positioning of the front foot
            - **Head Position**: How well the head is positioned over the front foot
            - **Swing Control**: Smoothness and consistency of arm movement
            - **Balance**: Spine angle and overall body stability
            - **Follow Through**: Completion and control of the shot finish
            
            ### 🔧 Troubleshooting
            - **Low Detection Rate**: Try better lighting or clearer background
            - **Processing Errors**: Ensure video file is not corrupted
            - **Poor Scores**: Focus on the specific feedback provided for each metric
            """)
        
        gr.Markdown("""
        ---
        **Note**: This tool provides automated analysis for educational purposes. 
        For professional coaching, consult with qualified cricket coaches.
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Set to True to create a public link
        debug=False
    )