# Real-Time Cover Drive Analysis

This repo processes a *full* cricket video in real-time, runs pose estimation on each frame, overlays live biomechanical metrics with feedback, and outputs an annotated video plus a final evaluation JSON.

## Features
- Full video pipeline (OpenCV): read → analyze → annotate → write
- Pose estimation: MediaPipe Pose (lightweight, CPU-friendly)
- Metrics (per-frame + rolling):
  - Front elbow angle (shoulder–elbow–wrist)
  - Spine lean (shoulder–hip vs vertical)
  - Head-over-knee x-alignment
  - Front-foot direction vs crease (x-axis)
- Live overlays with real-time metric readouts and ✅/❌ cues
- Final evaluation with 5 categories + actionable comments
- Robustness: missing joints handled w/ smoothing + interpolation
- phase segmentation, contact detection, smoothness charts

## Quickstart
```bash
# Create env
python -m venv .venv && source .venv/bin/activate  # (Windows) .venv\\Scripts\\activate

pip install -r requirements.txt

# Edit config.yaml
# - set video.input to a local file 
# - tweak thresholds if needed

python cover_drive_analysis_realtime.py --config config.yaml

Copy
python cover_drive_analysis_realtime.py --config config.yaml \
  --input path_or_youtube_url \
  --output_dir ./output \
  --resize 960x540 \
  --fps 25
Outputs
output/annotated_video.mp4 – full-length video with overlays

output/evaluation.json – final scores + comments

output/metrics.csv – data

output/elbow_angle.png, output/spine_lean.png – charts

Assumptions & Notes
Right-handed batter by default; swap left/right landmarks for left-handed.

“Crease/x-axis” is approximated by the video x-axis.

Angles are computed in the 2D image plane; 

Heuristics are tuned for the sample video; adjust in config.yaml.

Limitations
Camera perspective and motion blur affect accuracy.

Bat detection is not included; optional HSV/YOLO-lite can be added.

Impact detection is heuristic (wrist velocity spike).
