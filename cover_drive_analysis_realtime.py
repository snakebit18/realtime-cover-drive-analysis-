
import os, cv2, math, json, time, yaml, csv
import numpy as np
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional

import mediapipe as mp
from tqdm import tqdm
import matplotlib.pyplot as plt


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def angle_abc(a, b, c):
    if a is None or b is None or c is None: return None
    ab, cb = a-b, c-b
    nab, ncb = np.linalg.norm(ab), np.linalg.norm(cb)
    if nab < 1e-6 or ncb < 1e-6: return None
    cosang = np.dot(ab, cb) / (nab*ncb)
    cosang = np.clip(cosang, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def line_to_vertical_angle(pt_top, pt_bottom):
    """Angle between the (pt_top -> pt_bottom) line and image vertical (0 deg is perfectly vertical)."""
    if pt_top is None or pt_bottom is None: return None
    dx, dy = pt_bottom[0]-pt_top[0], pt_bottom[1]-pt_top[1]
    if abs(dy) < 1e-6 and abs(dx) < 1e-6: return None
    angle_line = math.degrees(math.atan2(dy, dx))  # 0 right, 90 down
    return float(abs(90.0 - angle_line))

def foot_direction(ankle, toe):
    """Angle of foot vector (ankle->toe) vs image x-axis (crease surrogate)."""
    if ankle is None or toe is None: return None
    v = toe - ankle
    if np.linalg.norm(v) < 1e-6: return None
    ang = math.degrees(math.atan2(v[1], v[0])) # vs x-axis
    return float(abs(180.0 - ang)) # yaw from x-axis; abs for directionless tolerance

def projected_x_dist(p1, p2):
    """Absolute x distance in pixels between two points (e.g., head and feet_index)."""
    if p1 is None or p2 is None: return None
    return float(abs(p1[0]-p2[0]))

def moving_average(arr: List[float], win: int) -> List[float]:
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
        """
        Convert a PoseLandmark to (x, y, z) image coordinates.
        """
        return np.array([lm.x*w, lm.y*h, lm.z*w], dtype=np.float32)

    def extract(self, frame_bgr) -> Dict[str, Optional[np.ndarray]]:
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
        except Exception:
            pass
        return J
    

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
        # Choose "front" side based on handedness (right-handed => left foot/left elbow is front)
        if handedness == "right":
            shoulder = J["l_shoulder"]; elbow = J["l_elbow"]; wrist = J["l_wrist"]
            hip_top = J["l_shoulder"]; hip_bottom = J["l_hip"]
            head = J["head"]
            ankle = J["l_ankle"]; toe = J["l_foot_index"]
        else:
            shoulder = J["r_shoulder"]; elbow = J["r_elbow"]; wrist = J["r_wrist"]
            hip_top = J["r_shoulder"]; hip_bottom = J["r_hip"]
            head = J["head"]
            ankle = J["r_ankle"]; toe = J["r_foot_index"]

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

    def log(self, t: float, metrics: Dict[str, Optional[float]]):
        for k in ["elbow_angle","spine_lean","head_toe_x","foot_dir"]:
            self.logs[k].append(metrics.get(k, None))
        self.logs["time"].append(t)

    def smoothed(self) -> Dict[str, List[Optional[float]]]:
        """Smooth the logs by:
        1. Replacing None with the previous valid value
        2. Computing a moving average over the smoothed values
        """
        out = {}
        for k, arr in self.logs.items():
            if k == "time":
                # leave timestamps as is
                out[k] = arr
                continue
            # Replace None with previous valid value
            clean = []
            last = None
            for v in arr:
                if v is None:
                    # if a value is missing, use the last valid one
                    clean.append(last if last is not None else None)
                else:
                    clean.append(v); last = v

            # Compute a moving average over the smoothed values
            nums = [x for x in clean if x is not None]
            if not nums:
                # if no valid values, leave as is
                out[k] = clean
                continue
            # forward fill Nones then MA
            ff = []
            last = nums[0]
            for v in clean:
                if v is None:
                    # fill in the gaps with the last valid value
                    ff.append(last)
                else:
                    last = v; ff.append(v)
            out[k] = moving_average(ff, self.win)
        return out
    
GREEN = (40, 200, 40)
RED = (40, 40, 220)
WHITE = (245, 245, 245)
YELL = (30, 220, 220)

def draw_pose(frame, J: Dict[str, Optional[np.ndarray]]):
    pairs = [
    ("l_shoulder","l_elbow"), ("l_elbow","l_wrist"),
    ("r_shoulder","r_elbow"), ("r_elbow","r_wrist"),
    ("l_shoulder","l_hip"), ("r_shoulder","r_hip"),
    ("l_hip","l_knee"), ("l_knee","l_ankle"),
    ("r_hip","r_knee"), ("r_knee","r_ankle"),
    ("l_ankle","l_foot_index"), ("r_ankle","r_foot_index"),
    ("l_shoulder","r_shoulder"), ("l_hip","r_hip")
    ]
    for a,b in pairs:
        pa, pb = J[a][:2], J[b][:2]
        if pa is not None and pb is not None:
            cv2.line(frame, tuple(pa.astype(int)), tuple(pb.astype(int)), WHITE, 2)
    for k, pt in J.items():
        if pt is not None:
            pt= pt[:2] 
            cv2.circle(frame, tuple(pt.astype(int)), 3, YELL, -1)
    return frame

def put_metric_text(frame, x, y, label, value, unit="", good: Optional[bool]=None):
    text = f"{label}: {value if value is not None else '—'}{unit}"
    color = GREEN if good is True else (RED if good is False else WHITE)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def cue(frame, x, y, ok: bool, msg_ok="Good", msg_bad="Check"):
    msg = f" {msg_ok}" if ok else f" {msg_bad}"
    color = GREEN if ok else RED
    cv2.putText(frame, msg, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)




def evaluate(cfg: dict, tracks: Dict[str, List[Optional[float]]]) -> Dict[str, object]:
    th = cfg["metrics"]

    def score_footwork(foot_dir):
        valid = [x for x in foot_dir if x is not None]
        if not valid: return 5.0, "Insufficient foot detection."
        dev = np.median([abs(x) for x in valid])
        s = np.interp(th["foot_dir_max"] - dev, [0, th["foot_dir_max"]], [4, 10])
        tip = "Keep front foot aligned within ~{}°".format(th["foot_dir_max"])
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
        # variance penalizes jerkiness
        var = np.var(valid)
        s = np.interp(30 - min(var, 30), [0, 30], [4, 10])  # 30 is max variance
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
        s = 4 + 6*frac  # convert fraction to score 1-10
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



def segment_phases(times: List[float], wrist_xy: List[Tuple[Optional[float],Optional[float]]]) -> Dict[str, List[int]]:
    N = len(times)
    downswing_start = 0
    impact = None
    # crude: find max finite-difference speed in wrist x,y as "impact"
    vx = []
    for i in range(1,N):
        x0,y0 = wrist_xy[i-1]
        x1,y1 = wrist_xy[i]
        if None in (x0,y0,x1,y1):
            vx.append(0.0); continue
        dt = max(1e-3, times[i]-times[i-1])
        v = math.hypot(x1-x0, y1-y0)/dt
        vx.append(v)


    if vx:
        impact = 1 + int(np.argmax(vx))
    # downswing start: first frame where elbow angle begins increasing steadily before impact
    if impact is not None:
        k = max(0, impact-8)
        downswing_start = max(0, k)
        return {"stance":[0, max(0, downswing_start-1)],
        "downswing":[downswing_start, (impact or N//2)],
        "follow_through":[(impact or N//2)+1, N-1]}


class MetricLogger:
    def __init__(self):
        self.rows = []

    def log(self, t, metrics):
        row = {"time": t}
        row.update(metrics)
        self.rows.append(row)

    def to_csv(self, path):
        if not self.rows: return
        keys = self.rows[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(r)

    def to_charts(self, out_dir):
        if plt is None or not self.rows: return
        times = [r["time"] for r in self.rows]
        for key in ["elbow_angle", "spine_lean"]:
            vals = [r[key] for r in self.rows]
            plt.figure()
            plt.plot(times, vals, label=key)
            plt.xlabel("Time (s)")
            plt.ylabel(key)
            if key == "spine_lean":
                plt.axhline(35, color="k", linestyle="--")
            else:
                plt.axhline(80, color="k", linestyle="--")
                plt.axhline(130, color="k", linestyle="--")
            plt.title(key + " over time")
            plt.legend()
            plt.savefig(os.path.join(out_dir, f"{key}.png"))
            plt.close()


def analyze_video(input_path: str, cfg_path: str = None, output_dir: str = "./output",

handedness: str = "right") -> Dict[str, object]:

    cfg = load_yaml(cfg_path) if cfg_path else {}
    
    ensure_dir(output_dir)

    # Video IO
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video.")


    W = int(cfg["video"].get("width", 960))
    H = int(cfg["video"].get("height", 540))
    TGT_FPS = int(cfg["video"].get("target_fps", 25))
    fourcc = cv2.VideoWriter_fourcc(*cfg["video"].get("fourcc", "mp4v"))
    out_path = os.path.join(output_dir, "annotated_video.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, TGT_FPS, (W, H))

    pose = PoseEstimator(cfg)
    tracker = MetricTracker(cfg)
    logger=MetricLogger()

    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or TGT_FPS
    frame_interval = 1.0 / max(src_fps, 1e-6)  # time(sec) for each frame

    t0 = time.time()
    frames_proc = 0

    wrist_trace = []  # for phase heuristic

    for i in tqdm(range(nframes or 100000)):
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        J = pose.extract(frame)

        # metrics
        m = tracker.compute(J, handedness=handedness)
    
    
        tracker.log(frames_proc * frame_interval, m)    # frame no * interval -> time in seconds
        logger.log(frames_proc * frame_interval, m)
        # overlays
        draw_pose(frame, J)
       
        # Good/bad checks
        th = tracker.th
        elbow = m["elbow_angle"]; spine = m["spine_lean"]
        hk = m["head_toe_x"]; fd = m["foot_dir"]

        good_elbow = (elbow is not None) and (th["elbow_good_min"] <= elbow <= th["elbow_good_max"])
        good_spine = (spine is not None) and (abs(spine) <= th["spine_lean_max"])
        good_head = (hk is not None) and (hk <= th["head_over_toe_max_x_px"])
        good_foot = (fd is not None) and (abs(fd) <= th["foot_dir_max"])

        put_metric_text(frame, 15, 30, "Elbow", None if elbow is None else round(elbow,1), "deg", good_elbow)
        put_metric_text(frame, 15, 60, "Spine", None if spine is None else round(spine,1), "deg", good_spine)
        put_metric_text(frame, 15, 90, "Head-Toe X", None if hk is None else int(hk), "px", good_head)
        put_metric_text(frame, 15, 120, "Foot Dir", None if fd is None else round(fd,1), "deg", good_foot)

        # cues
        cue(frame, 15, 200, good_elbow, "Good elbow elevation", "Raise front elbow")
        cue(frame, 15, 230, good_head, "Head over front foot", "Bring head over front foot")
        cue(frame, 15, 250, good_spine, "Balanced spine", "Reduce side lean")

        # collect wrist trace (use front wrist)
        wrist = J["l_wrist"] if handedness=="right" else J["r_wrist"]
        wrist_trace.append((None if wrist is None else float(wrist[0]),
                            None if wrist is None else float(wrist[1])))

        writer.write(frame)
        frames_proc += 1

    cap.release(); writer.release()

    # Smooth tracks and evaluate
    tracks = tracker.smoothed()
    result = evaluate(cfg, tracks)

    # Phases (optional heuristic)
    phases = segment_phases(tracks["time"], wrist_trace)
    result["phases"] = phases

    # Save evaluation.json
    eval_path = os.path.join(output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)

    # Save metrics CSV and charts
    csv_path = os.path.join(output_dir, "metrics.csv")
    logger.to_csv(csv_path)
    logger.to_charts(output_dir)

    # FPS log
    elapsed = time.time() - t0
    avg_fps = frames_proc / elapsed if elapsed > 0 else 0.0
    result["avg_fps"] = round(float(avg_fps), 2)
    print(f"[INFO] Processed {frames_proc} frames in {elapsed:.2f}s ({avg_fps:.2f} FPS).")
    print(f"[OK] Wrote: {out_path}")
    print(f"[OK] Wrote: {eval_path}")
    print(f"[OK] Wrote: {csv_path}")
    print(f"[OK] Wrote charts in: {output_dir}")
    return result

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Cover Drive Analysis")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--input", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./output")
    ap.add_argument("--handedness", type=str, default="right")
    args = ap.parse_args()
    analyze_video(args.input, cfg_path=args.config, output_dir=args.output_dir,
                  handedness=args.handedness)

if __name__=="__main__":
    main()
