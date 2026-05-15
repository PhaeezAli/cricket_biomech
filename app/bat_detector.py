"""
Cricket bat detector.

Two modes:
  1. Geometric inference (default, no extra model needed)
     Estimates bat position by extending the elbow→wrist vector past the grip.
     Works immediately with the existing YOLO11-Pose keypoints.

  2. Custom YOLO bat model (optional, higher accuracy)
     Pass --bat-model path/to/best.pt when running main.py.

     To train your own bat model:
       a. Download a cricket bat dataset from https://universe.roboflow.com
          (search "cricket bat" — several public datasets available)
       b. yolo train model=yolo11n.pt data=<dataset>/data.yaml epochs=80 imgsz=640
       c. python app/main.py --source file --input ... --bat-model runs/detect/train/weights/best.pt
"""
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# COCO keypoint indices used here
_LW, _RW = 9,  10   # wrists
_LE, _RE = 7,  8    # elbows
_LS, _RS = 5,  6    # shoulders
_NOSE    = 0
_LA, _RA = 15, 16   # ankles

_KP_CONF        = 0.3
_TRAIL_LEN      = 45   # frames (~1.5 s at 30 fps)
_BAT_TO_HEIGHT  = 0.50 # bat length ≈ 50% of person height in pixels
_PHASE_VEL_THRESH = 3.5  # px/frame for phase transition


@dataclass
class BatDetection:
    grip_xy:             tuple         # (x, y) center of hands on handle
    bat_head_xy:         tuple         # (x, y) estimated blade tip
    angle_from_vertical: float         # 0 = vertical, 90 = horizontal
    swing_phase:         str           # "Backswing" | "Downswing" | "Set" | "Follow-through"
    bat_face_label:      str           # "Straight" | "Angled" | "Cross-bat" | "Horizontal"
    backswing_pct:       float         # 0–150 (100 = bat head at head height, >100 = above head)
    method:              str           # "geometric" | "yolo"
    confidence:          float         # 0–1
    trail: list = field(default_factory=list, repr=False)  # recent bat_head positions


class BatDetector:
    """
    Detects cricket bat position and orientation each frame.
    Maintains a swing trail and phase history internally.
    """

    def __init__(self, yolo_model_path: str | None = None):
        self._trail: deque = deque(maxlen=_TRAIL_LEN)
        self._yolo = None

        import torch
        self._device = "mps" if torch.backends.mps.is_available() else \
                       "cuda" if torch.cuda.is_available() else "cpu"

        if yolo_model_path:
            from ultralytics import YOLO
            print(f"Loading bat detection model: {yolo_model_path}  [{self._device}]")
            self._yolo = YOLO(yolo_model_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray, keypoints: np.ndarray,
               person_bbox: tuple | None = None) -> BatDetection | None:
        """
        Returns a BatDetection for the current frame, or None if bat/hands not visible.
        If a YOLO model is loaded it is tried first; falls back to geometric.
        """
        if self._yolo:
            det = self._yolo_detect(frame, keypoints, person_bbox)
            if det:
                return det
        return self._geometric_detect(keypoints, person_bbox)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _kp(self, keypoints: np.ndarray, idx: int) -> tuple | None:
        x, y, c = keypoints[idx]
        return (float(x), float(y)) if c >= _KP_CONF else None

    def _person_height_px(self, keypoints: np.ndarray,
                          person_bbox: tuple | None) -> float:
        if person_bbox:
            _, y1, _, y2 = person_bbox
            return max(abs(y2 - y1), 1.0)
        nose  = self._kp(keypoints, _NOSE)
        ankle = self._kp(keypoints, _LA) or self._kp(keypoints, _RA)
        if nose and ankle:
            return max(abs(ankle[1] - nose[1]), 1.0)
        return 400.0

    def _compute_phase(self) -> str:
        if len(self._trail) < 8:
            return "Set"
        ys  = [p[1] for p in list(self._trail)[-14:]]
        vel = [ys[i+1] - ys[i] for i in range(len(ys) - 1)]
        avg = sum(vel) / len(vel)
        if avg < -_PHASE_VEL_THRESH:
            return "Backswing"
        if avg > _PHASE_VEL_THRESH:
            return "Downswing"
        return "Set"

    @staticmethod
    def _face_label(angle: float) -> str:
        if angle < 15:   return "Straight"
        if angle < 35:   return "Angled"
        if angle < 60:   return "Cross-bat"
        return "Horizontal"

    def _backswing_pct(self, bat_head_y: float, keypoints: np.ndarray,
                       person_bbox: tuple | None) -> float:
        """
        How high is the bat head relative to the person?
        0 = at ankle level, 100 = at head level, >100 = above head.
        """
        nose  = self._kp(keypoints, _NOSE)
        ankle = self._kp(keypoints, _LA) or self._kp(keypoints, _RA)
        if nose and ankle:
            top_y    = min(nose[1], ankle[1])
            bottom_y = max(nose[1], ankle[1])
            pct = (bottom_y - bat_head_y) / max(bottom_y - top_y, 1) * 100.0
            return float(np.clip(pct, 0, 150))
        return 50.0

    # ── Geometric detection ───────────────────────────────────────────────────

    def _geometric_detect(self, keypoints: np.ndarray,
                          person_bbox: tuple | None) -> BatDetection | None:
        lw = self._kp(keypoints, _LW)
        rw = self._kp(keypoints, _RW)
        le = self._kp(keypoints, _LE)
        re = self._kp(keypoints, _RE)
        ls = self._kp(keypoints, _LS)
        rs = self._kp(keypoints, _RS)

        if not lw and not rw:
            return None

        # Grip = wrist midpoint (or single visible wrist)
        grip = ((lw[0]+rw[0])/2, (lw[1]+rw[1])/2) if (lw and rw) else (lw or rw)

        # Elbow mid → defines the handle axis direction
        if le and re:
            elbow_mid = ((le[0]+re[0])/2, (le[1]+re[1])/2)
        elif le or re:
            elbow_mid = le or re
        elif ls and rs:
            elbow_mid = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
        else:
            return None

        # Bat direction: from elbow_mid → grip, then extend past the grip
        # (blade is on the far side of the hands from the elbows)
        dx = grip[0] - elbow_mid[0]
        dy = grip[1] - elbow_mid[1]
        dist = np.hypot(dx, dy)
        if dist < 1:
            return None

        ux, uy = dx / dist, dy / dist
        bat_len = self._person_height_px(keypoints, person_bbox) * _BAT_TO_HEIGHT
        bat_head = (grip[0] + ux * bat_len, grip[1] + uy * bat_len)

        angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

        self._trail.append(bat_head)

        return BatDetection(
            grip_xy             = grip,
            bat_head_xy         = bat_head,
            angle_from_vertical = angle,
            swing_phase         = self._compute_phase(),
            bat_face_label      = self._face_label(angle),
            backswing_pct       = self._backswing_pct(bat_head[1], keypoints, person_bbox),
            method              = "geometric",
            confidence          = 0.70,
            trail               = list(self._trail),
        )

    # ── YOLO bat detection ────────────────────────────────────────────────────

    def _yolo_detect(self, frame: np.ndarray, keypoints: np.ndarray,
                     person_bbox: tuple | None) -> BatDetection | None:
        results = self._yolo.predict(
            frame, device=self._device,
            conf=0.40, verbose=False, classes=[0],
        )
        if not results or len(results[0].boxes) == 0:
            return None

        boxes = results[0].boxes
        confs  = [float(b.conf[0].cpu()) for b in boxes]
        best   = int(np.argmax(confs))
        bbox   = boxes[best].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = bbox

        # Derive grip from wrists; fall back to bbox center
        lw = self._kp(keypoints, _LW)
        rw = self._kp(keypoints, _RW)
        grip = ((lw[0]+rw[0])/2, (lw[1]+rw[1])/2) if (lw and rw) \
               else (lw or rw or ((x1+x2)/2, (y1+y2)/2))

        # Bat head = bbox corner farthest from grip
        corners = [(x1,y1),(x1,y2),(x2,y1),(x2,y2)]
        bat_head = max(corners,
                       key=lambda p: (p[0]-grip[0])**2 + (p[1]-grip[1])**2)

        dx    = bat_head[0] - grip[0]
        dy    = bat_head[1] - grip[1]
        angle = float(np.degrees(np.arctan2(abs(dx), abs(dy))))

        self._trail.append(bat_head)

        return BatDetection(
            grip_xy             = grip,
            bat_head_xy         = bat_head,
            angle_from_vertical = angle,
            swing_phase         = self._compute_phase(),
            bat_face_label      = self._face_label(angle),
            backswing_pct       = self._backswing_pct(bat_head[1], keypoints, person_bbox),
            method              = "yolo",
            confidence          = float(confs[best]),
            trail               = list(self._trail),
        )
