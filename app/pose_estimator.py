"""
YOLO11-Pose wrapper — single model for person detection + pose estimation.
Replaces the old separate YOLO detector + MediaPipe pipeline.
"""
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO


# COCO 17 keypoint indices
KP = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

# COCO skeleton connections for drawing
SKELETON = [
    (KP["nose"], KP["left_ear"]),
    (KP["nose"], KP["right_ear"]),
    (KP["left_ear"], KP["left_shoulder"]),
    (KP["right_ear"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["right_shoulder"]),
    (KP["left_shoulder"], KP["left_elbow"]),
    (KP["left_elbow"], KP["left_wrist"]),
    (KP["right_shoulder"], KP["right_elbow"]),
    (KP["right_elbow"], KP["right_wrist"]),
    (KP["left_shoulder"], KP["left_hip"]),
    (KP["right_shoulder"], KP["right_hip"]),
    (KP["left_hip"], KP["right_hip"]),
    (KP["left_hip"], KP["left_knee"]),
    (KP["left_knee"], KP["left_ankle"]),
    (KP["right_hip"], KP["right_knee"]),
    (KP["right_knee"], KP["right_ankle"]),
]


@dataclass
class PersonPose:
    """Detected person with pose keypoints."""
    bbox: tuple          # (x1, y1, x2, y2) in pixel coords
    keypoints: np.ndarray  # shape (17, 3): (x, y, confidence)
    bbox_area: float


class PoseEstimator:
    """
    Wraps YOLO11-Pose to detect the largest person and return 17 COCO keypoints.
    Uses Apple Silicon MPS backend when available.
    """

    CONF_THRESHOLD = 0.5    # minimum detection confidence
    KP_CONF_THRESHOLD = 0.3  # below this, keypoint is considered invisible

    def __init__(self, model_name: str = "yolo11n-pose.pt"):
        import torch
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Loading YOLO11-Pose on device: {self.device}")
        self.model = YOLO(model_name)
        self.KP = KP
        self.SKELETON = SKELETON

    def estimate(self, frame: np.ndarray) -> PersonPose | None:
        """
        Run pose estimation on a frame.
        Returns the PersonPose for the largest detected person, or None if no person found.
        """
        results = self.model.predict(
            frame,
            device=self.device,
            conf=self.CONF_THRESHOLD,
            verbose=False,
            classes=[0],  # person only
        )

        if not results or results[0].keypoints is None:
            return None

        boxes = results[0].boxes
        kps = results[0].keypoints

        if len(boxes) == 0:
            return None

        # Pick the largest bounding box (assumed to be the main batsman)
        areas = [(box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                 for box in boxes]
        areas_cpu = [float(a.cpu()) if hasattr(a, 'cpu') else float(a) for a in areas]
        best_idx = int(np.argmax(areas_cpu))

        box = boxes[best_idx].xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        area = areas_cpu[best_idx]

        # keypoints: shape (17, 3) — x, y, conf in pixel coords
        kp_data = kps[best_idx].data[0].cpu().numpy()  # (17, 3)

        return PersonPose(
            bbox=(x1, y1, x2, y2),
            keypoints=kp_data,
            bbox_area=area,
        )

    def get_xy(self, keypoints: np.ndarray, name: str) -> tuple[float, float] | None:
        """
        Return (x, y) for a named keypoint if confidence is above threshold.
        Returns None if the keypoint is not visible.
        """
        idx = self.KP[name]
        x, y, conf = keypoints[idx]
        if conf < self.KP_CONF_THRESHOLD:
            return None
        return float(x), float(y)
