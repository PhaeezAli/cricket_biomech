"""
BiomechanicsResult dataclass — single container for all per-frame metrics.
"""
from dataclasses import dataclass, field


@dataclass
class BiomechanicsResult:
    # Weight distribution
    front_foot_weight_pct: float = 50.0   # 0–100
    back_foot_weight_pct: float = 50.0    # 100 - front

    # Head position
    head_position: str = "Center"          # "Front" | "Center" | "Back"
    head_tilt_deg: float = 0.0             # lateral tilt; 0 = level

    # Knee flexion (180 = fully straight)
    left_knee_angle: float = 180.0
    right_knee_angle: float = 180.0

    # Posture
    spine_tilt_deg: float = 0.0            # forward lean from vertical
    shoulder_alignment_deg: float = 0.0   # 0 = perfectly level
    hip_alignment_deg: float = 0.0

    # Session metadata
    handedness: str = "right"              # "right" | "left"

    # Raw wrist positions for shot classifier (pixel coords, may be None)
    left_wrist_xy: tuple | None = field(default=None, repr=False)
    right_wrist_xy: tuple | None = field(default=None, repr=False)
    left_shoulder_xy: tuple | None = field(default=None, repr=False)
    right_shoulder_xy: tuple | None = field(default=None, repr=False)
    front_ankle_xy: tuple | None = field(default=None, repr=False)
    back_ankle_xy: tuple | None = field(default=None, repr=False)
    stance_width: float = 0.0

    # Bat metrics (populated when bat detector runs)
    bat_angle_deg: float = 0.0             # 0 = vertical, 90 = horizontal
    bat_swing_phase: str = "Set"           # Backswing | Downswing | Set | Follow-through
    bat_face_label: str = ""               # Straight | Angled | Cross-bat | Horizontal
    bat_backswing_pct: float = 0.0         # 0–150 (100 = head at head height)
    bat_detected: bool = False
