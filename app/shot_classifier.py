"""
Rule-based cricket shot classifier.

Uses a rolling buffer of the last N BiomechanicsResult frames to classify
the type of shot being played. Rules are based on weight distribution,
knee angles, and wrist position relative to the shoulders.
"""
from collections import deque
from dataclasses import dataclass

from app.biomechanics.metrics import BiomechanicsResult


BUFFER_SIZE = 30  # ~1 second at 30fps


@dataclass
class ShotResult:
    shot_type: str    # e.g. "Front Foot Drive"
    confidence: float  # 0.0–1.0 (fraction of buffer frames matching)


# ── Rule definitions ────────────────────────────────────────────────────────
# Each rule is (label, test_fn) where test_fn(result) → bool for a single frame.
# Rules are tested in priority order; first match wins.

def _wrists_above_shoulders(r: BiomechanicsResult) -> bool:
    """True if at least one wrist is above both shoulders."""
    wrist = r.left_wrist_xy or r.right_wrist_xy
    shoulder = r.left_shoulder_xy or r.right_shoulder_xy
    if not wrist or not shoulder:
        return False
    # In image coords, smaller y = higher on screen
    return wrist[1] < shoulder[1]


def _lateral_wrist_displacement(r: BiomechanicsResult) -> bool:
    """True if wrists are displaced laterally > 30% of stance width."""
    if not r.left_wrist_xy or not r.right_wrist_xy or r.stance_width < 1:
        return False
    wrist_mid_x = (r.left_wrist_xy[0] + r.right_wrist_xy[0]) / 2.0
    if not (r.front_ankle_xy and r.back_ankle_xy):
        return False
    feet_mid_x = (r.front_ankle_xy[0] + r.back_ankle_xy[0]) / 2.0
    displacement = abs(wrist_mid_x - feet_mid_x)
    return displacement > 0.3 * r.stance_width


def _front_knee_deep(r: BiomechanicsResult) -> bool:
    """True if the front knee is deeply bent (sweep/slog position)."""
    front_knee = r.left_knee_angle if r.handedness == "right" else r.right_knee_angle
    return front_knee < 100.0


def _knees_extended(r: BiomechanicsResult) -> bool:
    """True if both knees are mostly extended (drive position)."""
    return r.left_knee_angle > 150.0 and r.right_knee_angle > 150.0


RULES = [
    ("Sweep",           lambda r: r.front_foot_weight_pct > 60 and _front_knee_deep(r)),
    ("Pull Shot",       lambda r: r.back_foot_weight_pct > 60 and _wrists_above_shoulders(r)),
    ("Cut Shot",        lambda r: r.back_foot_weight_pct > 55 and _lateral_wrist_displacement(r)),
    ("Front Foot Drive",lambda r: r.front_foot_weight_pct > 65 and _knees_extended(r)),
    ("Back Foot Defense",lambda r: r.back_foot_weight_pct > 55 and r.spine_tilt_deg < 20),
]
DEFAULT_SHOT = "Neutral Stance"


# ── Classifier ───────────────────────────────────────────────────────────────

class ShotClassifier:
    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self._buffer: deque[BiomechanicsResult] = deque(maxlen=buffer_size)
        self._buffer_size = buffer_size

    def update(self, result: BiomechanicsResult) -> ShotResult:
        """
        Push a new frame result into the buffer and return the current shot classification.
        """
        self._buffer.append(result)

        if len(self._buffer) < self._buffer_size // 2:
            # Not enough frames yet
            return ShotResult(shot_type="Detecting...", confidence=0.0)

        return self._classify()

    def _classify(self) -> ShotResult:
        frames = list(self._buffer)
        n = len(frames)

        for label, rule in RULES:
            match_count = sum(1 for f in frames if rule(f))
            confidence = match_count / n
            if confidence >= 0.5:  # majority of frames must match
                return ShotResult(shot_type=label, confidence=round(confidence, 2))

        return ShotResult(shot_type=DEFAULT_SHOT, confidence=1.0)
