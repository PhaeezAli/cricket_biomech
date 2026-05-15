"""
Angle calculations: knee bend, spine tilt, shoulder and hip alignment.
"""
import numpy as np


def joint_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    Angle (degrees) at point b, formed by vectors b→a and b→c.
    Returns 0.0 if any coordinate is invalid.
    """
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def line_angle_from_horizontal(p1: tuple, p2: tuple) -> float:
    """
    Angle (degrees) of the line p1→p2 from the horizontal axis.
    Positive = p2 is higher than p1 on screen (y decreases upward in image coords).
    Returns value in [-90, 90].
    """
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]  # image y: down = positive
    if dx == 0 and dy == 0:
        return 0.0
    return float(np.degrees(np.arctan2(-dy, dx)))  # negate dy for screen → math coords


def line_angle_from_vertical(p_bottom: tuple, p_top: tuple) -> float:
    """
    Angle (degrees) of the line from bottom to top point, measured from the vertical axis.
    0° = perfectly vertical; positive = leaning toward p2's x direction.
    """
    dx = p_top[0] - p_bottom[0]
    dy = p_bottom[1] - p_top[1]  # convert image coords: up = positive
    if dx == 0 and dy == 0:
        return 0.0
    return float(np.degrees(np.arctan2(dx, dy)))


def compute_angles(kp, get_xy) -> dict:
    """
    Compute all angle-based metrics from keypoints.

    Args:
        kp: numpy array (17, 3) of keypoints
        get_xy: callable(name) → (x, y) | None

    Returns dict with keys:
        left_knee_angle, right_knee_angle,
        spine_tilt_deg, shoulder_alignment_deg, hip_alignment_deg
    """
    result = {
        "left_knee_angle": 180.0,
        "right_knee_angle": 180.0,
        "spine_tilt_deg": 0.0,
        "shoulder_alignment_deg": 0.0,
        "hip_alignment_deg": 0.0,
    }

    # Knee angles
    lh, lk, la = get_xy("left_hip"), get_xy("left_knee"), get_xy("left_ankle")
    if lh and lk and la:
        result["left_knee_angle"] = joint_angle(lh, lk, la)

    rh, rk, ra = get_xy("right_hip"), get_xy("right_knee"), get_xy("right_ankle")
    if rh and rk and ra:
        result["right_knee_angle"] = joint_angle(rh, rk, ra)

    # Spine tilt: hip midpoint → shoulder midpoint, angle from vertical
    ls, rs = get_xy("left_shoulder"), get_xy("right_shoulder")
    lhip, rhip = get_xy("left_hip"), get_xy("right_hip")

    if ls and rs and lhip and rhip:
        shoulder_mid = ((ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2)
        hip_mid = ((lhip[0] + rhip[0]) / 2, (lhip[1] + rhip[1]) / 2)
        result["spine_tilt_deg"] = abs(line_angle_from_vertical(hip_mid, shoulder_mid))

        # Shoulder alignment: levelness of shoulder line
        result["shoulder_alignment_deg"] = abs(line_angle_from_horizontal(ls, rs))

        # Hip alignment: levelness of hip line
        result["hip_alignment_deg"] = abs(line_angle_from_horizontal(lhip, rhip))

    return result
