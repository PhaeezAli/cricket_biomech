"""
Front / back foot weight distribution via Centre of Mass (CoM) proxy.

CoM proxy = horizontal midpoint of the two hips.
Front foot is determined by handedness:
  - Right-handed batsman faces left in a side-on camera view.
    Their front foot (left foot) has a LOWER x-value.
  - Left-handed batsman: front foot (right foot) has a HIGHER x-value.
"""
import numpy as np


def compute_weight_balance(get_xy, handedness: str) -> dict:
    """
    Returns dict with:
        front_foot_weight_pct  (0–100)
        back_foot_weight_pct   (100 - front)
        front_ankle_xy
        back_ankle_xy
        stance_width
        com_xy
    """
    lhip = get_xy("left_hip")
    rhip = get_xy("right_hip")
    lankle = get_xy("left_ankle")
    rankle = get_xy("right_ankle")

    default = {
        "front_foot_weight_pct": 50.0,
        "back_foot_weight_pct": 50.0,
        "front_ankle_xy": lankle,
        "back_ankle_xy": rankle,
        "stance_width": 0.0,
        "com_xy": None,
    }

    if not (lhip and rhip and lankle and rankle):
        return default

    com_x = (lhip[0] + rhip[0]) / 2.0
    com_y = (lhip[1] + rhip[1]) / 2.0

    # Assign front/back ankle based on handedness
    # Right-handed: front foot = left ankle (lower x in side-on view)
    # Left-handed:  front foot = right ankle (higher x)
    if handedness == "right":
        front_ankle = lankle
        back_ankle = rankle
    else:
        front_ankle = rankle
        back_ankle = lankle

    front_x = front_ankle[0]
    back_x = back_ankle[0]
    stance_width = abs(front_x - back_x)

    if stance_width < 1:
        return {**default, "com_xy": (com_x, com_y)}

    # How far CoM has shifted toward front foot (0 = fully back, 100 = fully front)
    front_pct = (com_x - back_x) / (front_x - back_x) * 100.0
    front_pct = float(np.clip(front_pct, 0.0, 100.0))

    return {
        "front_foot_weight_pct": round(front_pct, 1),
        "back_foot_weight_pct": round(100.0 - front_pct, 1),
        "front_ankle_xy": front_ankle,
        "back_ankle_xy": back_ankle,
        "stance_width": stance_width,
        "com_xy": (com_x, com_y),
    }
