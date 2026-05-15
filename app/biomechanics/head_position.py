"""
Head position analysis:
  - Plumb-line: where is the head over the feet? (Front / Center / Back)
  - Head tilt: lateral tilt from level (degrees)
"""
import numpy as np
from .angles import line_angle_from_horizontal


# Offsets beyond this fraction of stance width → labelled Front or Back
CENTER_THRESHOLD = 0.15


def compute_head_position(get_xy, front_ankle_xy, back_ankle_xy, stance_width: float) -> dict:
    """
    Returns dict with:
        head_position  ("Front" | "Center" | "Back")
        head_tilt_deg  (0 = level; positive = tilted toward front foot)
        head_xy        (pixel coords of head center, or None)
    """
    default = {"head_position": "Center", "head_tilt_deg": 0.0, "head_xy": None}

    # Head center: prefer ear midpoint, fallback to nose
    lear = get_xy("left_ear")
    rear = get_xy("right_ear")
    nose = get_xy("nose")

    if lear and rear:
        head_xy = ((lear[0] + rear[0]) / 2.0, (lear[1] + rear[1]) / 2.0)
        head_tilt = line_angle_from_horizontal(lear, rear)
    elif nose:
        head_xy = nose
        head_tilt = 0.0
    else:
        return default

    if not (front_ankle_xy and back_ankle_xy) or stance_width < 1:
        return {**default, "head_xy": head_xy, "head_tilt_deg": round(head_tilt, 1)}

    feet_mid_x = (front_ankle_xy[0] + back_ankle_xy[0]) / 2.0
    offset = (head_xy[0] - feet_mid_x) / stance_width  # normalised [-∞, ∞]

    # Determine direction: positive offset toward front or back depends on handedness
    # We keep it simple: positive x offset relative to feet mid
    # The caller already set front_ankle as the lower-x (right-handed) or higher-x (left-handed)
    # so we compare to a signed threshold using the front ankle direction
    front_direction = np.sign(front_ankle_xy[0] - back_ankle_xy[0])  # +1 or -1
    signed_offset = offset * front_direction  # positive = toward front foot

    if signed_offset > CENTER_THRESHOLD:
        position = "Front"
    elif signed_offset < -CENTER_THRESHOLD:
        position = "Back"
    else:
        position = "Center"

    return {
        "head_position": position,
        "head_tilt_deg": round(abs(head_tilt), 1),
        "head_xy": head_xy,
    }
