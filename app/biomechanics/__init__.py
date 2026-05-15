"""
Biomechanics analysis sub-package.
"""
from .metrics import BiomechanicsResult
from .angles import compute_angles
from .weight_balance import compute_weight_balance
from .head_position import compute_head_position


def analyse(person_pose, handedness: str, get_xy_fn, bat_detection=None) -> BiomechanicsResult:
    """
    Full biomechanics analysis for one frame.

    Args:
        person_pose: PersonPose (from pose_estimator)
        handedness: "right" | "left"
        get_xy_fn: callable(name) → (x, y) | None

    Returns a populated BiomechanicsResult.
    """
    result = BiomechanicsResult(handedness=handedness)

    # 1. Weight balance
    wb = compute_weight_balance(get_xy_fn, handedness)
    result.front_foot_weight_pct = wb["front_foot_weight_pct"]
    result.back_foot_weight_pct = wb["back_foot_weight_pct"]
    result.front_ankle_xy = wb["front_ankle_xy"]
    result.back_ankle_xy = wb["back_ankle_xy"]
    result.stance_width = wb["stance_width"]

    # 2. Head position
    hp = compute_head_position(
        get_xy_fn,
        wb["front_ankle_xy"],
        wb["back_ankle_xy"],
        wb["stance_width"],
    )
    result.head_position = hp["head_position"]
    result.head_tilt_deg = hp["head_tilt_deg"]

    # 3. Angles
    ang = compute_angles(person_pose.keypoints, get_xy_fn)
    result.left_knee_angle = ang["left_knee_angle"]
    result.right_knee_angle = ang["right_knee_angle"]
    result.spine_tilt_deg = ang["spine_tilt_deg"]
    result.shoulder_alignment_deg = ang["shoulder_alignment_deg"]
    result.hip_alignment_deg = ang["hip_alignment_deg"]

    # 4. Store raw coords for shot classifier
    result.left_wrist_xy = get_xy_fn("left_wrist")
    result.right_wrist_xy = get_xy_fn("right_wrist")
    result.left_shoulder_xy = get_xy_fn("left_shoulder")
    result.right_shoulder_xy = get_xy_fn("right_shoulder")

    # 5. Bat metrics (optional)
    if bat_detection is not None:
        result.bat_angle_deg    = bat_detection.angle_from_vertical
        result.bat_swing_phase  = bat_detection.swing_phase
        result.bat_face_label   = bat_detection.bat_face_label
        result.bat_backswing_pct = bat_detection.backswing_pct
        result.bat_detected     = True

    return result
