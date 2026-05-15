"""
Frame annotation — premium sports-broadcast-style overlay.

Visual style:
  - Neon skeleton with glow effect and gradient limbs
  - Frosted glass panels with accent borders
  - Gradient weight gauge with animated-style needle
  - Pill-shaped insight badges
  - Subtle vignette + corner branding
"""
import cv2
import numpy as np
import math

from app.bat_detector import BatDetection
from app.biomechanics.metrics import BiomechanicsResult
from app.insights import Insight
from app.shot_classifier import ShotResult


# ── Palette ───────────────────────────────────────────────────────────────────
# Modern sports broadcast colour scheme
C_ACCENT   = (250, 180, 40)    # warm gold accent
C_ACCENT2  = (255, 100, 60)    # coral secondary accent
C_GREEN    = (80, 230, 100)
C_YELLOW   = (50, 220, 255)
C_RED      = (80, 70, 240)
C_CYAN     = (230, 210, 60)
C_WHITE    = (245, 245, 245)
C_DIM      = (130, 130, 140)
C_BG       = (15, 15, 20)
C_PANEL    = (22, 22, 30)
C_PANEL_LT = (32, 32, 42)
C_BORDER   = (55, 55, 65)

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD  = cv2.FONT_HERSHEY_DUPLEX


# ── Drawing primitives ───────────────────────────────────────────────────────

def _alpha_rect(img, x1, y1, x2, y2, colour, alpha: float):
    """Semi-transparent filled rectangle."""
    y1, y2 = max(0, y1), min(img.shape[0], y2)
    x1, x2 = max(0, x1), min(img.shape[1], x2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    solid = np.full_like(roi, colour)
    cv2.addWeighted(solid, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _gradient_rect_h(img, x1, y1, x2, y2, col_left, col_right, alpha: float):
    """Horizontal gradient rectangle."""
    y1, y2 = max(0, y1), min(img.shape[0], y2)
    x1, x2 = max(0, x1), min(img.shape[1], x2)
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return
    w = x2 - x1
    grad = np.zeros_like(roi)
    for i in range(3):
        grad[:, :, i] = np.linspace(col_left[i], col_right[i], w, dtype=np.uint8)
    cv2.addWeighted(grad, alpha, roi, 1 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def _glow_circle(img, center, radius, colour, glow_radius=None, intensity=0.35):
    """Circle with soft outer glow."""
    if glow_radius is None:
        glow_radius = radius * 3
    overlay = img.copy()
    for r in range(int(glow_radius), radius, -1):
        a = intensity * (1 - (r - radius) / (glow_radius - radius)) ** 2
        c = tuple(int(v * a) for v in colour)
        cv2.circle(overlay, center, r, c, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
    cv2.circle(img, center, radius, colour, -1, cv2.LINE_AA)
    cv2.circle(img, center, radius + 1, (255, 255, 255), 1, cv2.LINE_AA)


def _glow_line(img, p1, p2, colour, thickness=2, glow_thickness=None):
    """Line with soft glow."""
    if glow_thickness is None:
        glow_thickness = thickness + 6
    overlay = img.copy()
    # outer glow layers
    for t in range(glow_thickness, thickness, -1):
        a = 0.08
        glow_col = tuple(int(v * 0.4) for v in colour)
        cv2.line(overlay, p1, p2, glow_col, t, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
    cv2.line(img, p1, p2, colour, thickness, cv2.LINE_AA)


def _lerp_colour(c1, c2, t: float):
    """Linearly interpolate between two BGR colours."""
    t = max(0.0, min(1.0, t))
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def _status_colour(value: float, good_lo: float, good_hi: float) -> tuple:
    margin = (good_hi - good_lo) * 0.25
    if good_lo <= value <= good_hi:
        return C_GREEN
    elif (good_lo - margin) <= value <= (good_hi + margin):
        return C_YELLOW
    return C_RED


def _severity_colour(severity: str) -> tuple:
    return {"good": C_GREEN, "warn": C_YELLOW, "bad": C_RED}.get(severity, C_WHITE)


# ── Vignette ──────────────────────────────────────────────────────────────────

def _apply_vignette(img, strength: float = 0.35):
    """Subtle darkening at edges for cinematic look."""
    h, w = img.shape[:2]
    X = cv2.getGaussianKernel(w, w * 0.6)
    Y = cv2.getGaussianKernel(h, h * 0.6)
    mask = Y @ X.T
    mask = mask / mask.max()
    mask = (1.0 - strength) + strength * mask
    for i in range(3):
        img[:, :, i] = np.clip(img[:, :, i] * mask, 0, 255).astype(np.uint8)


# ── Skeleton (neon glow) ─────────────────────────────────────────────────────

_LIMB_GROUPS = {
    "torso":  [(5,6),(5,11),(6,12),(11,12)],
    "arms":   [(5,7),(7,9),(6,8),(8,10)],
    "legs":   [(11,13),(13,15),(12,14),(14,16)],
    "head":   [(0,3),(0,4),(3,5),(4,6)],
}
_GROUP_NEON = {
    "torso": (200, 220, 60),     # warm teal-gold
    "arms":  (60, 200, 240),     # cyan
    "legs":  (80, 240, 140),     # neon green
    "head":  (240, 180, 60),     # amber
}
_JOINT_COLOUR = (255, 255, 255)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray, kp_conf: float = 0.3):
    # Draw glow limbs
    for group, pairs in _LIMB_GROUPS.items():
        colour = _GROUP_NEON[group]
        for i, j in pairs:
            xi, yi, ci = keypoints[i]
            xj, yj, cj = keypoints[j]
            if ci >= kp_conf and cj >= kp_conf:
                p1 = (int(xi), int(yi))
                p2 = (int(xj), int(yj))
                _glow_line(frame, p1, p2, colour, thickness=2, glow_thickness=8)

    # Draw keypoint joints with glow
    for x, y, conf in keypoints:
        if conf >= kp_conf:
            pt = (int(x), int(y))
            _glow_circle(frame, pt, 4, _JOINT_COLOUR, glow_radius=12, intensity=0.3)


# ── Top bar ───────────────────────────────────────────────────────────────────

TOP_BAR_H = 50


def draw_top_bar(frame: np.ndarray, shot: ShotResult, fps: float,
                 handedness: str, h: int, w: int):
    # Background with subtle gradient
    _gradient_rect_h(frame, 0, 0, w, TOP_BAR_H, (12, 12, 18), (20, 20, 30), 0.88)
    # Bottom accent line
    _gradient_rect_h(frame, 0, TOP_BAR_H - 2, w, TOP_BAR_H, C_ACCENT, C_ACCENT2, 0.7)

    # Handedness badge (left) — pill shape
    hand_text = f"{'RIGHT' if handedness == 'right' else 'LEFT'}-HAND"
    (tw, th), _ = cv2.getTextSize(hand_text, FONT, 0.42, 1)
    bx1, by1 = 12, 12
    bx2, by2 = bx1 + tw + 16, by1 + th + 12
    _alpha_rect(frame, bx1, by1, bx2, by2, (40, 40, 50), 0.7)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), C_BORDER, 1)
    cv2.putText(frame, hand_text, (bx1 + 8, by2 - 6),
                FONT, 0.42, C_DIM, 1, cv2.LINE_AA)

    # Shot type (center) — prominent pill
    is_active = shot.shot_type not in ("Detecting...", "Neutral Stance")
    shot_text = shot.shot_type.upper()
    conf_text = f"  {shot.confidence*100:.0f}%" if is_active else ""
    full_text = f" {shot_text}{conf_text} "
    shot_colour = C_ACCENT if is_active else C_DIM

    (tw, th), _ = cv2.getTextSize(full_text, FONT_BOLD, 0.60, 1)
    sx1 = (w - tw - 20) // 2
    sy1 = 8
    sx2 = sx1 + tw + 20
    sy2 = sy1 + th + 16

    if is_active:
        _alpha_rect(frame, sx1, sy1, sx2, sy2, (40, 35, 15), 0.8)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), C_ACCENT, 1)
    else:
        _alpha_rect(frame, sx1, sy1, sx2, sy2, C_PANEL, 0.7)
        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), C_BORDER, 1)

    cv2.putText(frame, full_text, (sx1 + 10, sy2 - 8),
                FONT_BOLD, 0.60, shot_colour, 1, cv2.LINE_AA)

    # FPS (right) — with colour dot
    fps_col = C_GREEN if fps >= 20 else C_YELLOW if fps >= 12 else C_RED
    fps_text = f"{fps:.0f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_text, FONT, 0.46, 1)
    fx = w - tw - 30
    cv2.circle(frame, (fx - 8, 28), 4, fps_col, -1, cv2.LINE_AA)
    cv2.putText(frame, fps_text, (fx, 32), FONT, 0.46, fps_col, 1, cv2.LINE_AA)


# ── Right metrics panel ───────────────────────────────────────────────────────

_METRICS = [
    ("Head",       lambda b: b.head_position,
                   lambda b: C_GREEN if b.head_position == "Center" else C_YELLOW if b.head_position == "Front" else C_RED),
    ("Head Tilt",  lambda b: f"{b.head_tilt_deg:.1f}",
                   lambda b: _status_colour(b.head_tilt_deg, 0, 8)),
    ("L Knee",     lambda b: f"{b.left_knee_angle:.0f}",
                   lambda b: _status_colour(b.left_knee_angle, 115, 165)),
    ("R Knee",     lambda b: f"{b.right_knee_angle:.0f}",
                   lambda b: _status_colour(b.right_knee_angle, 115, 165)),
    ("Spine",      lambda b: f"{b.spine_tilt_deg:.1f}",
                   lambda b: _status_colour(b.spine_tilt_deg, 0, 20)),
    ("Shoulders",  lambda b: f"{b.shoulder_alignment_deg:.1f}",
                   lambda b: _status_colour(b.shoulder_alignment_deg, 0, 8)),
    ("Hips",       lambda b: f"{b.hip_alignment_deg:.1f}",
                   lambda b: _status_colour(b.hip_alignment_deg, 0, 8)),
]

_BAT_METRICS = [
    ("Bat Face",   lambda b: b.bat_face_label if b.bat_face_label else "--",
                   lambda b: C_GREEN if b.bat_face_label == "Straight" else
                             C_YELLOW if b.bat_face_label == "Angled" else
                             C_RED if b.bat_face_label else C_DIM),
    ("Bat Angle",  lambda b: f"{b.bat_angle_deg:.0f}",
                   lambda b: _status_colour(b.bat_angle_deg, 0, 30)),
    ("Swing",      lambda b: b.bat_swing_phase,
                   lambda b: C_GREEN if b.bat_swing_phase == "Set" else C_YELLOW),
    ("Backswing",  lambda b: f"{b.bat_backswing_pct:.0f}%",
                   lambda b: C_GREEN if 60 <= b.bat_backswing_pct <= 110
                             else C_YELLOW if b.bat_backswing_pct > 40 else C_RED),
]

PANEL_W    = 200
CARD_H     = 32
CARD_GAP   = 2
PANEL_PAD  = 10


def _draw_mini_bar(img, x, y, w, h, value: float, max_val: float,
                   colour: tuple, bg_colour=(40, 40, 50)):
    """Tiny horizontal progress bar."""
    _alpha_rect(img, x, y, x + w, y + h, bg_colour, 0.6)
    fill_w = max(1, int(w * min(value / max_val, 1.0)))
    _alpha_rect(img, x, y, x + fill_w, y + h, colour, 0.8)


def draw_metrics_panel(frame: np.ndarray, bio: BiomechanicsResult, h: int, w: int):
    top_offset = TOP_BAR_H + 8
    n_body = len(_METRICS)
    n_bat  = len(_BAT_METRICS) + 1 if bio.bat_detected else 0
    total_rows = n_body + n_bat
    panel_h = total_rows * (CARD_H + CARD_GAP) + PANEL_PAD * 2 + 20  # +20 for header

    px1 = w - PANEL_W - 8
    py1 = top_offset
    px2 = w - 8
    py2 = py1 + panel_h

    # Frosted glass panel
    _alpha_rect(frame, px1, py1, px2, py2, C_PANEL, 0.82)
    # Left accent stripe
    _alpha_rect(frame, px1, py1, px1 + 3, py2, C_ACCENT, 0.6)
    cv2.rectangle(frame, (px1, py1), (px2, py2), C_BORDER, 1)

    # Header
    cv2.putText(frame, "BIOMECHANICS", (px1 + 12, py1 + 18),
                FONT_BOLD, 0.44, C_ACCENT, 1, cv2.LINE_AA)
    header_y = py1 + 24

    def _draw_card(i, label, value, colour, start_y):
        cy = start_y + i * (CARD_H + CARD_GAP)

        # Card background (alternating subtle shade)
        if i % 2 == 0:
            _alpha_rect(frame, px1 + 4, cy, px2 - 4, cy + CARD_H, C_PANEL_LT, 0.4)

        # Status bar on left edge
        _alpha_rect(frame, px1 + 4, cy + 4, px1 + 7, cy + CARD_H - 4, colour, 0.9)

        # Label
        cv2.putText(frame, label, (px1 + 14, cy + CARD_H // 2 + 5),
                    FONT, 0.40, C_DIM, 1, cv2.LINE_AA)

        # Value (right-aligned, bold, coloured)
        val_text = str(value)
        (vw, _), _ = cv2.getTextSize(val_text, FONT_BOLD, 0.46, 1)
        cv2.putText(frame, val_text, (px2 - vw - 12, cy + CARD_H // 2 + 5),
                    FONT_BOLD, 0.46, colour, 1, cv2.LINE_AA)

    # Body metrics
    body_start = header_y + 4
    for i, (label, val_fn, col_fn) in enumerate(_METRICS):
        _draw_card(i, label, val_fn(bio), col_fn(bio), body_start)

    # Bat section
    if bio.bat_detected:
        bat_sep_y = body_start + n_body * (CARD_H + CARD_GAP) + 2
        # Separator with label
        cv2.line(frame, (px1 + 10, bat_sep_y + 8), (px2 - 10, bat_sep_y + 8),
                 C_BORDER, 1)
        cv2.putText(frame, "BAT", ((px1 + px2) // 2 - 12, bat_sep_y + 5),
                    FONT_BOLD, 0.38, C_CYAN, 1, cv2.LINE_AA)

        bat_start = bat_sep_y + CARD_H
        for j, (label, val_fn, col_fn) in enumerate(_BAT_METRICS):
            _draw_card(j, label, val_fn(bio), col_fn(bio), bat_start)


# ── Head ring ─────────────────────────────────────────────────────────────────

_HEAD_COLOURS = {
    "Front":  C_GREEN,
    "Center": C_ACCENT,
    "Back":   C_RED,
}


def draw_head_ring(frame: np.ndarray, bio: BiomechanicsResult, head_xy):
    if not head_xy:
        return
    colour = _HEAD_COLOURS.get(bio.head_position, C_WHITE)
    pt = (int(head_xy[0]), int(head_xy[1]) - 12)

    # Double ring with glow
    _glow_circle(frame, pt, 18, colour, glow_radius=28, intensity=0.25)
    cv2.circle(frame, pt, 21, colour, 1, cv2.LINE_AA)

    # Floating label with background pill
    label = bio.head_position.upper()
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.40, 1)
    lx = pt[0] - tw // 2
    ly = pt[1] - 30
    _alpha_rect(frame, lx - 6, ly - th - 2, lx + tw + 6, ly + 4, C_BG, 0.65)
    cv2.putText(frame, label, (lx, ly),
                FONT_BOLD, 0.40, colour, 1, cv2.LINE_AA)


# ── Knee labels ───────────────────────────────────────────────────────────────

def draw_knee_labels(frame: np.ndarray, bio: BiomechanicsResult,
                     keypoints: np.ndarray, kp_conf: float = 0.3):
    for idx, angle, side in [(13, bio.left_knee_angle, "L"),
                              (14, bio.right_knee_angle, "R")]:
        x, y, conf = keypoints[idx]
        if conf >= kp_conf:
            colour = _status_colour(angle, 115, 165)
            label = f"{angle:.0f}"
            lx, ly = int(x) + 10, int(y) + 4
            # Background pill
            (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.46, 1)
            _alpha_rect(frame, lx - 4, ly - th - 2, lx + tw + 4, ly + 4, C_BG, 0.55)
            cv2.putText(frame, label, (lx, ly),
                        FONT_BOLD, 0.46, colour, 1, cv2.LINE_AA)


# ── Horizontal weight gauge ───────────────────────────────────────────────────

GAUGE_H       = 42
GAUGE_MARGIN  = 14
GAUGE_Y_OFF   = 8


def draw_weight_gauge(frame: np.ndarray, bio: BiomechanicsResult, h: int, w: int,
                      gauge_top: int):
    gx1 = GAUGE_MARGIN
    gx2 = w - GAUGE_MARGIN
    gy1 = gauge_top
    gy2 = gauge_top + GAUGE_H
    bar_w = gx2 - gx1

    # Background panel
    _alpha_rect(frame, 0, gy1 - GAUGE_Y_OFF - 2, w, gy2 + 18, C_BG, 0.85)

    # Label above gauge
    cv2.putText(frame, "WEIGHT DISTRIBUTION", (gx1 + 2, gy1 - 4),
                FONT, 0.36, C_DIM, 1, cv2.LINE_AA)

    # Split point
    split_x = gx1 + int(bar_w * bio.front_foot_weight_pct / 100.0)

    # Gradient fills
    back_left   = (180, 100, 40)
    back_right  = (220, 140, 50)
    front_left  = (40, 160, 70)
    front_right = (70, 220, 100)

    _gradient_rect_h(frame, gx1, gy1, split_x, gy2, back_left, back_right, 0.88)
    _gradient_rect_h(frame, split_x, gy1, gx2, gy2, front_left, front_right, 0.88)

    # 50% center tick + subtle markers at 25% and 75%
    mid_x = gx1 + bar_w // 2
    for pct in [25, 50, 75]:
        tick_x = gx1 + int(bar_w * pct / 100)
        col = (180, 180, 180) if pct == 50 else (70, 70, 70)
        thick = 2 if pct == 50 else 1
        cv2.line(frame, (tick_x, gy1), (tick_x, gy2), col, thick)

    # Needle — glowing triangle + line
    needle_x = split_x
    # Glow behind needle
    _alpha_rect(frame, needle_x - 3, gy1, needle_x + 3, gy2, C_WHITE, 0.25)
    # Triangle marker
    pts = np.array([
        [needle_x,     gy1 - 3],
        [needle_x - 8, gy1 - 14],
        [needle_x + 8, gy1 - 14],
    ], dtype=np.int32)
    cv2.fillPoly(frame, [pts], C_WHITE)
    cv2.polylines(frame, [pts], True, C_BG, 1, cv2.LINE_AA)
    # Needle line
    cv2.line(frame, (needle_x, gy1), (needle_x, gy2), C_WHITE, 2)

    # Outer border
    cv2.rectangle(frame, (gx1, gy1), (gx2, gy2), C_BORDER, 1)

    # Side labels with percentage — styled
    back_col  = C_RED if bio.back_foot_weight_pct > 70 else C_YELLOW if bio.back_foot_weight_pct > 60 else C_WHITE
    front_col = C_RED if bio.front_foot_weight_pct > 75 else C_GREEN if bio.front_foot_weight_pct >= 45 else C_WHITE

    # Back foot (inside bar, left)
    cv2.putText(frame, f"BACK  {bio.back_foot_weight_pct:.0f}%",
                (gx1 + 6, gy1 + 26),
                FONT_BOLD, 0.52, back_col, 1, cv2.LINE_AA)

    # Front foot (inside bar, right-aligned)
    front_label = f"{bio.front_foot_weight_pct:.0f}%  FRONT"
    (fw, _), _ = cv2.getTextSize(front_label, FONT_BOLD, 0.52, 1)
    cv2.putText(frame, front_label, (gx2 - fw - 6, gy1 + 26),
                FONT_BOLD, 0.52, front_col, 1, cv2.LINE_AA)


# ── Insights strip ────────────────────────────────────────────────────────────

INSIGHT_H      = 26
INSIGHT_PAD    = 8
INSIGHT_MARGIN = 14


def draw_insights_strip(frame: np.ndarray, insights: list[Insight],
                        h: int, w: int, strip_top: int):
    if not insights:
        return

    n = len(insights)
    strip_h = n * (INSIGHT_H + 3) + INSIGHT_PAD * 2
    _alpha_rect(frame, 0, strip_top, w, strip_top + strip_h, C_BG, 0.85)

    for i, ins in enumerate(insights):
        y = strip_top + INSIGHT_PAD + i * (INSIGHT_H + 3)
        text_y = y + INSIGHT_H - 6
        colour = _severity_colour(ins.severity)

        # Severity pill badge
        badge = ins.icon
        (bw, bh), _ = cv2.getTextSize(badge, FONT_BOLD, 0.40, 1)
        pill_x1 = INSIGHT_MARGIN
        pill_x2 = pill_x1 + bw + 16
        pill_y1 = y + 2
        pill_y2 = y + INSIGHT_H - 2
        _alpha_rect(frame, pill_x1, pill_y1, pill_x2, pill_y2, colour, 0.25)
        cv2.rectangle(frame, (pill_x1, pill_y1), (pill_x2, pill_y2), colour, 1)
        cv2.putText(frame, badge, (pill_x1 + 8, text_y - 2),
                    FONT_BOLD, 0.40, colour, 1, cv2.LINE_AA)

        # Insight text
        cv2.putText(frame, ins.text, (pill_x2 + 10, text_y - 2),
                    FONT, 0.43, C_WHITE, 1, cv2.LINE_AA)


# ── Bat overlay ───────────────────────────────────────────────────────────────

C_BAT_SHAFT = (40,  220, 240)
C_BAT_HEAD  = (0,   170, 255)
C_BAT_GRIP  = (140, 130, 210)
C_BAT_TRAIL = (40,  200, 220)


def draw_bat_overlay(frame: np.ndarray, bat: BatDetection):
    if not bat:
        return

    trail = bat.trail
    n = len(trail)

    # Swing trail — neon gradient
    if n >= 2:
        overlay = frame.copy()
        for i in range(1, n):
            t = i / n
            brightness = int(t * 240)
            colour = (int(30 + t * 20), brightness, min(brightness + 30, 255))
            thickness = max(1, int(t * 4))
            p1 = (int(trail[i-1][0]), int(trail[i-1][1]))
            p2 = (int(trail[i][0]),   int(trail[i][1]))
            cv2.line(overlay, p1, p2, colour, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

    # Bat shaft with glow
    g  = (int(bat.grip_xy[0]),     int(bat.grip_xy[1]))
    bh = (int(bat.bat_head_xy[0]), int(bat.bat_head_xy[1]))
    _glow_line(frame, g, bh, C_BAT_SHAFT, thickness=3, glow_thickness=10)

    # Bat head
    _glow_circle(frame, bh, 8, C_BAT_HEAD, glow_radius=16, intensity=0.35)

    # Grip
    cv2.circle(frame, g, 5, C_BAT_GRIP, -1, cv2.LINE_AA)
    cv2.circle(frame, g, 5, C_BG, 1, cv2.LINE_AA)

    # Floating HUD — with background pill
    hud_x = bh[0] + 16
    hud_y = bh[1] - 4
    label1 = f"{bat.bat_face_label}  {bat.angle_from_vertical:.0f}"
    label2 = bat.swing_phase
    (tw1, th1), _ = cv2.getTextSize(label1, FONT_BOLD, 0.40, 1)
    (tw2, _), _ = cv2.getTextSize(label2, FONT, 0.36, 1)
    max_tw = max(tw1, tw2)
    _alpha_rect(frame, hud_x - 4, hud_y - th1 - 2, hud_x + max_tw + 8,
                hud_y + 18, C_BG, 0.55)
    cv2.putText(frame, label1, (hud_x, hud_y),
                FONT_BOLD, 0.40, C_BAT_HEAD, 1, cv2.LINE_AA)
    cv2.putText(frame, label2, (hud_x, hud_y + 16),
                FONT, 0.36, C_DIM, 1, cv2.LINE_AA)


# ── Corner branding ───────────────────────────────────────────────────────────

def draw_branding(frame: np.ndarray, h: int, w: int):
    """Subtle bottom-right watermark."""
    label = "CRICKET BIOMECHANICS"
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.35, 1)
    x = w - tw - 10
    y = h - 8
    cv2.putText(frame, label, (x, y), FONT, 0.35, (50, 50, 55), 1, cv2.LINE_AA)


# ── Main annotate ─────────────────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    keypoints: np.ndarray,
    skeleton: list,
    bio: BiomechanicsResult,
    shot: ShotResult,
    fps: float,
    head_xy,
    insights: list[Insight],
    bat_detection: BatDetection | None = None,
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # 0. Vignette (cinematic darkening at edges)
    _apply_vignette(out, strength=0.30)

    # 1. Neon skeleton
    draw_skeleton(out, keypoints)

    # 2. Bat overlay
    if bat_detection:
        draw_bat_overlay(out, bat_detection)

    # 3. Head ring
    draw_head_ring(out, bio, head_xy)

    # 4. Knee labels
    draw_knee_labels(out, bio, keypoints)

    # ── Bottom layout ─────────────────────────────────────────────────────────
    n_insights = len(insights)
    insight_strip_h = n_insights * (INSIGHT_H + 3) + INSIGHT_PAD * 2 if n_insights else 0
    gauge_h_total   = GAUGE_H + GAUGE_Y_OFF + 22
    bottom_block    = insight_strip_h + gauge_h_total

    gauge_top   = h - bottom_block + GAUGE_Y_OFF
    insight_top = gauge_top + GAUGE_H + 20

    # 5. Top bar
    draw_top_bar(out, shot, fps, bio.handedness, h, w)

    # 6. Metrics panel
    draw_metrics_panel(out, bio, h, w)

    # 7. Weight gauge
    draw_weight_gauge(out, bio, h, w, gauge_top)

    # 8. Insights strip
    if insights:
        draw_insights_strip(out, insights, h, w, insight_top)

    # 9. Corner branding
    draw_branding(out, h, w)

    return out
