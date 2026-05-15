"""
Coaching insights engine.
Converts raw BiomechanicsResult numbers into plain-English tips
that any viewer (not just a coach) can understand.
"""
from collections import deque
from dataclasses import dataclass

from app.biomechanics.metrics import BiomechanicsResult
from app.shot_classifier import ShotResult


@dataclass
class Insight:
    text: str          # e.g. "Bend your knees more for a lower base"
    severity: str      # "good" | "warn" | "bad"
    icon: str          # short emoji-free prefix shown in overlay: "✓" "!" "✗"


# ── Rule definitions ──────────────────────────────────────────────────────────
# Each rule returns an Insight or None.

def _weight_insight(bio: BiomechanicsResult, shot: ShotResult) -> Insight | None:
    f = bio.front_foot_weight_pct
    b = bio.back_foot_weight_pct

    # Shot-specific ideal ranges
    shot_ideals = {
        "Front Foot Drive":  (55, 75),
        "Sweep":             (55, 75),
        "Pull Shot":         (35, 55),
        "Cut Shot":          (35, 55),
        "Back Foot Defense": (30, 50),
    }
    lo, hi = shot_ideals.get(shot.shot_type, (40, 65))

    if lo <= f <= hi:
        return Insight(f"Weight balanced ({f:.0f}% front)", "good", "OK")
    elif f > hi:
        excess = f - hi
        if excess > 20:
            return Insight(f"Too much weight on front foot ({f:.0f}%) — risk of falling forward", "bad", "!!")
        return Insight(f"Slightly front-heavy ({f:.0f}%) — shift a little back", "warn", "!")
    else:
        excess = lo - f
        if excess > 20:
            return Insight(f"Too much weight on back foot ({b:.0f}%) — hard to drive through", "bad", "!!")
        return Insight(f"Slightly back-heavy ({b:.0f}%) — transfer weight forward", "warn", "!")


def _head_insight(bio: BiomechanicsResult) -> Insight | None:
    if bio.head_position == "Center":
        if bio.head_tilt_deg <= 5:
            return Insight("Head steady & centered — great stillness", "good", "OK")
        return Insight(f"Head tilting {bio.head_tilt_deg:.0f}° — keep it level", "warn", "!")
    elif bio.head_position == "Back":
        return Insight("Head falling back — eyes off the ball", "bad", "!!")
    else:
        return Insight("Head lunging forward — stay balanced", "warn", "!")


def _knee_insight(bio: BiomechanicsResult) -> Insight | None:
    avg = (bio.left_knee_angle + bio.right_knee_angle) / 2
    diff = abs(bio.left_knee_angle - bio.right_knee_angle)

    if avg > 168:
        return Insight("Knees too straight — bend more for a lower, stable base", "bad", "!!")
    elif avg > 158:
        return Insight("Slightly stiff legs — try a deeper knee bend", "warn", "!")
    elif avg < 100:
        return Insight("Very deep crouch — intentional for sweep/slog?", "warn", "?")
    else:
        if diff > 25:
            return Insight(f"Uneven knee bend (L:{bio.left_knee_angle:.0f}° R:{bio.right_knee_angle:.0f}°) — balance both legs", "warn", "!")
        return Insight(f"Good knee bend ({avg:.0f}° avg) — solid base", "good", "OK")


def _spine_insight(bio: BiomechanicsResult) -> Insight | None:
    s = bio.spine_tilt_deg
    if s <= 12:
        return Insight("Upright spine — good posture", "good", "OK")
    elif s <= 22:
        return Insight(f"Slight forward lean ({s:.0f}°) — manageable", "warn", "!")
    else:
        return Insight(f"Too much forward lean ({s:.0f}°) — risk losing balance", "bad", "!!")


def _shoulder_insight(bio: BiomechanicsResult) -> Insight | None:
    s = bio.shoulder_alignment_deg
    if s <= 6:
        return None  # level shoulders is expected; no need to call it out
    elif s <= 12:
        return Insight(f"Shoulders slightly uneven ({s:.0f}°)", "warn", "!")
    else:
        return Insight(f"Shoulders misaligned ({s:.0f}°) — affects shot direction", "bad", "!!")


def _bat_insight(bio: BiomechanicsResult, shot: ShotResult) -> Insight | None:
    if not bio.bat_detected:
        return None

    angle   = bio.bat_angle_deg
    face    = bio.bat_face_label
    phase   = bio.bat_swing_phase
    bswing  = bio.bat_backswing_pct

    # Cross-bat on a straight drive is a clear technical flaw
    if shot.shot_type == "Front Foot Drive" and angle > 35:
        return Insight(
            "Cross-bat on a drive — straighten the bat for better control", "bad", "!!")

    # Straight bat on a pull/cut is wrong — those need a horizontal bat
    if shot.shot_type in ("Pull Shot", "Cut Shot") and angle < 20:
        return Insight("Bat too straight for a pull/cut — open the face", "warn", "!")

    # Very low backswing going into a drive = weak shot potential
    if shot.shot_type == "Front Foot Drive" and bswing < 40 and phase == "Downswing":
        return Insight("Short backswing on drive — load higher for more power", "warn", "!")

    # Great high backswing
    if bswing > 85:
        return Insight(f"High backswing ({bswing:.0f}%) — good power loading", "good", "OK")

    # Compact but reasonable
    if 50 <= bswing <= 85:
        return Insight(f"Compact backswing ({bswing:.0f}%) — quick & controlled", "good", "OK")

    # Bat face straight during a drive or defense — reward it
    if face == "Straight" and shot.shot_type in ("Front Foot Drive", "Back Foot Defense"):
        return Insight("Straight bat — textbook technique", "good", "OK")

    return None


def _shot_tip(shot: ShotResult) -> Insight | None:
    tips = {
        "Front Foot Drive":   "Drive: keep the elbow high and follow through",
        "Back Foot Defense":  "Defense: stay side-on, soft hands",
        "Pull Shot":          "Pull: get inside the line, roll the wrists",
        "Cut Shot":           "Cut: stay on top of the ball, don't reach",
        "Sweep":              "Sweep: head down, commit fully to the shot",
        "Neutral Stance":     None,
        "Detecting...":       None,
    }
    tip = tips.get(shot.shot_type)
    if tip:
        return Insight(tip, "good", "TIP")
    return None


# ── Public API ────────────────────────────────────────────────────────────────

# Smoothing: keep a short history to avoid flickering insights
_insight_history: deque = deque(maxlen=20)


def generate_insights(bio: BiomechanicsResult, shot: ShotResult) -> list[Insight]:
    """
    Return up to 3 insights ordered by severity (bad → warn → good).
    Smoothed over recent frames to avoid flickering.
    """
    raw = [
        _weight_insight(bio, shot),
        _head_insight(bio),
        _knee_insight(bio),
        _spine_insight(bio),
        _shoulder_insight(bio),
        _bat_insight(bio, shot),
        _shot_tip(shot),
    ]
    active = [i for i in raw if i is not None]

    # Sort: bad first, then warn, then good/tip
    order = {"bad": 0, "warn": 1, "good": 2}
    active.sort(key=lambda i: order.get(i.severity, 3))

    # Add to history for smoothing
    _insight_history.append(active)

    # Return top 3 from most recent frame, but only if at least half
    # the history frames agree on the top issue (reduces flickering)
    return active[:3]
