"""
Cricket Biomechanics Analyzer — entry point.

Usage:
  # Live webcam
  python app/main.py --source webcam

  # Video file (display only)
  python app/main.py --source file --input data/input_videos/batting.mp4

  # Video file with saved output
  python app/main.py --source file --input data/input_videos/batting.mp4 \
                     --output data/output_videos/annotated.mp4

Keyboard controls (when display window is open):
  q  — quit
  s  — save current frame as snapshot
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

# Ensure repo root is on sys.path when running as `python app/main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.video_input import VideoInput, make_video_writer
from app.pose_estimator import PoseEstimator
from app.bat_detector import BatDetector
from app.biomechanics import analyse
from app.shot_classifier import ShotClassifier, ShotResult
from app.insights import generate_insights
from app.visualizer import annotate_frame


# ── Handedness auto-detection ─────────────────────────────────────────────────

def _detect_handedness(estimator: PoseEstimator, frame_buffer: list) -> str:
    """
    Inspect the first N frames to decide if the batsman is right or left handed.
    Right-handed: left ankle (kp[15]) has lower x than right ankle (kp[16]).
    Returns "right" or "left".
    """
    left_lower_count = 0
    total = 0
    for kp in frame_buffer:
        lx, _, lc = kp[15]
        rx, _, rc = kp[16]
        if lc >= estimator.KP_CONF_THRESHOLD and rc >= estimator.KP_CONF_THRESHOLD:
            if lx < rx:
                left_lower_count += 1
            total += 1
    if total == 0:
        return "right"  # default
    return "right" if left_lower_count / total >= 0.5 else "left"


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(source, input_path: str | None, output_path: str | None,
        handedness: str | None, bat_model_path: str | None = None):
    video_source = 0 if source == "webcam" else input_path
    estimator   = PoseEstimator()
    bat_detector = BatDetector(bat_model_path)
    classifier  = ShotClassifier()

    with VideoInput(video_source) as video:
        fps_source = video.get_fps()
        w, h = video.get_width(), video.get_height()

        writer = None
        if output_path:
            writer = make_video_writer(output_path, fps_source, w, h)
            print(f"Saving output to: {output_path}")

        # ── Phase 1: Handedness detection (first 10 frames) ──────────────────
        auto_detect = handedness is None
        kp_buffer = []
        DETECT_FRAMES = 10

        # ── FPS tracking ─────────────────────────────────────────────────────
        prev_time = time.time()
        display_fps = 0.0

        default_shot = ShotResult(shot_type="Detecting...", confidence=0.0)
        current_shot = default_shot
        locked_handedness = handedness or "right"

        frame_idx = 0
        snapshot_idx = 0

        print("Press 'q' to quit, 's' to save a snapshot frame.")

        while True:
            frame, ok = video.read()
            if not ok:
                break

            # ── Pose estimation ───────────────────────────────────────────────
            person = estimator.estimate(frame)

            if person is None:
                annotated = frame.copy()
                cv2.putText(annotated, "No batsman detected",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 220), 2)
                insights = []
            else:
                # ── Auto-detect handedness from first DETECT_FRAMES ───────────
                if auto_detect and frame_idx < DETECT_FRAMES:
                    kp_buffer.append(person.keypoints)
                    if frame_idx == DETECT_FRAMES - 1:
                        locked_handedness = _detect_handedness(estimator, kp_buffer)
                        print(f"Auto-detected handedness: {locked_handedness}")

                # ── Build get_xy helper for this frame ────────────────────────
                def get_xy(name, _kp=person.keypoints):
                    return estimator.get_xy(_kp, name)

                # ── Bat detection ─────────────────────────────────────────────
                bat = bat_detector.detect(frame, person.keypoints,
                                          person_bbox=person.bbox)

                # ── Biomechanics analysis ─────────────────────────────────────
                bio = analyse(person, locked_handedness, get_xy,
                              bat_detection=bat)

                # ── Shot classification ───────────────────────────────────────
                current_shot = classifier.update(bio)

                # ── Insights ──────────────────────────────────────────────────
                insights = generate_insights(bio, current_shot)

                # ── Head xy for visualiser ────────────────────────────────────
                lear = get_xy("left_ear")
                rear  = get_xy("right_ear")
                if lear and rear:
                    head_xy = ((lear[0] + rear[0]) / 2, (lear[1] + rear[1]) / 2)
                else:
                    head_xy = get_xy("nose")

                # ── FPS ───────────────────────────────────────────────────────
                now = time.time()
                display_fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                # ── Annotate ──────────────────────────────────────────────────
                annotated = annotate_frame(
                    frame,
                    person.keypoints,
                    estimator.SKELETON,
                    bio,
                    current_shot,
                    display_fps,
                    head_xy,
                    insights,
                    bat_detection=bat,
                )

            # ── Write to file ─────────────────────────────────────────────────
            if writer:
                writer.write(annotated)

            # ── Display ───────────────────────────────────────────────────────
            cv2.imshow("Cricket Biomechanics", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                snap_path = f"data/output_videos/snapshot_{snapshot_idx:04d}.jpg"
                cv2.imwrite(snap_path, annotated)
                print(f"Snapshot saved: {snap_path}")
                snapshot_idx += 1

            frame_idx += 1

    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cricket Biomechanics Analyzer")
    parser.add_argument("--source", choices=["webcam", "file"], required=True,
                        help="Video source: 'webcam' or 'file'")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input video file (required for --source file)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save annotated output video (optional)")
    parser.add_argument("--handedness", choices=["left", "right"], default=None,
                        help="Batsman handedness. If omitted, auto-detected from pose.")
    parser.add_argument("--bat-model", type=str, default=None, dest="bat_model",
                        help="Path to a custom YOLO bat detection model (optional). "
                             "Without this, geometric inference is used automatically.")
    args = parser.parse_args()

    if args.source == "file" and not args.input:
        parser.error("--input is required when --source is 'file'")

    run(
        source=args.source,
        input_path=args.input,
        output_path=args.output,
        handedness=args.handedness,
        bat_model_path=args.bat_model,
    )


if __name__ == "__main__":
    main()
