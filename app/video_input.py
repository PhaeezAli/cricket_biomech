"""
Unified video input abstraction for both webcam and file sources.
"""
import cv2
import numpy as np


class VideoInput:
    """
    Thin wrapper around cv2.VideoCapture that supports both
    live webcam (source=0) and video file (source='path/to/file.mp4').
    Supports use as a context manager.
    """

    def __init__(self, source):
        """
        Args:
            source: int (webcam index, usually 0) or str (path to video file).
        """
        self.source = source
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(f"Could not open video source: {source}")

    def read(self) -> tuple[np.ndarray | None, bool]:
        """Returns (frame, success). Frame is None if read failed."""
        ok, frame = self._cap.read()
        if not ok:
            return None, False
        return frame, True

    def get_fps(self) -> float:
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def get_width(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def get_height(self) -> int:
        return int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame_count(self) -> int:
        """Returns total frame count for file sources (-1 for webcam)."""
        return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        self._cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


def make_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create an OpenCV VideoWriter for saving annotated output."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))
