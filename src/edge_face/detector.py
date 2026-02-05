"""
Face detection via Haar Cascade.
Wraps OpenCV's CascadeClassifier — keeps detection params in one place
so they can be tuned via config without touching inference or collection code.
"""

import cv2


class FaceDetector:
    def __init__(self, cascade_path: str, scale_factor: float, min_neighbors: int):
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise IOError(
                f"Failed to load Haar cascade from '{cascade_path}'.\n"
                "Check that opencv-python is installed and the cascade XML exists."
            )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, gray_frame):
        """Return list of (x, y, w, h) bounding boxes for detected faces."""
        return self.detector.detectMultiScale(
            gray_frame, self.scale_factor, self.min_neighbors
        )
