"""
Real-time recognition loop.
Reads webcam frames, runs detection + KNN, draws overlay,
and handles the 'o' / 'q' key controls.
"""

import cv2
import csv
from datetime import datetime
from pathlib import Path

from .detector import FaceDetector
from .model import FaceKNN


class RecognitionPipeline:
    def __init__(self, detector: FaceDetector, model: FaceKNN, cfg: dict):
        self.detector = detector
        self.model = model
        self.cfg = cfg
        self.face_size = tuple(cfg["face"]["size"])
        self.conf_threshold = cfg["runtime"]["confidence_threshold"]
        self.frame_skip = cfg["runtime"]["frame_skip"]
        self.attendance_dir = Path(cfg["paths"]["attendance_dir"])
        self.unknown_label = cfg["runtime"]["unknown_label"]
        self.reject_unknowns = cfg["runtime"]["reject_unknowns"]

    # ------------------------------------------------------------------
    # attendance
    # ------------------------------------------------------------------
    def _log_attendance(self, names: list[str]):
        self.attendance_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        csv_path = self.attendance_dir / f"{now.strftime('%Y-%m-%d')}.csv"
        file_exists = csv_path.exists()

        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["NAME", "TIME"])
            for name in names:
                time_str = now.strftime("%H:%M:%S")
                writer.writerow([name, time_str])
                print(f"[ATTENDANCE] {name} at {time_str}")

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, video: cv2.VideoCapture):
        print("\n[INFO] Face recognition started")
        print("[CONTROLS] 'o' = log attendance | 'q' = quit\n")

        frame_id = 0
        current_detections: dict[str, float] = {}  # name -> confidence

        while True:
            ok, frame = video.read()
            if not ok:
                print("[ERROR] Webcam read failed")
                break

            # Frame skipping: only run detection on every Nth frame.
            # On skipped frames we still display the last overlay.
            if frame_id % self.frame_skip != 0:
                frame_id += 1
                cv2.imshow("Edge Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect(gray)

            current_detections.clear()

            for (x, y, w, h) in faces:
                face = cv2.resize(frame[y:y+h, x:x+w], self.face_size)
                vec = face.reshape(1, -1)

                name = self.model.predict(vec)
                conf = self.model.confidence(vec)
                
                # Unknown face rejection: replace low-confidence predictions
                if self.reject_unknowns and conf < self.conf_threshold:
                    name = self.unknown_label
                
                # Only track known faces for attendance (exclude unknowns)
                if name != self.unknown_label:
                    current_detections[name] = conf

                # Visual distinction: green for known (high conf), orange for low conf/unknown
                color = (0, 255, 0) if conf >= self.conf_threshold else (0, 165, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(
                    frame, f"{name} ({conf:.0f}%)", (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

            cv2.imshow("Edge Face Recognition", frame)
            frame_id += 1

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("[INFO] Quit")
                break
            elif key == ord("o") and current_detections:
                self._log_attendance(list(current_detections.keys()))

        video.release()
        cv2.destroyAllWindows()
