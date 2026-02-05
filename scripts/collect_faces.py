"""
Standalone face collection script.
Does not require the package to be installed — works as a plain script.

    python scripts/collect_faces.py --name Alice

Reads configs/default.yaml for all parameters.
Run from the repo root directory.
"""

import argparse
import sys
import cv2
import numpy as np
import pickle
from pathlib import Path

# ---------------------------------------------------------------------------
# Inline config loader (avoids importing the package — this script is meant
# to work without `pip install -e .`)
# ---------------------------------------------------------------------------
import yaml


def _load_cfg(path: str) -> dict:
    path = Path(path)
    if not path.exists():
        print(f"[ERROR] Config not found at '{path}'. Run from the repo root.")
        sys.exit(1)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["face"]["cascade"] = cv2.data.haarcascades + cfg["face"]["cascade"]
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Capture face samples for one person")
    parser.add_argument("--name", required=True, help="Person's name (used as label)")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config path")
    args = parser.parse_args()

    cfg = _load_cfg(args.config)
    face_size = tuple(cfg["face"]["size"])
    samples_needed = cfg["face"]["samples_per_person"]
    data_dir = Path(cfg["paths"]["data_dir"])

    # -- detector --
    detector = cv2.CascadeClassifier(cfg["face"]["cascade"])
    if detector.empty():
        print("[ERROR] Failed to load Haar cascade. Check opencv-python installation.")
        sys.exit(1)

    # -- webcam --
    cam = cv2.VideoCapture(cfg["camera"]["index"])
    if not cam.isOpened():
        print("[ERROR] Cannot access webcam.")
        sys.exit(1)

    print(f"\n[INFO] Collecting {samples_needed} samples for: {args.name}")
    print("[INFO] Position your face centrally. Press 'q' to cancel.\n")

    faces_data = []
    frame_count = 0

    while len(faces_data) < samples_needed:
        ok, frame = cam.read()
        if not ok:
            print("[ERROR] Webcam read failed")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector.detectMultiScale(
            gray, cfg["face"]["scale_factor"], cfg["face"]["min_neighbors"]
        )

        for (x, y, w, h) in rects:
            if len(faces_data) < samples_needed and frame_count % 10 == 0:
                face = cv2.resize(frame[y:y+h, x:x+w], face_size)
                faces_data.append(face.reshape(-1))

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"Collected: {len(faces_data)}/{samples_needed}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
            )

        frame_count += 1
        cv2.imshow("Collecting Faces — press Q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[WARNING] Cancelled by user")
            break

    cam.release()
    cv2.destroyAllWindows()

    if not faces_data:
        print("[ERROR] No faces collected.")
        sys.exit(1)

    # -- persist --
    print(f"[SUCCESS] Collected {len(faces_data)} samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    faces_path = data_dir / "faces_data.pkl"
    names_path = data_dir / "names.pkl"

    new_faces = np.array(faces_data)
    new_labels = [args.name] * len(faces_data)

    if faces_path.exists() and names_path.exists():
        with open(faces_path, "rb") as f:
            existing_faces = pickle.load(f)
        with open(names_path, "rb") as f:
            existing_labels = pickle.load(f)
        new_faces = np.vstack([existing_faces, new_faces])
        new_labels = existing_labels + new_labels
        print(f"[INFO] Appended. Total samples: {len(new_faces)}")
    else:
        print(f"[INFO] Created new dataset. Total samples: {len(new_faces)}")

    with open(faces_path, "wb") as f:
        pickle.dump(new_faces, f, protocol=4)
    with open(names_path, "wb") as f:
        pickle.dump(new_labels, f, protocol=4)

    print(f"[SUCCESS] Saved to '{data_dir}'")


if __name__ == "__main__":
    main()
