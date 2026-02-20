"""
CLI entrypoint.  Installed as the 'edge-face' command via pyproject.toml.

    edge-face collect --name Alice       # capture 100 face samples
    edge-face run                        # launch real-time recognition
    edge-face run --config my_config.yaml
"""

import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

from .config import load_config
from .detector import FaceDetector
from .dataset import FaceDataset
from .model import FaceKNN
from .pipeline import RecognitionPipeline
from .camera import open_camera


# ------------------------------------------------------------------
# collect
# ------------------------------------------------------------------
def _collect(args):
    cfg = load_config(args.config)
    face_size = tuple(cfg["face"]["size"])
    samples_needed = cfg["face"]["samples_per_person"]
    data_dir = cfg["paths"]["data_dir"]

    detector = FaceDetector(
        cfg["face"]["cascade"],
        cfg["face"]["scale_factor"],
        cfg["face"]["min_neighbors"],
    )

    cam = open_camera(
        cfg["camera"]["index"],
        cfg["camera"]["width"],
        cfg["camera"]["height"]
    )

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
        rects = detector.detect(gray)

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
        cv2.imshow("Collecting Faces - press Q to cancel", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[WARNING] Cancelled by user")
            break

    cam.release()
    cv2.destroyAllWindows()

    if not faces_data:
        print("[ERROR] No faces collected. Exiting.")
        sys.exit(1)

    print(f"[SUCCESS] Collected {len(faces_data)} samples")
    dataset = FaceDataset(data_dir)
    dataset.append(np.array(faces_data), args.name)


# ------------------------------------------------------------------
# run
# ------------------------------------------------------------------
def _run(args):
    cfg = load_config(args.config)

    dataset = FaceDataset(cfg["paths"]["data_dir"])
    X, y = dataset.load()
    print(f"[INFO] Loaded {len(X)} samples, {len(set(y))} individuals")

    model = FaceKNN(cfg["knn"]["k"], cfg["knn"]["weights"])
    model.train(X, y)

    detector = FaceDetector(
        cfg["face"]["cascade"],
        cfg["face"]["scale_factor"],
        cfg["face"]["min_neighbors"],
    )

    cam = open_camera(
        cfg["camera"]["index"],
        cfg["camera"]["width"],
        cfg["camera"]["height"]
    )

    RecognitionPipeline(detector, model, cfg).run(cam)


# ------------------------------------------------------------------
# parser
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        prog="edge-face",
        description="CPU-only real-time face recognition (Haar Cascade + KNN)",
    )
    sub = parser.add_subparsers(dest="command")

    # collect
    collect_parser = sub.add_parser("collect", help="Capture face samples for one person")
    collect_parser.add_argument("--name", required=True, help="Person's name (label)")
    collect_parser.add_argument("--config", default="default.yaml", help="Path to YAML config")

    # run
    run_parser = sub.add_parser("run", help="Launch real-time recognition")
    run_parser.add_argument("--config", default="default.yaml", help="Path to YAML config")

    args = parser.parse_args()

    if args.command == "collect":
        _collect(args)
    elif args.command == "run":
        _run(args)
    else:
        parser.print_help()