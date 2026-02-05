# Edge Face Recognition (CPU-Only)

[![PyPI version](https://img.shields.io/pypi/v/edgeface-knn.svg)](https://pypi.org/project/edgeface-knn/)
[![License](https://img.shields.io/pypi/l/edgeface-knn.svg)](https://pypi.org/project/edgeface-knn/)

> ⚠️ Repository name contains `v2` to indicate the refactored codebase.  
> The installable Python package is **`edgeface-knn`** and remains versioned independently of the repository name.

Real-time face recognition designed for CPU-only environments — laptops, embedded systems, and Raspberry-Pi-class devices.

Classical computer vision pipeline (Haar Cascade + KNN) engineered for deterministic low-latency inference without GPUs or deep learning frameworks.

**Latency:** ~40 ms per processed frame
**Effective throughput:** ~15 FPS (frame-skipped real-time UX)

> Originally built as a Raspberry Pi prototype (Sept 2024).
> Refactored into a modular installable Python package (Dec 2025).

---

## What this project is

A lightweight identity recognition system intended for:

* Attendance systems
* Lab / hostel / office access logging
* Offline environments
* Edge devices with no GPU
* Privacy-sensitive deployments (no cloud inference)

The system prioritizes **correct identification over aggressive guessing** — unknown faces are rejected instead of force-matched.

---

## Quick Install (Windows — recommended for usage)

Install the published package and run directly:

```bash
pip install edgeface-knn
edge-face --help
```

---

### Collect faces

```bash
edge-face collect --name Alice
edge-face collect --name Bob
```

Captures 100 samples per person via webcam automatically.

---

### Run recognition

```bash
edge-face run
```

Controls:

| Key | Action         |
| --- | -------------- |
| `o` | Log attendance |
| `q` | Quit           |

Logs saved to:

```
attendance/YYYY-MM-DD.csv
```

---

## Development Setup (WSL)

Use this only if you want to modify the codebase or build the package.

> WSL does not provide webcam access to OpenCV — runtime testing must be done on Windows.

```bash
git clone https://github.com/SakshamBjj/edge-face-recognition-v2.git
cd edge-face-recognition-v2
pip install -e .
edge-face --help
```

---

## Testing the Development Version (Windows)

After installing the editable package from WSL, open **Windows terminal (PowerShell/CMD)** inside the same project and run:

```bash
edge-face collect --name TestUser
edge-face run
```

---

### 3) Optional configuration

Override default parameters:

```bash
edge-face run --config configs/my_config.yaml
```

---

## Why this split exists

WSL is a virtualized Linux environment and does not provide direct access to webcam hardware.
The package itself is OS-independent, but real-time capture requires native OS execution.

Typical workflow:

| Task                    | Environment |
| ----------------------- | ----------- |
| Development / packaging | WSL         |
| Face collection         | Windows     |
| Real-time recognition   | Windows     |

---

## Runtime Pipeline

```
Camera (30 FPS)
 → Grayscale conversion
 → Haar Cascade detection (~20 ms)
 → Crop + resize (50×50)
 → Flatten vector
 → KNN classification (~15 ms)
 → Confidence scoring
 → Unknown rejection
 → Overlay + logging
```

Frame skipping processes every 2nd frame to maintain smooth real-time UX.

---

## Unknown Face Handling

The system favors **precision over recall**.

Instead of always predicting a nearest neighbor:

| Confidence  | Result            |
| ----------- | ----------------- |
| ≥ threshold | Person identified |
| < threshold | Marked “Unknown”  |

Prevents the most serious failure in face recognition systems: logging the wrong person.

---

## Why Classical ML instead of Deep Learning?

| Factor        | This Project (KNN)  | CNN Face Recognition |
| ------------- | ------------------- | -------------------- |
| Model size    | <1 MB               | ~90 MB               |
| CPU inference | ~40 ms              | ~300 ms              |
| GPU required  | No                  | Yes                  |
| Training data | ~100 samples/person | 1000+ samples/person |

**Design goal:** predictable latency on CPU hardware
—not maximum accuracy on servers.

Deep learning was prototyped but exceeded real-time limits without GPU acceleration.

---

## Performance

### Accuracy (typical indoor lighting)

| Condition    | Accuracy |
| ------------ | -------- |
| Frontal face | ~95%     |
| Glasses      | ~90%     |
| Mask         | ~75%     |
| ±30° angle   | ~70%     |

### Latency

| Stage          | Time       |
| -------------- | ---------- |
| Detection      | 20 ms      |
| Preprocess     | 5 ms       |
| Classification | 15 ms      |
| **Total**      | **~40 ms** |

---

## Known Limitations

1. Low lighting reduces detection reliability
2. Side profiles (>30°) often not detected
3. Performance degrades beyond ~100 identities (KNN O(n))
4. Not spoof-proof (photo attacks possible)

---

## Package Layout

```
edge-face-recognition-v2/
├── configs/default.yaml
├── src/edge_face/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── detector.py
│   ├── dataset.py
│   ├── model.py
│   └── pipeline.py
├── scripts/collect_faces.py
├── data/              (generated)
└── attendance/        (generated)
```

---

## Engineering Tradeoffs

| Decision          | Reason                         | Cost                        |
| ----------------- | ------------------------------ | --------------------------- |
| Haar Cascade      | 20 ms detection                | Angle robustness            |
| Raw pixels        | No feature extraction overhead | Less compact representation |
| KNN               | No training step               | Scaling limits              |
| Frame skipping    | Real-time UX                   | Slight temporal jitter      |
| Unknown rejection | Avoid false positives          | Occasional false negatives  |

---

## Project Evolution

| Version | Focus                                 |
| ------- | ------------------------------------- |
| v1      | Embedded Raspberry Pi prototype       |
| v2      | Installable reusable software package |

Archived prototype available in repository history.

---

## What this demonstrates

* Designing ML systems under hardware constraints
* Latency-driven model selection
* Converting prototype code into a distributable tool
* Building CLI-driven reproducible workflows

---

## References

* Viola-Jones Face Detection
* OpenCV face recognition documentation
* scikit-learn KNN implementation

---

**Author:** Saksham Bajaj
**License:** MIT