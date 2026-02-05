# Edge Face Recognition (CPU-Only)

> ⚠️ Repository name contains `v2` to indicate the refactored codebase.  
> The installable Python package remains `edge-face-recognition` (no v2) to preserve upgrade compatibility.

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

## Installation

### From source (recommended)

```bash
git clone https://github.com/SakshamBjj/edge-face-recognition-v2.git
cd edge-face-recognition-v2
pip install -e .
```

Verify:

```bash
edge-face --help
```

### After PyPI release

```bash
pip install edgeface-knn
```

---

## Quick Start

### 1) Register people

```bash
edge-face collect --name Alice
edge-face collect --name Bob
```

Captures 100 samples per person via webcam automatically.

---

### 2) Run recognition

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

### 3) Optional configuration

All runtime parameters are editable:

```bash
edge-face run --config configs/my_config.yaml
```

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