# Edge Face Recognition (CPU-Only)

[![PyPI version](https://img.shields.io/pypi/v/edgeface-knn.svg)](https://pypi.org/project/edgeface-knn/)
[![License](https://img.shields.io/pypi/l/edgeface-knn.svg)](https://pypi.org/project/edgeface-knn/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![Platform](https://img.shields.io/badge/platform-linux%20(native)%20%7C%20windows%20%7C%20macos-lightgrey)

**Real-time face recognition designed for CPU-only environments (laptops, embedded devices, Raspberry Pi).**

A classical computer-vision pipeline (Haar Cascade + KNN) delivering ~40 ms inference latency without GPUs or deep-learning frameworks. Intended for offline attendance and privacy-sensitive deployments where cloud inference is not viable.

> **Install:** `pip install edgeface-knn`  
> **Package name:** `edgeface-knn` (PyPI) | **Repository:** `edge-face-recognition-v2` (GitHub)

**Performance:** ~40 ms per processed frame (~15 FPS effective)

---

## Problem Context

A lightweight identity recognition system intended for:

* Attendance systems
* Lab / hostel / office access logging
* Offline environments
* Edge devices with no GPU
* Privacy-sensitive deployments (no cloud inference)

The system prioritizes **correct identification over aggressive guessing** — unknown faces are rejected instead of force-matched.

---

## Who is this for?

- **Want to use it** → follow Quick Install
- **Want to modify it** → follow Development Setup

---

## Installation

### Quick Install (Recommended)
Camera capture requires native OS execution (WSL users see section below).

```bash
pip install edgeface-knn
edge-face --help
# Expected: shows collect/run commands
```

### Development Setup

```bash
git clone https://github.com/SakshamBjj/edge-face-recognition-v2.git
cd edge-face-recognition-v2
pip install -e .
```

> Originally built as a Raspberry Pi prototype (Sept 2024).  
> Refactored into a modular installable Python package (Dec 2025).

---

## Usage

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

Override default parameters:

```bash
edge-face run --config configs/my_config.yaml
```

---

## WSL Development Notes

> **Important:** Webcam access requires native OS execution. This limitation comes from WSL hardware virtualization — not the library.

**Development workflow:**

| Task                     | Environment       |
| ------------------------ | ----------------- |
| Code editing / packaging | WSL               |
| Face collection          | Windows (native)  |
| Real-time recognition    | Windows (native)  |

**Testing from WSL:**

```bash
# In WSL: Install editable package
pip install -e .

# In Windows terminal (same project directory):
edge-face collect --name TestUser
edge-face run
```

The package itself is OS-independent, but webcam access requires native execution.

---

## Runtime Pipeline

```
Camera (30 FPS)
 → Grayscale conversion
 → Haar Cascade detection
 → Crop + resize (50×50)
 → Flatten vector
 → KNN classification
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
| < threshold | Marked "Unknown"  |

Prevents the most serious failure in face recognition systems: logging the wrong person.

---

## Model Selection Rationale

| Factor        | This Project (KNN)  | CNN Face Recognition |
| ------------- | ------------------- | -------------------- |
| Model size    | <1 MB               | ~90 MB               |
| CPU inference | ~40 ms              | ~300 ms              |
| GPU required  | No                  | Yes                  |
| Training data | ~100 samples/person | 1000+ samples/person |

**Design goal:** predictable latency on CPU hardware — not maximum accuracy on servers.

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

## Engineering Tradeoffs

| Decision          | Reason                         | Cost                        |
| ----------------- | ------------------------------ | --------------------------- |
| Haar Cascade      | 20 ms detection                | Angle robustness            |
| Raw pixels        | No feature extraction overhead | Less compact representation |
| KNN               | No training step               | Scaling limits              |
| Frame skipping    | Real-time UX                   | Slight temporal jitter      |
| Unknown rejection | Avoid false positives          | Occasional false negatives  |


---

## Package Layout

```
edge-face-recognition-v2/
├── src/edge_face/
├── ├── default.yaml
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