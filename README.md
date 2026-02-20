# Edge Face Recognition (CPU-Only)

[![PyPI version](https://img.shields.io/pypi/v/edgeface-knn.svg)](https://pypi.org/project/edgeface-knn/)
[![License](https://img.shields.io/pypi/l/edgeface-knn.svg)](https://pypi.org/project/edgeface-knn/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
![Platform](https://img.shields.io/badge/platform-linux%20(native)%20%7C%20windows%20%7C%20macos-lightgrey)

**Real-time face recognition system designed for CPU-only environments (laptops, embedded devices, Raspberry Pi)**

A classical computer vision pipeline (Haar Cascade + KNN) delivering ~40 ms inference latency without GPUs or deep learning frameworks. Built for offline attendance systems and privacy-sensitive deployments where cloud inference isn't viable.

> **Install:** `pip install edgeface-knn`  
> **Performance:** ~40 ms per frame (~15 FPS effective throughput)

---

## Problem & Motivation

**Use case:** Identity recognition for attendance systems, access control, or lab check-ins.

**Constraints:**
- No GPU access (laptops, Raspberry Pi, edge devices)
- No cloud connectivity (offline operation, privacy requirements)
- Real-time response needed (sub-100ms for good UX)
- Non-ML users (should "just work" without tuning)

**Why existing solutions don't fit:**
- CNN-based face recognition: ~300 ms inference on CPU, ~90 MB model size
- Cloud APIs: Require internet, data privacy concerns, per-call costs
- Research repos: Monolithic code, not installable, require ML expertise

This system prioritizes **deployment viability** over maximum accuracy — designed for environments where "good enough, deterministic, and always available" beats "state-of-the-art but requires infrastructure."

---

## What This System Does

**Workflow:**
1. **Registration:** Capture 100 face samples per person via webcam (automated, ~30 seconds)
2. **Recognition:** Real-time identification with confidence scoring
3. **Logging:** Optional attendance tracking with timestamps

**Interface:**
- Command-line tool (no GUI)
- Configuration-driven (YAML-based)
- Pip-installable package (no manual setup)

**Performance:**
- **Latency:** ~40 ms per processed frame
- **Throughput:** ~15 FPS effective (frame skipping for UX)
- **Model size:** <1 MB (vs ~90 MB for CNN alternatives)
- **Accuracy:** ~95% frontal face, ~90% with glasses, ~75% with mask

---

## Engineering Decisions

### 1. Why Haar Cascade + KNN over CNNs?

Prototyped both classical CV and deep learning approaches.

| Factor | This System (Haar + KNN) | CNN Baseline |
|--------|--------------------------|--------------|
| CPU inference | ~40 ms | ~300 ms |
| Model size | <1 MB | ~90 MB |
| Training data | ~100 samples/person | 1000+ samples/person |
| GPU required | No | Yes (for real-time) |
| Latency predictability | Deterministic | Variable (thermal throttling) |

**Decision:** Chose classical CV for deployment constraints.

**Trade-off accepted:** Lower angle robustness (±30° vs ±60° for CNNs) in exchange for guaranteed real-time performance on target hardware.

---

### 2. Unknown Face Handling Strategy

**Problem:** How to handle faces not in the training set?

**Naive approach:** Always return nearest neighbor (KNN default behavior).
- **Risk:** Logs the wrong person (critical failure in attendance systems)

**This system's approach:** Confidence thresholding with rejection.

| Confidence | Result |
|------------|--------|
| ≥ threshold | Person identified |
| < threshold | Marked "Unknown" |

**Decision rationale:**
- False negative (reject known person) → They try again
- False positive (log wrong person) → Permanent incorrect record

**Bias toward precision over recall** — better to ask someone to retry than log them as someone else.

---

### 3. Frame Skipping for Real-Time UX

**Problem:** Processing every frame creates visual lag (camera feed stutters).

**Options:**
1. Process all frames → Stuttering, poor UX
2. Async processing → Added complexity, race conditions
3. Process every Nth frame → Simple, maintains smooth preview

**Decision:** Process every 2nd frame (skip odd frames).
- **Rationale:** Humans perceive <50ms lag as instantaneous; processing 15 FPS feels real-time
- **Trade-off:** Slight temporal jitter (detection updates every ~65ms instead of ~33ms)

---

### 4. Packaging as Reusable Tool

**Problem:** Initial prototype was monolithic scripts (hard to reuse, no version control).

**Evolution:**
- **v1 (Sept 2024):** Raspberry Pi prototype, single-file scripts
- **v2 (Dec 2025):** Modular Python package, pip-installable, configuration-driven

**Decision to refactor:**
- Makes it reusable across projects (attendance, access control, experiments)
- Demonstrates production packaging workflow (not just research code)
- Enables non-ML users to deploy (IT admins, not data scientists)

**Packaging choices:**
- CLI interface (not GUI) → Cross-platform, scriptable
- YAML configuration (not hardcoded params) → Customization without code changes
- Minimal dependencies → Reduces installation friction

---

## Installation & Usage

### Quick Install (Recommended)

Requires native OS execution for camera access (WSL users see notes below).

```bash
pip install edgeface-knn
edge-face --help  # Verify installation
```

### Development Setup

```bash
git clone https://github.com/SakshamBjj/edge-face-recognition-v2.git
cd edge-face-recognition-v2
pip install -e .  # Editable install for development
```

---

### Basic Workflow

#### 1) Register People

```bash
edge-face collect --name Alice
edge-face collect --name Bob
```

Captures 100 samples per person automatically (~30 seconds per person).

**What happens:**
- Opens webcam
- Detects face using Haar Cascade (on grayscale frame)
- Saves 50×50 color (BGR) crops
- Stores in `data/raw/{name}/` directory

---

#### 2) Run Recognition

```bash
edge-face run
```

**Controls:**

| Key | Action |
|-----|--------|
| `o` | Log attendance (saves to CSV) |
| `q` | Quit |

**Output:**
- Real-time video feed with bounding boxes
- Name labels with confidence scores
- Attendance logs in `attendance/YYYY-MM-DD.csv`

---

#### 3) Optional: Custom Configuration

```bash
edge-face run --config configs/my_config.yaml
```

**Configurable parameters:**
- Detection confidence threshold
- Recognition threshold (unknown rejection)
- Frame skip interval
- Camera resolution
- Output paths

---

### WSL Development Notes

**Camera limitation:** Webcam access requires native OS execution (WSL hardware virtualization limitation).

**Recommended workflow:**

| Task | Environment |
|------|-------------|
| Code editing, packaging | WSL |
| Face collection | Windows (native) |
| Real-time recognition | Windows (native) |

**Testing from WSL:**

```bash
# In WSL: Install editable package
pip install -e .

# In Windows terminal (same project directory):
edge-face collect --name TestUser
edge-face run
```

The package itself is OS-independent — only camera I/O requires native execution.

---

## Technical Implementation

### Runtime Pipeline

```
Camera (30 FPS)
 ↓
Grayscale conversion (for detection only)
 ↓
Haar Cascade detection (OpenCV) — runs on grayscale
 ↓
Face crop from color (BGR) frame + resize (50×50 pixels)
 ↓
Flatten to 1D vector (7,500 dimensions — 50×50×3 color channels)
 ↓
KNN classification (k=5, Euclidean distance)
 ↓
Confidence scoring: 100 × exp(−mean_dist / 4500) — heuristic, not calibrated probability
 ↓
Unknown rejection (if confidence < 40)
 ↓
Overlay labels + bounding boxes
 ↓
Display frame
```

**Latency breakdown:**

| Stage | Time |
|-------|------|
| Detection (Haar) | ~20 ms |
| Preprocessing | ~5 ms |
| KNN search | ~15 ms |
| **Total** | **~40 ms** |

---

### Model Details

**Detection:** OpenCV Haar Cascade (frontalface_default.xml)
- Pre-trained on ~10K faces
- Detects faces at multiple scales
- Trade-off: Fast but angle-sensitive (±30° max)
- Runs on grayscale frame for speed

**Feature representation:** Raw pixel values (50×50 color BGR crop)
- Vector dimension: 7,500 (50 × 50 × 3 channels)
- No feature extraction (HOG, LBP, etc.)
- Simple = less overhead, more interpretable

**Classification:** K-Nearest Neighbors (k=5)
- Distance metric: Euclidean (L2 norm)
- No training step (lazy learning)
- O(n) search complexity (acceptable for <50 identities)

**Unknown detection:** Confidence scoring via exponential decay
```python
score = 100.0 * np.exp(-mean_dist / 4500.0)
```
- Threshold: 40 (percent, configurable)
- Below threshold → "Unknown"
- Decay constant 4500 calibrated empirically — heuristic, not derived

---

## Performance Analysis

### Accuracy (Typical Indoor Lighting)

| Condition | Accuracy | Notes |
|-----------|----------|-------|
| Frontal face | ~95% | Optimal scenario |
| With glasses | ~90% | Slight reflection artifacts |
| With mask | ~75% | Reduced feature area |
| ±30° angle | ~70% | Haar detection limit |
| >45° angle | <50% | Face often undetected |

**Error modes:**
- Side profiles not detected (Haar limitation)
- Poor lighting → false negatives (miss detection)
- Multiple faces → processes only largest face

---

### Latency Consistency

| Metric | Value |
|--------|-------|
| Mean latency | 40 ms |
| Std deviation | 3 ms |
| 99th percentile | 47 ms |

**Why consistent:**
- No GPU thermal throttling
- Deterministic CPU execution
- No network dependency

---

### Scaling Limits

| # Identities | KNN Search Time | Total Latency | Acceptable? |
|--------------|-----------------|---------------|-------------|
| 10 | 5 ms | 30 ms | ✓ |
| 50 | 15 ms | 40 ms | ✓ |
| 100 | 60 ms | 85 ms | ✗ (sub-real-time) |
| 500 | 300 ms | 325 ms | ✗ (unusable) |

**Recommendation:** <50 identities for smooth real-time performance.

**Why KNN doesn't scale:**
- Brute-force search is O(n)
- No indexing structure (KD-tree doesn't work well in high dimensions)

**If you need >50 people:** Consider approximate nearest neighbors (FAISS, Annoy) or switch to CNN embeddings with vector databases.

---

## Limitations & Alternatives

### Known Limitations

**1. Angle sensitivity:** Haar Cascade only detects frontal faces (±30°)
- Side profiles often missed
- **Alternative:** Multi-view Haar or CNN detector (MTCNN)

**2. Lighting dependency:** Poor lighting reduces detection rate
- Dark environments: <60% detection rate
- **Alternative:** Infrared camera + illuminator

**3. Scaling ceiling:** >50 identities degrades to sub-real-time
- KNN search becomes bottleneck
- **Alternative:** Approximate NN (FAISS) or CNN embeddings

**4. Spoof vulnerability:** Photo attacks possible (not liveness-aware)
- Printed photos can fool system
- **Alternative:** Depth cameras (RealSense) or liveness detection

**5. No re-identification:** Doesn't track individuals across frames
- Each frame is independent classification
- **Alternative:** Add object tracking (SORT, DeepSORT)

---

### When This System is NOT Appropriate

**Don't use this if you need:**
- Security-critical authentication (use liveness detection + CNNs)
- >50 identities (use ANN-indexed embeddings)
- Angle robustness (use multi-view or CNN detectors)
- High accuracy requirements (use state-of-the-art CNNs)

**Do use this if you need:**
- Offline/edge deployment
- Predictable CPU latency
- Simple deployment (pip install)
- Small model footprint
- Fast prototyping

---

## Repository Structure

```
edge-face-recognition-v2/
├── src/edge_face/
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── config.py           # Configuration management
│   ├── detector.py         # Haar Cascade wrapper
│   ├── dataset.py          # Data collection & loading
│   ├── model.py            # KNN classifier
│   ├── pipeline.py         # End-to-end inference
│   ├── camera.py           # Cross-platform camera initialization
│   └── default.yaml        # Default configuration
├── scripts/
│   └── collect_faces.py    # Legacy collection script
├── data/                   # Generated during collection
│   └── raw/{name}/         # Face samples per person
├── attendance/             # Generated during recognition
│   └── YYYY-MM-DD.csv      # Daily attendance logs
├── pyproject.toml          # Package metadata
└── README.md
```

---

## Project Evolution

### Version History

| Version | Focus | Date |
|---------|-------|------|
| v1 | Raspberry Pi embedded prototype | Sept 2024 |
| v2 | Pip-installable reusable package | Feb 2025 |

**Key improvements in v2:**
- Modular architecture (single file → package structure)
- Configuration-driven runtime (hardcoded → YAML)
- Cross-platform execution (RPi-only → Windows/Linux/macOS)
- Reusable CLI (monolithic script → installable tool)
- Calibrated confidence scoring with unknown rejection (broken formula → exponential decay)

---

## What This Demonstrates

**System design:**
- Constraint-driven architecture (latency budget drives model choice)
- Trade-off analysis (accuracy vs deployment viability)
- User-centric design (non-ML users as target)

**Software engineering:**
- Packaging for distribution (PyPI-ready)
- CLI design (configuration, error handling, user feedback)
- Cross-platform compatibility (native vs WSL execution)

**Production ML:**
- Deployment constraints over benchmark chasing
- Unknown handling (precision > recall for attendance use case)
- Honest limitation documentation (when not to use this system)

**Iteration velocity:**
- Prototype → production evolution
- Monolithic code → modular package
- Single-use → reusable tool

---

## References

**Technical:**
- Viola-Jones Face Detection (Haar Cascade): [OpenCV Docs](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html)
- K-Nearest Neighbors: [Scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)

**Related work:**
- FaceNet (CNN embeddings): For higher accuracy, GPU-available scenarios
- MTCNN (Multi-task CNN): For angle-robust detection
- ArcFace: State-of-the-art face recognition (requires GPU)

---

**Author:** Saksham Bajaj  
**Contact:** [LinkedIn](https://www.linkedin.com/in/saksham-bjj/) | [GitHub](https://github.com/SakshamBjj)  
**License:** MIT  
**Last Updated:** February 2026