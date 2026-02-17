# Contributing Guide

This project is maintained primarily as a portfolio piece demonstrating CPU-constrained ML system design and production Python packaging. External contributions are accepted case-by-case.

For design rationale and architectural decisions, see [ARCHITECTURE.md](ARCHITECTURE.md). This document covers setup, conventions, known constraints, and contribution process.

---

## Local Setup

**Requirement:** Native OS execution for webcam access. WSL works for editing and packaging; camera I/O requires Windows or Linux native.

```bash
git clone https://github.com/SakshamBjj/edge-face-recognition-v2.git
cd edge-face-recognition-v2

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -e .
edge-face --help                # Verify installation
```

**WSL workflow:**

```bash
# WSL: package install and editing
pip install -e .

# Windows terminal (same directory): camera commands
edge-face collect --name TestUser
edge-face run
```

---

## Testing Workflow

No automated test suite — webcam interaction requires manual verification.

```bash
# 1. Collect samples
edge-face collect --name Alice
# → Verify: data/raw/Alice/ contains 100 .jpg files

# 2. Run recognition
edge-face run
# → Alice: green bbox, confidence >40%
# → Uncollected face: "Unknown" label, orange bbox

# 3. Attendance logging
edge-face run
# → Press 'o' when Alice detected
# → Verify: attendance/YYYY-MM-DD.csv contains entry
```

**Confidence sanity check:** If all faces return Unknown regardless of enrollment, the dataset may be using different vector dimensions than what was trained on. Recollect and retrain from scratch.

---

## Known Constraints

These are documented design trade-offs, not bugs. Contributions addressing them are welcome if they stay within the CPU-only constraint.

| Constraint | Current Limit | Effort to Fix |
|------------|---------------|---------------|
| Scalability | <50 identities for real-time (>50 → sub-100ms fails) | Medium — replace KNN with FAISS ANN |
| Angular robustness | ±30° max (Haar limitation) | High — requires MTCNN (adds ~180ms CPU latency) |
| Low-light performance | <60% detection rate below 100 lux | Low — add CLAHE preprocessing (+8ms) |
| Confidence flicker | Near-threshold faces flicker between Known/Unknown | Low — add hysteresis or EMA smoothing |
| No liveness detection | Photo replay attacks succeed | High — blink detection or depth camera |
| No temporal tracking | Each frame is independent classification | Medium — integrate SORT/DeepSORT |

**Confidence Scoring Evolution:** The v1 prototype used a linear formula (`100 - distance`) which always returned 0 for 7,500D vectors (distances of ~1,600–1,800 make the result always negative). v2 replaced this with exponential decay: `score = 100 × exp(−mean_dist / 4500)`. The decay constant 4500 is a **calibrated heuristic** — set empirically so enrolled faces score 50–53%. The threshold of 40 (≈70% of self-score) provides variance headroom. Neither value is analytically derived.

---

## Code Conventions

### Naming

- Functions: `snake_case` — `load_dataset`, `detect_faces`
- Classes: `PascalCase` — `FaceDetector`, `FaceKNN`
- Constants: `UPPER_CASE` — `DEFAULT_CONFIG_PATH`
- Config keys: `lowercase` — `confidence_threshold`

### Error Handling

Prefer explicit errors over silent defaults.

```python
# Correct
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config not found: {config_path}")

# Wrong — user doesn't know their path failed
if not os.path.exists(config_path):
    config_path = "default.yaml"
```

Silent failures (wrong output with no error raised) are the highest-severity issue class in this codebase. The three load-time guards in `dataset.py` exist specifically to prevent them.

### Docstrings

Required on all public functions.

```python
def detect_faces(frame, cascade, scale_factor=1.3):
    """
    Detect faces in a frame using Haar Cascade.

    Args:
        frame: Grayscale image (numpy array, ndim=2)
        cascade: OpenCV CascadeClassifier instance
        scale_factor: Image pyramid reduction factor (default: 1.3)

    Returns:
        List of (x, y, w, h) bounding boxes. Empty list if none detected.

    Raises:
        ValueError: If frame is not 2D grayscale
    """
```

### Calibrated vs Empirical Values

Any value set to reproduce a target result — rather than derived from data — must be labeled as calibrated in the code, docstring, and any referencing documentation. Example: decay constant 4500 in `model.py::confidence()`.

---

## Contribution Process

### Bug Reports

Include: OS, Python version, package version, expected vs actual behavior, full stack trace, and exact reproduction steps.

### Feature Requests

Evaluated against four criteria: stays within CPU-only constraint, implementable in <1 week, benefits multiple use cases, low ongoing maintenance burden. Email before implementing to avoid wasted effort on rejected features.

### Pull Requests

Before submitting: test on at least one OS, update docs if API changes, no hardcoded paths (use config).

```
What: [One-sentence description]
Why: [Problem this solves]
Testing: [How you verified it works]
Breaking changes: [Any API changes]
```

---

## Contact

**Technical questions / feature discussions:** sakshambajaj@email.com  
**Bug reports:** GitHub Issues

---

**Last Updated:** February 2026