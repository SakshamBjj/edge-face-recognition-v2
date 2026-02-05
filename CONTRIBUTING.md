# Contributing Guide

This repository is maintained as a reference implementation of a low-latency CPU face recognition pipeline.

External contributions are minimal by design, but the development philosophy and extension points are documented for clarity.

---

## Design Principles

### 1. Deterministic Latency Over Maximum Accuracy

The system prioritizes deterministic latency over maximum accuracy. All components are chosen to guarantee real-time performance on CPU-only hardware.

This constraint drives the classical ML approach (Haar Cascade + KNN) rather than deep learning alternatives that require GPU acceleration or quantization for comparable latency.

---

### 2. Direct Pixel Features

The 50×50 grayscale pixel representation (2500D vectors) eliminates preprocessing overhead while providing sufficient discriminative power for small-scale deployments. This simplicity facilitates debugging and maintains a straightforward processing pipeline.

Feature extraction layers (HOG, SIFT) can add 10ms overhead for marginal accuracy gains—an unfavorable tradeoff given the target latency constraints.

---

### 3. Precision-First Unknown Rejection

The system implements confidence-based filtering to prioritize precision over recall:

```yaml
runtime:
  confidence_threshold: 40      # Tunable
  unknown_label: "Unknown"
  reject_unknowns: true
```

For attendance applications, false positives (incorrect identification) carry higher costs than false negatives (missed detection). Low-confidence predictions are rejected to prevent misidentification.

---

### 4. Fail-Fast Validation

Dataset integrity is validated at load time with three guards:
1. Empty dataset check
2. Length consistency (len(X) == len(y))
3. Dimensionality validation (X.ndim == 2)

This approach surfaces corruption errors immediately rather than during training with cryptic stack traces.

---

## Development Workflow

### Local Setup

```bash
git clone https://github.com/SakshamBjj/edge-face-recognition-v2.git
cd edge-face-recognition-v2
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e .
```

### Verification Steps

Testing is primarily manual due to webcam interaction requirements.

```bash
# Collect test data
edge-face collect --name TestPerson

# Run inference
edge-face run

# Verify attendance logging (press 'o')
cat attendance/$(date +%Y-%m-%d).csv
```

---

## Known Constraints

### 1. Scalability
KNN search exhibits O(n) complexity, limiting performance beyond ~100 identities (inference >100ms). Migration to approximate nearest neighbor search (FAISS) with compact embeddings is required for larger deployments.

### 2. Angular Robustness
Haar cascade detection is optimized for frontal faces. Side angles exceeding ±30° exhibit 20% detection rates. Keypoint-based detectors (MTCNN) provide improved angular coverage.

### 3. Illumination Sensitivity
Gradient-based features degrade in low-contrast conditions, with accuracy dropping 40% in uniform darkness. IR cameras or adaptive histogram equalization mitigate this constraint.

### 4. Anti-Spoofing
The system does not implement liveness detection. Production deployments require additional safeguards against photo/video replay attacks.

---

## Architecture Rationale

### Package Structure (src/edge_face/)
- Pip-installable distribution
- CLI entrypoint (`edge-face` command)
- Modular components: config, detector, dataset, model, pipeline

### Configuration (configs/default.yaml)
- Centralized parameter management
- No hardcoded constants in source code
- Parameter tuning without code modification

### Standalone Scripts (scripts/)
- Zero-installation data collection for users without package setup
- Reads identical YAML configuration
- Functionally equivalent to CLI interface

---

## Potential Extensions

### Phase 1: Enhanced Detection
Replace Haar with MTCNN for 70% accuracy at ±30° angles (vs 20% current).

### Phase 2: Feature Compression
Implement FaceNet embeddings (128D) for 20× vector size reduction and O(log n) FAISS search.

### Phase 3: Hardware Acceleration
Deploy on Jetson Nano (GPU) for 30 FPS without frame skipping.

---

**Last updated:** December 2025
