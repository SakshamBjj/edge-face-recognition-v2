# Architecture Reference

## System Overview

Real-time CPU face recognition pipeline. Every component chosen to guarantee sub-100ms deterministic latency without GPU acceleration.

**Core constraint:** 40ms processing budget per frame at 30 FPS input → frame skipping required for real-time UX.

---

## Processing Pipeline

```
Camera Input (640×480 RGB @ 30 FPS)
 ↓
Grayscale Conversion (~5 ms)
 → Detection only — Haar requires grayscale; saves ~15ms downstream
 ↓
Haar Cascade Detection (~20 ms)
 → scaleFactor=1.3, minNeighbors=5
 → Output: bounding boxes [(x, y, w, h), ...]
 ↓
Per-Face Processing (~5 ms)
 → Crop from COLOR (BGR) frame — not grayscale
 → Resize to 50×50 pixels
 → Flatten → 7,500D vector (50 × 50 × 3 channels)
 ↓
KNN Classification (~15 ms for ≤500 samples)
 → k=5, weights='distance', Euclidean distance
 ↓
Confidence Scoring
 → score = 100 × exp(−mean_dist / 4500)
 → Heuristic — decay constant 4500 calibrated empirically, not derived
 → Self-confidence (enrolled face): 50–53%
 → Hard negatives (family members): 28–35%
 → Threshold: 40 (percent, configurable)
 ↓
Unknown Rejection
 → confidence < 40 → label = "Unknown", excluded from attendance log
 ↓
Output: {name, confidence, bbox}
Total: ~40 ms per processed frame
```

---

## Component Decisions

### Detection: Haar Cascade

**Alternatives considered:** HOG+SVM (2× slower), MTCNN (10× slower on CPU), MediaPipe (requires TFLite).

**Chosen:** Haar Cascade — only option meeting the 20ms detection budget on CPU.

**Parameter rationale:**

```python
scaleFactor=1.3   # Empirically chosen: smaller → more detections, slower;
                  # larger → faster, misses faces. 1.3 = balance point.
minNeighbors=5    # Controls false positive rate.
                  # 5 = good precision/recall for indoor single-subject scenarios.
```

**Known limit:** Side profiles (>30°) have <20% detection rate. Accepted trade-off for latency budget.

---

### Feature Representation: Raw Pixels

**Alternatives considered:** HOG (+10ms overhead), LBP (more lighting-robust but slower), CNN embeddings (GPU required).

**Chosen:** 50×50 color (BGR) crop, flattened to 7,500D vector.

**Rationale:** Zero extraction overhead; sufficient discriminative power for <50 identities in controlled environments. Easy to debug — pixel values are directly inspectable.

**Trade-off accepted:** Less robust to lighting and pose variation than engineered features.

---

### Classification: KNN

**Alternatives considered:** SVM (requires training step, no confidence scores), Random Forest (2× slower inference), neural net (overkill for 7,500D pixel features).

**Chosen:** KNN (k=5, distance-weighted).

**Hyperparameter rationale:**

```python
k=5               # k=3: too sensitive to outliers
                  # k=7: no accuracy gain, slower
weights='distance' # Closer neighbors weighted higher;
                   # improves confidence score quality
```

**Scaling limit:** O(n) brute-force search. Acceptable for n < 500 samples (~15ms). Becomes bottleneck above ~5,000 samples.

---

### Confidence Scoring

**Problem:** KNN always returns a nearest neighbor, even for unknown faces. Raw distances (~1,600–1,800 for 7,500D vectors) make linear subtraction (`100 - distance`) always negative → always 0.

**Fix:** Exponential decay over mean k-neighbor distance.

```python
score = 100.0 * np.exp(-mean_dist / 4500.0)
```

**Calibration:** Decay constant 4500 set so enrolled faces score 50–53%. Threshold at 40% (≈70% of self-score) provides variance headroom while rejecting visually similar non-enrolled faces (28–35%).

This is a **heuristic** — not a calibrated probability. Label it as such in any external communication.

---

### Frame Skipping

**Problem:** 40ms processing exceeds 33ms frame budget → processing every frame causes visual stuttering.

**Solution:** Process every 2nd frame. Display last overlay on skipped frames.

| Metric | All Frames | Every 2nd Frame |
|--------|------------|-----------------|
| Processing load | 121% (overloaded) | 60% |
| Effective FPS | ~12 (stutters) | ~15 (smooth) |
| Detection latency | 33ms | 66ms |

**Why it works:** Human perception threshold for visual lag is ~50ms. 66ms frame interval is imperceptible.

---

### Unknown Rejection

**Problem:** False positive in attendance = permanent incorrect record. False negative = user retries. Errors are asymmetric.

**Solution:** Confidence threshold — predictions below 40% labeled "Unknown" and excluded from CSV logging.

```yaml
runtime:
  confidence_threshold: 40   # Percent (0–100). Higher = stricter.
  unknown_label: "Unknown"
  reject_unknowns: true
```

**Visual feedback:** Green bbox (confidence ≥ 40), orange bbox + "Unknown" label (confidence < 40).

---

## Latency Budget

| Component | Latency | Notes |
|-----------|---------|-------|
| Grayscale conversion | ~5 ms | Constant |
| Haar Cascade detection | ~20 ms | Scales with image size, not face count (up to ~3 faces) |
| Crop + resize + flatten | ~3 ms | Per face |
| KNN search | ~15 ms | Scales linearly with sample count |
| Confidence + labeling | ~2 ms | Per face |
| **Total (single face)** | **~40 ms** | Meets real-time constraint with frame skipping |

**Multi-face degradation:** Each additional face adds ~5ms. Beyond 5 faces, latency exceeds 60ms (sub-real-time). Current behavior: process only the largest face.

---

## Scaling Limits

| # Identities | # Samples | KNN Search | Total Latency | Real-time? |
|--------------|-----------|------------|---------------|------------|
| 10 | 1,000 | ~10 ms | ~35 ms | ✓ |
| 50 | 5,000 | ~30 ms | ~55 ms | ✓ (borderline) |
| 100 | 10,000 | ~60 ms | ~85 ms | ✗ |
| 500 | 50,000 | ~300 ms | ~325 ms | ✗ |

**Recommendation:** <50 identities for smooth real-time performance.

**Upgrade path:** FAISS approximate nearest neighbors (O(log n)) for 50–500 identities; CNN embeddings (FaceNet/128D) + FAISS for larger datasets. Both require architectural changes outside this project's scope.

---

## Dataset Integrity Guards

Three load-time checks in `dataset.py::load()` prevent runtime failures from corrupt data.

```python
if len(X) == 0:
    raise ValueError("Dataset is empty. Run 'edge-face collect' first.")

if len(X) != len(y):
    raise ValueError(f"Feature count ({len(X)}) != label count ({len(y)})")

if X.ndim != 2:
    raise ValueError(f"Expected 2D array, got {X.ndim}D")
```

**Why load-time:** Surfaces corruption immediately with an actionable message, not as a cryptic KNN failure mid-inference.

---

## Configuration Reference

```yaml
detection:
  scale_factor: 1.3
  min_neighbors: 5
  min_size: [30, 30]

recognition:
  n_neighbors: 5
  weights: "distance"

runtime:
  confidence_threshold: 40    # Percent. Calibrated heuristic — see Confidence Scoring.
  unknown_label: "Unknown"
  reject_unknowns: true
  frame_skip: 2
  camera_id: 0

output:
  attendance_dir: "attendance"
  bbox_color_known: [0, 255, 0]      # Green
  bbox_color_unknown: [0, 165, 255]  # Orange
```

All runtime parameters are configurable without code changes. No hardcoded constants in source.

---

## Known Gaps

| Gap | Impact | Mitigation |
|-----|--------|------------|
| No liveness detection | Photo/video replay attacks succeed | Acceptable for low-stakes attendance; add blink detection or depth camera for security-critical use |
| No temporal smoothing | Confidence flicker near threshold | Hysteresis (dual thresholds) or EMA smoothing — deferred |
| No multi-face tracking | Each frame independent | Add SORT/DeepSORT if cross-frame identity continuity needed |
| Raw face storage | GDPR exposure | Delete images post-extraction; encrypt feature vectors at rest |

---

## References

- Viola-Jones (2001): Haar Cascade face detection
- Cover & Hart (1967): K-Nearest Neighbors
- [OpenCV Cascade Classifier](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [scikit-learn KNN](https://scikit-learn.org/stable/modules/neighbors.html)
- [FAISS](https://github.com/facebookresearch/faiss) — upgrade path for large datasets

---

**Last Updated:** February 2026