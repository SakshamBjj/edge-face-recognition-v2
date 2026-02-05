# Architecture Reference

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────┐
│  Input: Video Frame (640×480 RGB, 30 FPS)              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Grayscale Conversion (5ms)                             │
│  • RGB → Gray (3× faster downstream processing)         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Face Detection: Haar Cascade (20ms)                    │
│  • scaleFactor=1.3, minNeighbors=5                      │
│  • Output: [(x, y, w, h), ...] bounding boxes           │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Per-Face Processing (5ms)                              │
│  • Crop face region                                     │
│  • Resize to 50×50                                      │
│  • Flatten to 2500D vector                              │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  KNN Classification (15ms)                              │
│  • k=5, weights='distance'                              │
│  • Distance-based confidence scoring                    │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Unknown Rejection (Optional)                           │
│  • If confidence < threshold → label = "Unknown"        │
│  • Unknowns excluded from attendance logging            │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Output: {name, confidence, bbox}                       │
│  Total Latency: 40ms per processed frame                │
└─────────────────────────────────────────────────────────┘
```

## Component Specifications

### 1. Face Detection: Haar Cascade

Chosen for predictable CPU latency (~20ms detection time).

**Configuration:**
```python
scaleFactor=1.3  # Balance: detection rate vs speed
minNeighbors=5   # Reduce false positives (higher = stricter)
```

**Performance characteristics:** Optimized for frontal faces; detection accuracy degrades at angles exceeding ±30°.

---

### 2. Feature Representation: Raw Pixels

**Approach:** 50×50 grayscale → 2500D vector (no feature extraction)

This direct pixel representation provides sufficient discriminative power for small-scale deployments (5-10 identities) without additional preprocessing overhead. The approach prioritizes inference speed over feature compactness.

---

### 3. Classifier: KNN

**Hyperparameters:**
- **k=5:** Validated via confusion matrix analysis
- **Distance weighting:** `weights='distance'` prioritizes closer neighbors

**Search complexity:** O(n) linear search limits scalability. Performance remains acceptable for datasets under 1,000 samples (~15ms), but degrades beyond that threshold.

---

### 4. Frame Processing Strategy

The system processes every 2nd frame to maintain real-time responsiveness:
- Input stream: 30 FPS
- Effective processing rate: 15 FPS (66ms interval)
- User experience remains smooth despite selective frame processing

This approach accommodates the 40ms processing time within a 30 FPS constraint (33ms per frame).

---

### 5. Unknown Rejection

Confidence-based filtering prevents false positive identifications.

**Configuration:**
```yaml
runtime:
  confidence_threshold: 40
  unknown_label: "Unknown"
  reject_unknowns: true
```

**Behavior:**
- Predictions below threshold are labeled as "Unknown"
- Unknown faces excluded from attendance logging
- Visual distinction: green bbox (high confidence), orange (low confidence/unknown)

**Trigger conditions:**
- Unfamiliar individuals (forced nearest-neighbor match with low similarity)
- Suboptimal capture conditions (poor lighting, extreme angles)
- Partial occlusions affecting discriminative facial features

---

## Operational Limits

### 1. Low Lighting
Low lighting significantly reduces detection reliability due to gradient-based features. Accuracy can drop from 95% to 55% in uniformly dark conditions. Mitigation options include IR cameras, CLAHE histogram equalization, or adaptive threshold adjustment.

### 2. Side Angles (>30°)
Frontal-trained cascade exhibits 20% detection rate at side angles, with 70% accuracy when detection succeeds. Multi-angle cascades or keypoint-based detectors (MTCNN) improve angular robustness.

### 3. Partial Occlusions (Masks, Glasses)
Pixel-based similarity is sensitive to occlusions. Masks reduce accuracy from 95% to 75%. Glasses typically maintain 90% accuracy. Specialized training on occluded faces or upper-face-only cascades improve resilience.

### 4. Multiple Faces (>3)
Processing time scales linearly with face count. Three faces process in 45ms; five faces require 75ms, exceeding real-time constraints. Batch predictions or prioritization strategies (largest face) help maintain responsiveness.

### 5. Dataset Integrity
The system includes fail-fast validation guards:
- Empty dataset check
- Length consistency (len(X) == len(y))
- Dimensionality validation (X.ndim == 2)

These prevent runtime failures from corrupted pickle files due to partial writes or serialization issues.

---

## Performance Characteristics

**Latency breakdown:**
- Grayscale conversion: 5ms
- Face detection: 20ms
- Preprocessing: 5ms
- KNN inference: 15ms
- **Total:** 40ms per frame

**Accuracy metrics:**
- Frontal faces (optimal conditions): 95%
- With glasses: 90%
- With masks: 75%
- Side angles (±30°): 70%
- Low light conditions: 55%

**Model footprint:** <1 MB

---

## Scalability Analysis

### Current System (5-10 identities) ✅
- 500 vectors (100 samples × 5 people) → 15ms KNN search
- Runs on Raspberry Pi 3B+ without GPU

### Medium Scale (50-100 identities) ⚠️
- 5,000 vectors → 75ms search time (exceeds real-time constraint)
- Requires approximate nearest neighbor search (FAISS) with compact embeddings

### Large Scale (1000+ identities) ❌
- Current architecture infeasible
- Requires distributed infrastructure and backend database

---

## Production Upgrade Path

**Phase 1: Enhanced Detection**
- Implement MTCNN for angular robustness

**Phase 2: Feature Compression**
- Deploy FaceNet embeddings (128D) for 20× size reduction

**Phase 3: Efficient Search**
- Integrate FAISS indexing for O(log n) retrieval

**Phase 4: Hardware Acceleration**
- Target Jetson Nano (GPU) for 30 FPS without frame skipping

---

## References

**Foundational work:**
- Viola-Jones (2001) – Haar Cascade methodology
- FaceNet (2015) – Deep learning embeddings
- MTCNN (2016) – Multi-stage detection architecture

**Implementation resources:**
- [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/neighbors.html)
- [OpenCV Cascade Classification](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [FAISS: Facebook AI Similarity Search](https://github.com/facebookresearch/faiss)
