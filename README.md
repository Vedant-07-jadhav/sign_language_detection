# SIGN LANGUAGE DETECTION

- used the ffmeg and prallel processing using multiprocessing, and ffmpeg comand for cropping and grayscaling the videos 
- done preprocessing and moveda and splited all the datafiles in train and test in data folder along with there labels in txt file, and also .yaml file to get easy detils for any model 

__ Next steps find good model and also try to make the good use of the available data

# date : 22 nd dec 2025 to 4th jan 2026

## Phase 1: Baseline Video Sign Recognition (NSLT/WLASL-100)

### Objective
Build a **from-scratch, end-to-end video classification pipeline** for sign language recognition, focusing on understanding:
- video preprocessing decisions,
- spatial vs temporal modeling,
- common dataset pitfalls in real-world ML.

This phase prioritizes **correctness, debugging, and learning**, not peak accuracy.

---

### Dataset
- **NSLT / WLASL-100 subset**
- Each sample: short video clip corresponding to a single sign
- Labels originally provided as sparse class IDs
- Severe class imbalance and non-class-stratified splits

---

### Video Preprocessing Pipeline
1. Parsed JSON annotations to extract:
   - video paths
   - class labels
   - temporal boundaries
2. Trimmed videos to annotated start–end frames
3. Cropped frames using provided bounding boxes
4. Converted frames to grayscale (motion-focused representation)
5. Uniformly sampled **T = 32 frames per video**
6. Resized frames to **112 × 112**
7. Stored each processed video as a `.pt` tensor of shape: (T, H, W) = (32, 112, 112)


8. Verified preprocessing correctness by:
- visualizing tensors
- checking temporal frame differences
- validating value ranges

---

### Critical Dataset Debugging
During initial experiments, training accuracy remained near random.

Root causes identified and fixed:
- **Class mismatch across splits**  
Train, validation, and test sets did not share identical class sets.
- **Non-contiguous label space**  
Class IDs were sparse and incompatible with `CrossEntropyLoss`.

Fixes applied:
- Computed intersection of classes across all splits
- Filtered all datasets to shared classes only
- Re-mapped class IDs to a contiguous range `[0 … C−1]`
- Added strict sanity assertions before training

This debugging step was essential; no model learned before it.

---

### Baseline Models Evaluated

#### 1. Frame-wise CNN + Temporal Mean Pooling
- Simple 2D CNN trained from scratch
- Frame features averaged across time
- Result: failed to learn meaningful representations

#### 2. CNN + LSTM
- LSTM applied over per-frame features
- Result: marginal improvement
- Diagnosis: weak spatial features collapse temporal modeling

#### 3. CNN + Temporal Convolution (Conv1D over time)
- Local temporal motion modeling
- Result: slight improvement but still limited
- Conclusion: temporal models cannot compensate for weak spatial encoders

---

### Transfer Learning Upgrade (Key Improvement)
To address the spatial bottleneck:

- Replaced custom CNN with **pretrained ResNet-18 (ImageNet)**
- Converted grayscale frames to 3-channel format
- Froze backbone initially to prevent overfitting
- Extracted 512-D frame embeddings
- Applied temporal convolution for motion modeling

Final architecture:




---

### Results (Phase 1)
- Number of classes after filtering: **~40**
- Training accuracy: **~14–15%**
- Validation accuracy: **~6–8%**
- Loss significantly below random baseline
- Clear evidence of learning and early overfitting

While accuracy is modest, the pipeline is:
- technically correct
- debuggable
- extensible

---

### Key Learnings
- Data integrity and label alignment matter more than architecture choice
- Temporal models amplify spatial features; they do not replace them
- Pretraining is essential for small video datasets
- Debugging real datasets is a core ML skill, not an edge case

---

### Next Steps
- Fine-tune upper layers of the ResNet backbone
- Add light, sign-safe data augmentation
- Introduce early stopping and regularization
- Extend to larger WLASL subsets or 3D CNN baselines



