# Cuhk-mini-CLIP

This project has **two independent tracks**:

### Track A — CUHK Building Name Matching (Simple Task)：
- Build a minimal image–text matching system for CUHK buildings, where each caption contains **only the building name**.
- **Goal:** create a clear and working starting system that links building appearance to building identity.

### Track B — CLIP-style Trade-off (Main Task):
- use the flickr30k as dataset
- Run systematic CLIP-style representation learning experiments by varying key factors (vision encoder, text encoder, temperature, embedding size).
- **Goal:** identify model settings that best balance **Efficiency, Accuracy, and Robustness**

---

## Track B 
### Image encoders:
#### Design choices for SmallCNN

- **4 blocks (not too shallow, not too deep)**: \
   2–3 blocks are often under-capacity; very deep CNNs lose the “lightweight” advantage.
   4 blocks take a mid point.

- **Conv3×3 + BN + ReLU + Pool**: \
   A stable combination: computationally efficient and fast and lightweight(pool in each layer)

- **MLP projection to shared embedding space**: \
   Keeps the same CLIP-style interface as other encoders: output `(B, D)`.

- **Why this matters**: \
   SmallCNN is not “randomly simple.”
   It is a deliberate design to represent the lightweight side of the trade-off space while still preserving representation power.

#### Design choices for ResNet-18 and ResNet-50

- **Why not use pretrained weights?**: \
  The comparison of all three image encoders (SmallCNN / ResNet-18 / ResNet-50) must be fair — SmallCNN has no pretrained option, so neither should the others.

- **Why keep ResNet-18 and ResNet50 when we already have SmallCNN?**: \
  SmallCNN is the lightweight option. \
  ResNet-18 is the middle option — deep enough to learn richer features through resnet, but small enough to train on ~30K images without too much overfitting.  \
  ResNet-50 is the deepest option to learn richest features, but probable to overfit.

### Text encoders:
#### Design choices for TextCNN

- **Why kernel sizes (2, 3, 4, 5)?**: \
  2-gram catches short phrases ("red car", "young woman", which is quite common). \
  3–5-gram catches longer descriptive phrases common in Flickr30k captions. \
  Using all four at the same time lets the model recognize patterns at multiple scales.

- **Why max-pool over time (not average-pool)?**: \
  Max-pool extracts the strongest signal from each filter regardless of where it appears in the sentence.

- **Why word_dim is set to be 128?**: \
  128 is enough since there are not many words in captions. \
  If word_dim is 256/512, then the benefit would be limited however parameters would be far more.

# Testing

```bash
python -m pytest tests/ -v
