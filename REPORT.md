# REPORT – Task 4: a better leaf classifier

## 1. Setup
- **Baseline:** the Task 2 model – pretrained `mobilenet_v3_small`, last layer replaced
  with 4 outputs, the whole network fine-tuned, Adam (lr 1e-3), 15 epochs, batch 32,
  `CrossEntropyLoss`, no augmentation, ImageNet mean/std normalization.
- **Data:**
  - `leaves_new/train` – our group's photos, used for training in every variant
    (variant 8 adds `leaves_old/train`)
  - `leaves_new/val` – our group's validation photos (120 images); used only to pick the
    best epoch (variant 7)
  - `leaves_old/val` – last years' photos, other students and phones (120 images);
    **the main, camera-closer metric** – never used for any choice during training
- **Fair comparison:** one change at a time against the baseline, then combinations of the
  best changes. Same training images and a fixed seed (`torch.manual_seed`) in every run.
  Every row was run with **seeds 0, 1 and 2** (36 runs in total).
- **Noise:** one validation image = 1/120 = 0.0083. The baseline alone differs by
  8 images (0.842–0.908) on `leaves_old/val` between seeds. Runs with the same seed give
  identical results, so the runs are reproducible.
- **Agent:** the agent (Claude Code) wrote `experiments.py` from my Task 2/3 code with the
  instruction to keep the simple, readable code and change only what each variant needs.
  I chose the variants and the seeds and judged the results. Everything ran locally on the CPU, 
  taking about 1 minute per run (30 epochs took about 2 minutes)



## 2. Results
Mean over seeds 0, 1, 2 (min–max in brackets). Δ = difference of the mean against the
baseline, in images out of 120.

| # | variant | leaves_new/val | leaves_old/val | Δ old |
|---|---------|----------------|----------------|-------|
| 0 | baseline (Task 2 model) | 0.847 (0.833–0.858) | 0.881 (0.842–0.908) | – |
| 1 | geometric augmentation | 0.806 (0.733–0.850) | 0.856 (0.783–0.917) | −3.0 |
| 2 | colour augmentation (ColorJitter) | 0.844 (0.833–0.858) | 0.875 (0.850–0.900) | −0.7 |
| 3 | frozen network, only last layer trained | 0.725 (0.708–0.742) | 0.758 (0.742–0.767) | −14.7 |
| 4 | cosine LR, 30 epochs | 0.864 (0.842–0.892) | 0.908 (0.892–0.917) | +3.3 |
| 5 | label smoothing 0.1 | 0.844 (0.833–0.858) | 0.908 (0.900–0.917) | +3.3 |
| 6 | AdamW, weight decay 1e-2 | 0.828 (0.808–0.842) | 0.886 (0.842–0.908) | +0.7 |
| 7 | weights of the best epoch | 0.858* (0.850–0.867) | 0.908 (0.883–0.933) | +3.3 |
| 8 | + last years' training photos | 0.828 (0.808–0.842) | 0.886 (0.867–0.917) | +0.7 |
| C1 | 4 + 5 + 7 | 0.867* (0.850–0.875) | 0.911 (0.883–0.925) | +3.7 |
| C2 | 4 + 5 + 7 + 8 | 0.875* (0.867–0.883) | 0.917 (0.892–0.942) | +4.3 |
| C3 | 1 + 4 + 5 + 7 | 0.889* (0.883–0.900) | 0.917 (0.900–0.942) | +4.3 |

\* biased upward – the best epoch was chosen on this same set, so only `leaves_old/val`
is a fair comparison for these rows.


## 3. What each change brought
- **Cosine LR + 30 epochs (4), label smoothing (5), best epoch (7):** each about
  +3 images on `leaves_old/val`, and the minimum over seeds of each is above the baseline
  minimum. All three address the same weakness of the baseline. Its accuracy jumps a lot
  between epochs (seed 0: 0.933 in epoch 6, 0.775 in epoch 11, 0.892 at the end).
  The cosine schedule makes the steps smaller towards the end, label smoothing makes the
  network less overconfident, and the best epoch keeps a good point instead of the last one.
- **Combinations:** C1–C3 add up to about +4 images on average, and every single run of
  C1–C3 is above the baseline minimum. C2 and C3 tie at 0.917 on `leaves_old/val`.
- **Geometric augmentation (1):** worse alone (−3 images, one bad seed: 0.783), but helps
  inside C3. With augmentation the task is harder and 15 epochs are not
  enough, while C3 has 30 epochs and keeps the best epoch. With 3 seeds this is a guess,
  not a proven result.
- **Training loss with label smoothing:** in all runs with label smoothing (5, C1–C3) the
  training loss does not go to zero but stops at about 0.36 (C3 about 0.39, because the
  augmented images are harder). This is expected: the target for the correct class is
  0.925 instead of 1.0 and 0.025 for each other class, so the lowest possible
  cross-entropy is the entropy of this target.
  Without smoothing the training loss falls to 0.02–0.05 in the baseline and to about
  0.005 with the cosine schedule (variant 4) – the network learns the training photos
  almost perfectly by heart. Label smoothing stops this, and its accuracy on
  `leaves_old/val` is also the most stable over the seeds (0.900–0.917).

## 4. Is the difference larger than the noise?
- One image is 0.0083, and the baseline alone differs by **8 images** between seeds on
  `leaves_old/val`.
- **Only the frozen network (3) is clearly outside the noise** – it loses about 15 images
  and its range does not overlap with the baseline.
- The gains of 4, 5, 7 (+3 images) and of the combinations (+4 images) are **smaller than
  the baseline's spread**, so with 3 seeds they are a consistent hint, not a proof.
  They all go in the same direction, and their minimums are above the
  baseline minimum.
- C1, C2 and C3 cannot be separated from each other (differences below 1 image).

## 5. What did not help
- **Frozen network (3):** clearly worse (−15 images). The ImageNet features alone are not
  enough for leaves; the whole network has to be fine-tuned. Only the weights were
  frozen. In `model.train()` the BatchNorm running statistics still update and dropout
  is still on.
- **AdamW (6), colour augmentation (2):** no measurable effect (below 1 image).
- **More data – last years' training photos (8):** no effect alone (+0.7 images), and only
  +0.6 images inside C2 compared to C1. Small gain on `leaves_old/val` may only mean the photos are alike, not that
  the model is more robust.
- **Geometric augmentation alone (1):** slightly worse with 15 epochs.

## 6. The model I would put on the phone
**C3 – geometric augmentation + cosine LR with 30 epochs + label smoothing + best epoch**,
seed 0 (epoch 9: 0.883 on `leaves_new/val`, 0.900 on `leaves_old/val`).

Why:
- C2 and C3 are equal on `leaves_old/val`, but C3 is trained with random crops, flips and
  rotations – exactly what changes in the live camera (position, size and angle of the leaf).
- Seed 0 was chosen in advance, not the best seed on `leaves_old/val`, so that number
  stays a fair estimate.

**Tried on the phone:**  I added the model to my own page next to the original one (`leaf_model_c3.onnx`,
opened with `?model=c3`; without the parameter the page and the self-test still use the Task 2 model).
Samsung Galaxy A54, Chrome, one photo of each species:

| leaf | Task 2 model | C3 model |
|------|--------------|----------|
| apple | cherry (wrong) | **apple 82 %** (cherry 16 %) |
| cherry | cherry | cherry 97 % |
| chestnut | chestnut | chestnut 100 % |
| maple | maple | maple 84 % |

- The apple, which the Task 2 model always called cherry (all 8 validation apples in the
  confusion matrix), is now recognised. Apple and cherry stay the closest pair.
- On `check_image.jpg` (an apple) C3 still says cherry, but with 0.76 instead of 0.99999,
  and apple gets 0.20 instead of 5e-8. The new model is much less overconfident.

## 7. Conclusion
The biggest problem of the Task 2 model was its instability between epochs and its
apple → cherry confusion. Stabilising the training (cosine LR, label smoothing, best epoch)
and adding geometric augmentation improved the accuracy on last years' photos from 0.881
to 0.917 on average and the model now recognises apples on the phone, but with 120
validation images and 3 seeds the improvement is still within the noise of the baseline.