# Oreacle — Tucker + ML Mineral Unmixing Pipeline

This project compares two mineral unmixing approaches for hyperspectral mineral characterisation:

1. **Tucker-reduced ML unmixing**
2. **Tucker + VCA + FCLS baseline**

The goal is to test whether a Tucker-reduced hyperspectral representation can be used effectively for mineral abundance estimation, while comparing the ML approach against a conventional spectral unmixing baseline.

---

## Required Local Data

The large Cuprite dataset files are **not stored in GitHub** because of GitHub file-size limits.

Place the required dataset files inside the `data/` directory:

```text
data/
├── hsi.npy
├── abundances.npy
├── endmembers.npy
└── info.yaml
```

Large and generated files such as `hsi.npy`, `abundances.npy`, and `reduced_spectra_Z.npy` should remain excluded through `.gitignore`.

---

## Project Structure

```text
Python_RINTD/
│
├── data/
│   ├── hsi.npy
│   ├── abundances.npy
│   ├── endmembers.npy
│   └── info.yaml
│
├── tucker/
│   ├── extract_tucker.py
│   └── spectral_basis_U3.npy
│
├── ml_unmixing/
│   ├── train_cuprite_unmixing_reduced.py
│   └── compare_scene_abundances.py
│
├── baseline/
│   ├── sp_conveyor.py
│   └── unmixing.py
│
├── comparison/
│   └── compare_baseline_vs_ml.py
│
├── results/
│   ├── ml/
│   ├── baseline/
│   └── comparison/
│
├── .gitignore
└── README.md
```

---

## Pipeline

The current workflow is:

```text
HSI cube
   ↓
Tucker spectral reduction
188 spectral bands → 14 Tucker coefficients
   ↓
   ├── ML autoencoder → 12 mineral abundances
   │
   └── VCA + FCLS → 12 mineral abundances
   ↓
Performance comparison
```

The Tucker spectral factor `U3` is used to generate the reduced representation:

```text
Z = pinv(U3) @ X
```

where:

- `X` is the original hyperspectral data
- `U3` is the Tucker spectral factor
- `Z` is the reduced spectral representation

For the current Cuprite synthetic dataset:

```text
188 spectral bands → 14 Tucker features
```

---

## Running the Full Pipeline

Run all commands from the **repository root**.

### 1. Extract the Tucker Spectral Representation

```bash
python tucker/extract_tucker.py
```

This performs the Tucker decomposition and generates the spectral basis:

```text
tucker/spectral_basis_U3.npy
```

It also generates the reduced representation of the hyperspectral pixels.

---

### 2. Train the Reduced-Input ML Unmixing Model

```bash
python ml_unmixing/train_cuprite_unmixing_reduced.py
```

The ML model receives the 14 Tucker coefficients for each pixel and predicts the 12 mineral abundances.

The encoder is approximately:

```text
14 Tucker features
      ↓
     64
      ↓
     32
      ↓
12 mineral abundances
```

A Softmax output layer is used so that predicted abundances satisfy:

```text
abundance >= 0
sum(abundances) = 1
```

The decoder reconstructs the original 188-band spectrum from the predicted abundances.

Outputs are stored in:

```text
results/ml/
```

Important outputs include:

```text
predicted_abundances.npy
learned_endmembers.npy
metrics.txt
```

---

### 3. Compare True vs Predicted Mineral Abundances

```bash
python ml_unmixing/compare_scene_abundances.py
```

This compares the known Cuprite synthetic ground-truth abundances against the abundances predicted by the ML model.

Outputs are written to:

```text
results/ml/
```

including:

```text
true_vs_predicted_scene_abundance.csv
true_vs_predicted_scene_abundance.png
```

This provides an easy visual comparison between the true scene composition and the predicted scene composition.

---

### 4. Compare ML Against the Classical Baseline

```bash
python comparison/compare_baseline_vs_ml.py
```

This compares:

```text
Tucker → ML Autoencoder
```

against:

```text
Tucker → VCA → FCLS
```

The classical baseline uses:

- **VCA** for endmember extraction
- **FCLS/SUNSAL** for abundance estimation
- `baseline/unmixing.py` for the constrained unmixing solver

Comparison outputs are stored in:

```text
results/comparison/
```

including:

```text
method_comparison.csv
per_mineral_comparison.csv
heldout_rmse_comparison.png
scene_composition_comparison.png
baseline_predicted_abundances.npy
baseline_estimated_endmembers.npy
```

---

## Current Tucker + ML Results

Using the 14-dimensional Tucker representation:

```text
HSI shape                    : (307, 307, 188)
Spectral basis U3            : (188, 14)
Reduced dimensions           : 188 → 14

Held-out abundance RMSE      : 0.015499
Reconstruction RMSE          : 0.026155
Mean endmember SAD           : 2.392868°
Maximum endmember SAD        : 7.451428°

Whole-scene inference time   : 0.554625 s
Inference throughput         : ~169,933 pixels/s
Device                       : CPU
```

The current result shows that the 188-band hyperspectral input can be reduced to 14 Tucker coefficients while still allowing the ML model to estimate the 12 mineral abundances accurately on the current Cuprite synthetic experiment.

---

## Workflow Summary

```text
1. Place Cuprite dataset files in data/

2. Run Tucker extraction:
   python tucker/extract_tucker.py

3. Train ML unmixing model:
   python ml_unmixing/train_cuprite_unmixing_reduced.py

4. Generate true-vs-predicted abundance comparison:
   python ml_unmixing/compare_scene_abundances.py

5. Compare ML against VCA + FCLS:
   python comparison/compare_baseline_vs_ml.py

6. View comparison results in:
   results/comparison/
```

---

## Notes

- Large Cuprite dataset files are kept locally and are not pushed to GitHub.
- Generated result folders do not need to be committed unless the outputs are specifically required for reporting or presentation.
- Run the scripts from the repository root so that project-relative paths resolve correctly.
- The ML and classical baseline comparison use the same Cuprite synthetic data so abundance estimation performance can be compared consistently.
- `baseline/unmixing.py` contains the SUNSAL/FCLS solver used by the classical baseline.
