from pathlib import Path
import numpy as np
import tensorly as tl
from tensorly.decomposition import non_negative_tucker_hals

tl.set_backend("numpy")

# ---------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
TUCKER_DIR = ROOT / "tucker"

TUCKER_DIR.mkdir(parents=True, exist_ok=True)

K = 12
RANK_OFFSET = 2

# Marcus uses r3 = K + rank_offset
RANK = K + RANK_OFFSET   # 14


# ---------------------------------------------------------
# 1. LOAD CUPRITE HSI
# ---------------------------------------------------------
hsi = np.load(DATA_DIR / "hsi.npy").astype(np.float64)

print("\n[1] Loaded HSI")
print("    HSI shape:", hsi.shape)

H, W, B = hsi.shape

print(f"    Height: {H}")
print(f"    Width : {W}")
print(f"    Bands : {B}")


# ---------------------------------------------------------
# 2. TAKE A SMALL BURN-IN REGION
# ---------------------------------------------------------
# Marcus initializes Tucker using several burn-in scenes.
# For this extraction test, use the first 64 rows.
burnin_rows = min(64, H)

X_burn = hsi[:burnin_rows, :, :]

print("\n[2] Burn-in tensor")
print("    Shape:", X_burn.shape)


# ---------------------------------------------------------
# 3. NON-NEGATIVE TUCKER DECOMPOSITION
# ---------------------------------------------------------
# Marcus uses rank [r3, r3, r3]
ranks = [RANK, RANK, RANK]

print("\n[3] Running non-negative Tucker")
print("    Tucker ranks:", ranks)

tucker = non_negative_tucker_hals(
    tl.tensor(X_burn),
    rank=ranks,
    algorithm="fista",
    n_iter_max=30,
    tol=1e-5,
    init="random",
    random_state=42,
)

G = np.asarray(tucker.core)
U0 = np.asarray(tucker.factors[0])
U1 = np.asarray(tucker.factors[1])
U3 = np.asarray(tucker.factors[2])

print("\n[4] Tucker output")
print("    Core G :", G.shape)
print("    U0     :", U0.shape)
print("    U1     :", U1.shape)
print("    U3     :", U3.shape)


# ---------------------------------------------------------
# 4. CREATE MARCUS'S REDUCED SPECTRAL REPRESENTATION
# ---------------------------------------------------------
# Original HSI:
# H x W x B
#
# Flatten pixels:
# X_flat = B x N

X_flat = hsi.reshape(-1, B).T

print("\n[5] Original flattened spectra")
print("    X_flat:", X_flat.shape)

# Marcus's exact reduction:
#
# Z = pinv(U3) @ X_flat

U3_pinv = np.linalg.pinv(U3)
Z = U3_pinv @ X_flat

print("\n[6] Reduced spectral representation")
print("    U3      :", U3.shape)
print("    pinv(U3):", U3_pinv.shape)
print("    X_flat  :", X_flat.shape)
print("    Z       :", Z.shape)

print(
    f"\n    Each pixel has been reduced "
    f"from {B} bands -> {Z.shape[0]} Tucker coefficients."
)


# ---------------------------------------------------------
# 5. RECONSTRUCT SPECTRA FROM REDUCED REPRESENTATION
# ---------------------------------------------------------
X_projected = U3 @ Z

projection_error = (
    np.linalg.norm(X_flat - X_projected)
    /
    np.linalg.norm(X_flat)
)

print("\n[7] Projection test")
print(f"    Relative projection error: {projection_error:.6f}")


# ---------------------------------------------------------
# 6. SAVE OUTPUTS
# ---------------------------------------------------------
np.save(TUCKER_DIR / "spectral_basis_U3.npy", U3)

# Save in pixel-major format for ML:
# N x 14 rather than 14 x N
np.save(TUCKER_DIR / "reduced_spectra_Z.npy", Z.T)

print("\n[8] Saved")
print("    spectral_basis_U3.npy :", U3.shape)
print("    reduced_spectra_Z.npy :", Z.T.shape)

print("\nDONE.")