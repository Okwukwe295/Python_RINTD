from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CSV_PATH = Path(
    r"C:\Users\okwuk\OneDrive\Desktop\MASTERS COURSEWORK\SEMESTER 2"
    r"\Python_RINTD\results\ml\ml_conveyor_results.csv"
)

MOVING_AVG_WINDOW = 20 


print("Looking for CSV at:")
print(CSV_PATH)

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

OUTPUT_DIR = CSV_PATH.parent

df = pd.read_csv(CSV_PATH)

print("\nCSV loaded successfully.")
print("Rows:", len(df))
print("Columns:", list(df.columns))

df = df.sort_values("scene").reset_index(drop=True)

df["reconstruction_ma"] = (
    df["reconstruction_rmse"]
    .rolling(MOVING_AVG_WINDOW, min_periods=1)
    .mean()
)

df["abundance_ma"] = (
    df["abundance_rmse"]
    .rolling(MOVING_AVG_WINDOW, min_periods=1)
    .mean()
)


# Reconstruction RMSE
recon_path = OUTPUT_DIR / "reconstruction_error.png"

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["scene"],
    df["reconstruction_rmse"],
    linewidth=0.8,
    alpha=0.45,
    label="Per scene"
)

ax.plot(
    df["scene"],
    df["reconstruction_ma"],
    linewidth=2.5,
    label=f"{MOVING_AVG_WINDOW}-scene moving average"
)

ax.set_xlabel("Scene")
ax.set_ylabel("Reconstruction RMSE")
ax.set_title("Reconstruction Error per Scene")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(recon_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", recon_path)


# Abundance RMSE
abundance_path = OUTPUT_DIR / "abundance_rmse.png"

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["scene"],
    df["abundance_rmse"],
    linewidth=0.8,
    alpha=0.45,
    label="Per scene"
)

ax.plot(
    df["scene"],
    df["abundance_ma"],
    linewidth=2.5,
    label=f"{MOVING_AVG_WINDOW}-scene moving average"
)

ax.set_xlabel("Scene")
ax.set_ylabel("Abundance RMSE")
ax.set_title("Abundance RMSE per Scene")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(abundance_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("Saved:", abundance_path)

print("\nDone.")