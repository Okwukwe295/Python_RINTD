from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MATERIALS = [
    "Alunite", "Andradite", "Buddingtonite", "Dumortierite",
    "Kaolinite_1", "Kaolinite_2", "Muscovite", "Montmorillonite",
    "Nontronite", "Pyrope", "Sphene", "Chalcedony",
]


def main():
    ap = argparse.ArgumentParser()
    ROOT = Path(__file__).resolve().parents[1]
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--pred-dir", default=str(ROOT / "results" / "ml"))
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)

    true_cube = np.load(data_dir / "abundances.npy")
    pred_cube = np.load(pred_dir / "predicted_abundances.npy")

    if true_cube.shape != pred_cube.shape:
        raise ValueError(
            f"Shape mismatch: true {true_cube.shape} vs predicted {pred_cube.shape}"
        )

    true_avg = true_cube.mean(axis=(0, 1))
    pred_avg = pred_cube.mean(axis=(0, 1))

    signed_error = pred_avg - true_avg
    abs_error = np.abs(signed_error)

    scene_avg_rmse = np.sqrt(np.mean(signed_error ** 2))
    scene_avg_mae = np.mean(abs_error)

    df = pd.DataFrame({
        "Mineral": MATERIALS,
        "True abundance": true_avg,
        "Predicted abundance": pred_avg,
        "Signed error": signed_error,
        "Absolute error": abs_error,
    })

    print("\nTRUE vs PREDICTED SCENE-AVERAGE ABUNDANCE")
    print("=" * 78)
    print(
        f"{'Mineral':18s} {'True':>10s} {'Predicted':>12s} "
        f"{'Error':>10s} {'Abs err':>10s}"
    )
    print("-" * 78)

    for _, row in df.iterrows():
        print(
            f"{row['Mineral']:18s} "
            f"{row['True abundance']:10.5f} "
            f"{row['Predicted abundance']:12.5f} "
            f"{row['Signed error']:10.5f} "
            f"{row['Absolute error']:10.5f}"
        )

    print("-" * 78)
    print(f"True abundance sum      : {true_avg.sum():.6f}")
    print(f"Predicted abundance sum : {pred_avg.sum():.6f}")
    print(f"Scene-average RMSE      : {scene_avg_rmse:.6f}")
    print(f"Scene-average MAE       : {scene_avg_mae:.6f}")
    print(
        f"Scene-average RMSE      : {scene_avg_rmse * 100:.3f} percentage points"
    )
    print(
        f"Scene-average MAE       : {scene_avg_mae * 100:.3f} percentage points"
    )

    csv_path = pred_dir / "true_vs_predicted_scene_abundance.csv"
    df.to_csv(csv_path, index=False)

    # Side-by-side bar chart for presentation / report use.
    x = np.arange(len(MATERIALS))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, true_avg * 100, width, label="True")
    ax.bar(x + width / 2, pred_avg * 100, width, label="Predicted")
    ax.set_ylabel("Scene-average abundance (%)")
    ax.set_title("True vs Predicted Mineral Abundance — Tucker + ML")
    ax.set_xticks(x)
    ax.set_xticklabels(MATERIALS, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    png_path = pred_dir / "true_vs_predicted_scene_abundance.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved CSV : {csv_path}")
    print(f"Saved plot: {png_path}")


if __name__ == "__main__":
    main()
