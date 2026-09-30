from pathlib import Path
import argparse
import importlib.util
import time
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

MATERIALS = [
    "Alunite", "Andradite", "Buddingtonite", "Dumortierite",
    "Kaolinite_1", "Kaolinite_2", "Muscovite", "Montmorillonite",
    "Nontronite", "Pyrope", "Sphene", "Chalcedony",
]
EPS = 1e-12


def load_sunsal(path: Path):
    spec = importlib.util.spec_from_file_location("unmixing", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sunsal


def marcus_vca(Y: np.ndarray, p: int, seed: int = 0):
    """Exact VCA helper used in Marcus's sp_conveyor.py.

    Y is reduced data with shape (R, N). Returns selected columns and indices.
    """
    rng = np.random.default_rng(seed)
    _, N = Y.shape
    U_, _, _ = np.linalg.svd(Y, full_matrices=False)
    Ud = U_[:, :p]
    Xp = Ud.T @ Y
    A = np.zeros((p, p))
    A[-1, 0] = 1.0
    idxs = np.zeros(p, dtype=int)

    for i in range(p):
        w = rng.standard_normal(p)
        f = (np.eye(p) - A @ np.linalg.pinv(A)) @ w
        nf = np.linalg.norm(f)
        f = f / nf if nf > 1e-10 else w
        j = int(np.argmax(np.abs(f @ Xp)))
        A[:, i] = Xp[:, j]
        idxs[i] = j

    return Y[:, idxs], idxs


def sam_deg(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 90.0
    c = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def hungarian_sad(E_true_bk: np.ndarray, E_est_bk: np.ndarray):
    K = E_true_bk.shape[1]
    cost = np.array([
        [sam_deg(E_true_bk[:, i], E_est_bk[:, j]) for j in range(K)]
        for i in range(K)
    ])
    rows, cols = linear_sum_assignment(cost)
    perm = dict(zip(rows.tolist(), cols.tolist()))
    sads = np.array([cost[i, perm[i]] for i in range(K)])
    return sads, perm


def align_abundances(A_est_kn: np.ndarray, perm: dict, K: int) -> np.ndarray:
    aligned = np.zeros_like(A_est_kn)
    for true_idx in range(K):
        aligned[true_idx] = A_est_kn[perm[true_idx]]
    return aligned


def rowwise_sad(E_true_kb: np.ndarray, E_est_kb: np.ndarray) -> np.ndarray:
    dots = np.sum(E_true_kb * E_est_kb, axis=1)
    denom = np.linalg.norm(E_true_kb, axis=1) * np.linalg.norm(E_est_kb, axis=1)
    cosang = np.clip(dots / np.maximum(denom, EPS), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def parse_ml_metrics(metrics_path: Path):
    out = {}
    if not metrics_path.exists():
        return out
    txt = metrics_path.read_text(encoding="utf-8")
    patterns = {
        "reported_heldout_rmse": r"Held-out abundance RMSE:\s*([0-9.eE+-]+)",
        "reported_mean_sad": r"Mean endmember SAD \(deg\):\s*([0-9.eE+-]+)",
        "reported_max_sad": r"Max endmember SAD \(deg\):\s*([0-9.eE+-]+)",
        "reported_inference_seconds": r"Whole-scene inference seconds:\s*([0-9.eE+-]+)",
        "reported_pixels_per_second": r"Pixels per second:\s*([0-9.eE+-]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, txt)
        if m:
            out[key] = float(m.group(1))
    return out


def main():
    ROOT = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(
        description="Compare Marcus-style Tucker->VCA->FCLS against Tucker->ML on Cuprite synthetic."
    )
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--basis", default=str(ROOT / "tucker" / "spectral_basis_U3.npy"))
    ap.add_argument( "--unmixing", default=str(ROOT / "baseline" / "unmixing.py"))
    ap.add_argument("--ml-dir", default=str(ROOT / "results" / "ml"))
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "comparison"))
    ap.add_argument("--vca-seed", type=int, default=0)
    ap.add_argument("--fcls-iters", type=int, default=150)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    ml_dir = Path(args.ml_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hsi = np.load(data_dir / "hsi.npy").astype(np.float64)
    true_abund = np.load(data_dir / "abundances.npy").astype(np.float64)
    true_endmembers_kb = np.load(data_dir / "endmembers.npy").astype(np.float64)
    U3 = np.load(args.basis).astype(np.float64)
    ml_pred = np.load(ml_dir / "predicted_abundances.npy").astype(np.float64)
    ml_endmembers_kb = np.load(ml_dir / "learned_endmembers.npy").astype(np.float64)

    H, W, B = hsi.shape
    K = true_abund.shape[2]
    cut = int(0.8 * H)  # identical held-out split to reduced ML script

    if U3.shape[0] != B:
        raise ValueError(f"U3 should have {B} rows; got {U3.shape}")
    if true_endmembers_kb.shape != (K, B):
        raise ValueError(f"Expected true endmembers {(K, B)}, got {true_endmembers_kb.shape}")
    if ml_pred.shape != true_abund.shape:
        raise ValueError(f"ML prediction shape {ml_pred.shape} != truth {true_abund.shape}")

    sunsal = load_sunsal(Path(args.unmixing))

    # ------------------------------------------------------------------
    # Same Tucker coordinates as Marcus: Z = pinv(U3) @ X.
    # Endmembers are learned from the TRAINING region only so held-out
    # accuracy is directly comparable to the ML model's spatial holdout.
    # ------------------------------------------------------------------
    X_all_bn = hsi.reshape(-1, B).T                   # B x N
    X_train_bn = hsi[:cut].reshape(-1, B).T           # B x N_train
    U3_pinv = np.linalg.pinv(U3)                      # R x B

    t_reduce0 = time.perf_counter()
    Z_train = U3_pinv @ X_train_bn                    # R x N_train
    reduction_seconds = time.perf_counter() - t_reduce0

    t_vca0 = time.perf_counter()
    _, idx_train = marcus_vca(Z_train, K, seed=args.vca_seed)
    vca_seconds = time.perf_counter() - t_vca0

    # Marcus selects indices in reduced space, then retrieves ORIGINAL spectra.
    E_est_bk = X_train_bn[:, idx_train]                # B x K
    E_true_bk = true_endmembers_kb.T                   # B x K

    baseline_sads, perm = hungarian_sad(E_true_bk, E_est_bk)

    # Full-scene FCLS using Marcus's SUNSAL dependency.
    t_fcls0 = time.perf_counter()
    A_est_kn, _, _, _ = sunsal(
        E_est_bk,
        X_all_bn,
        positivity=True,
        addone=True,
        verbose=False,
        al_iters=args.fcls_iters,
    )
    fcls_seconds = time.perf_counter() - t_fcls0

    A_baseline_kn = align_abundances(A_est_kn, perm, K)
    baseline_cube = A_baseline_kn.T.reshape(H, W, K)

    # Numerical cleanup only for reporting scene averages; RMSE uses raw FCLS output.
    # This does NOT change the benchmark itself.
    baseline_scene_avg = baseline_cube.mean(axis=(0, 1))

    # ------------------------------------------------------------------
    # Accuracy: identical bottom-20% spatial holdout for both methods.
    # ------------------------------------------------------------------
    truth_test = true_abund[cut:]
    baseline_test = baseline_cube[cut:]
    ml_test = ml_pred[cut:]

    baseline_rmse = float(np.sqrt(np.mean((baseline_test - truth_test) ** 2)))
    ml_rmse = float(np.sqrt(np.mean((ml_test - truth_test) ** 2)))

    true_scene_avg = true_abund.mean(axis=(0, 1))
    ml_scene_avg = ml_pred.mean(axis=(0, 1))

    baseline_scene_rmse = float(np.sqrt(np.mean((baseline_scene_avg - true_scene_avg) ** 2)))
    ml_scene_rmse = float(np.sqrt(np.mean((ml_scene_avg - true_scene_avg) ** 2)))

    ml_sads = rowwise_sad(true_endmembers_kb, ml_endmembers_kb)

    # Pull the neural-network-only inference timing from the ML run if available.
    ml_reported = parse_ml_metrics(ml_dir / "metrics.txt")
    ml_inference_s = ml_reported.get("reported_inference_seconds", np.nan)
    ml_pps = ml_reported.get("reported_pixels_per_second", np.nan)

    baseline_pps = (H * W) / max(fcls_seconds, EPS)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("TUCKER + ML  vs  TUCKER + VCA + FCLS")
    print("=" * 88)
    print(f"HSI                       : {hsi.shape}")
    print(f"Tucker basis U3           : {U3.shape}  ({B} -> {U3.shape[1]} dimensions)")
    print(f"Spatial split             : rows 0:{cut} train, rows {cut}:{H} held out")
    print(f"FCLS iterations           : {args.fcls_iters}")
    print()
    print(f"{'Metric':36s} {'Tucker + ML':>18s} {'Tucker + VCA + FCLS':>24s}")
    print("-" * 82)
    print(f"{'Held-out abundance RMSE':36s} {ml_rmse:18.6f} {baseline_rmse:24.6f}")
    print(f"{'Scene-average RMSE':36s} {ml_scene_rmse:18.6f} {baseline_scene_rmse:24.6f}")
    print(f"{'Mean endmember SAD (deg)':36s} {ml_sads.mean():18.6f} {baseline_sads.mean():24.6f}")
    print(f"{'Max endmember SAD (deg)':36s} {ml_sads.max():18.6f} {baseline_sads.max():24.6f}")
    if np.isfinite(ml_inference_s):
        print(f"{'Whole-scene abundance time (s)':36s} {ml_inference_s:18.6f} {fcls_seconds:24.6f}")
    else:
        print(f"{'Whole-scene abundance time (s)':36s} {'n/a':>18s} {fcls_seconds:24.6f}")
    if np.isfinite(ml_pps):
        print(f"{'Pixels per second':36s} {ml_pps:18.1f} {baseline_pps:24.1f}")
    else:
        print(f"{'Pixels per second':36s} {'n/a':>18s} {baseline_pps:24.1f}")

    print("\nBaseline setup timing (not included in FCLS abundance time):")
    print(f"  Tucker projection of training region : {reduction_seconds:.6f} s")
    print(f"  Reduced-space VCA endmember finding  : {vca_seconds:.6f} s")

    print("\nPer-mineral scene-average abundance")
    print(f"{'Mineral':18s} {'True':>10s} {'ML':>10s} {'VCA+FCLS':>12s}")
    print("-" * 54)
    for i, name in enumerate(MATERIALS[:K]):
        print(f"{name:18s} {true_scene_avg[i]:10.5f} {ml_scene_avg[i]:10.5f} {baseline_scene_avg[i]:12.5f}")

    # ------------------------------------------------------------------
    # Save machine-readable results.
    # ------------------------------------------------------------------
    method_df = pd.DataFrame([
        {
            "Method": "Tucker + ML",
            "Held-out abundance RMSE": ml_rmse,
            "Scene-average RMSE": ml_scene_rmse,
            "Mean endmember SAD (deg)": float(ml_sads.mean()),
            "Max endmember SAD (deg)": float(ml_sads.max()),
            "Whole-scene abundance time (s)": ml_inference_s,
            "Pixels per second": ml_pps,
        },
        {
            "Method": "Tucker + VCA + FCLS",
            "Held-out abundance RMSE": baseline_rmse,
            "Scene-average RMSE": baseline_scene_rmse,
            "Mean endmember SAD (deg)": float(baseline_sads.mean()),
            "Max endmember SAD (deg)": float(baseline_sads.max()),
            "Whole-scene abundance time (s)": fcls_seconds,
            "Pixels per second": baseline_pps,
        },
    ])
    method_df.to_csv(out_dir / "method_comparison.csv", index=False)

    mineral_df = pd.DataFrame({
        "Mineral": MATERIALS[:K],
        "True scene abundance": true_scene_avg,
        "ML scene abundance": ml_scene_avg,
        "VCA+FCLS scene abundance": baseline_scene_avg,
        "ML SAD (deg)": ml_sads,
        "VCA+FCLS SAD (deg)": baseline_sads,
    })
    mineral_df.to_csv(out_dir / "per_mineral_comparison.csv", index=False)

    np.save(out_dir / "baseline_predicted_abundances.npy", baseline_cube)
    np.save(out_dir / "baseline_estimated_endmembers.npy", E_est_bk.T)

    # Held-out abundance RMSE chart.
    fig, ax = plt.subplots(figsize=(7, 5))
    methods = ["Tucker + ML", "Tucker + VCA + FCLS"]
    values = [ml_rmse, baseline_rmse]
    bars = ax.bar(methods, values)
    ax.set_ylabel("Held-out abundance RMSE")
    ax.set_title("Mineral abundance estimation: ML vs classical baseline")
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val:.4f}",
                ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(out_dir / "heldout_rmse_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # True vs both predicted scene compositions.
    x = np.arange(K)
    width = 0.26
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(x - width, true_scene_avg * 100, width, label="True")
    ax.bar(x, ml_scene_avg * 100, width, label="Tucker + ML")
    ax.bar(x + width, baseline_scene_avg * 100, width, label="Tucker + VCA + FCLS")
    ax.set_ylabel("Scene-average abundance (%)")
    ax.set_title("Scene composition: ground truth vs ML and classical unmixing")
    ax.set_xticks(x)
    ax.set_xticklabels(MATERIALS[:K], rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "scene_composition_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved comparison outputs to: {out_dir.resolve()}")
    print("  method_comparison.csv")
    print("  per_mineral_comparison.csv")
    print("  heldout_rmse_comparison.png")
    print("  scene_composition_comparison.png")
    print("  baseline_predicted_abundances.npy")
    print("  baseline_estimated_endmembers.npy")


if __name__ == "__main__":
    main()
