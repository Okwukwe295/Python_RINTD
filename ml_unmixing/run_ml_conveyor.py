#!/usr/bin/env python3
"""
Run the trained reduced-input ML unmixing model over the same synthetic
conveyor-scene stream used by baseline/sp_conveyor.py.

This is intended for an apples-to-apples 3000-scene baseline-vs-ML benchmark.
The script reproduces the baseline RNG sequence (including burn-in generation),
then measures Tucker projection + neural-network inference per scene.
"""

from pathlib import Path
import argparse
import csv
import importlib.util
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "baseline"
ML_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = ROOT / "results" / "ml" / "cuprite_reduced_unmixing_model.pt"
DEFAULT_ENDMEMBERS = ROOT / "data" / "endmembers.npy"
DEFAULT_RESULTS = ROOT / "results" / "ml" / "ml_conveyor_results.csv"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Reuse the exact scene generator from the baseline.
baseline = load_module("sp_conveyor", BASELINE_DIR / "sp_conveyor.py")
generate_scene = baseline.generate_scene

# Reuse the exact model definition used for training.
train_mod = load_module(
    "train_cuprite_unmixing_reduced",
    ML_DIR / "train_cuprite_unmixing_reduced.py",
)
ReducedInputUnmixingAE = train_mod.ReducedInputUnmixingAE


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n-scenes", type=int, default=3000)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--endmembers", default=str(DEFAULT_ENDMEMBERS))
    ap.add_argument("--results-csv", default=str(DEFAULT_RESULTS))

    # Keep these identical to sp_conveyor.py when you want the exact same scenes.
    ap.add_argument("--burnin", type=int, default=5)
    ap.add_argument("--scene-min", type=int, default=150)
    ap.add_argument("--scene-max", type=int, default=250)
    ap.add_argument("--snr-min", type=float, default=25.0)
    ap.add_argument("--snr-max", type=float, default=45.0)
    ap.add_argument("--alpha-min", type=float, default=0.3)
    ap.add_argument("--alpha-max", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--batch-size", type=int, default=8192)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ml-conveyor] device={device}")

    # ------------------------------------------------------------------
    # Load trained model + Tucker reduction statistics.
    # ------------------------------------------------------------------
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    R = int(checkpoint["input_dim"])
    B = int(checkpoint["n_bands"])
    K = int(checkpoint["n_endmembers"])

    model = ReducedInputUnmixingAE(R, B, K).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    U3 = np.asarray(checkpoint["basis_U3"], dtype=np.float32)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float32)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float32)
    U3_pinv = np.linalg.pinv(U3).astype(np.float32)

    if U3.shape != (B, R):
        raise ValueError(f"Checkpoint U3 should be {(B, R)}, got {U3.shape}")

    # sp_conveyor.py treats endmembers.npy as K x B, then transposes to B x K.
    endm = np.load(args.endmembers).astype(np.float32)
    if endm.ndim != 2 or endm.shape[1] != B:
        raise ValueError(
            f"Expected endmembers.npy shape (K_pool, {B}), got {endm.shape}"
        )

    E_pool = endm.T
    B_pool, K_pool = E_pool.shape

    if K_pool != K:
        raise ValueError(
            "For direct abundance-RMSE comparison, the trained model output "
            f"count ({K}) must match the conveyor endmember pool ({K_pool}). "
            "If you intentionally use K < K_pool, add an explicit global "
            "material-label mapping before comparing abundances."
        )

    # ------------------------------------------------------------------
    # Reproduce the same RNG sequence as run_conveyor().
    # ------------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    active = list(rng.choice(K_pool, size=K, replace=False))
    E_true = E_pool[:, active].copy()
    W_fix = args.scene_min + 8

    print(
        f"[ml-conveyor] reproducing baseline stream: "
        f"burn-in={args.burnin}, scenes={args.n_scenes}, "
        f"W={W_fix}, B={B}, K={K}, R={R}"
    )
    print(f"[ml-conveyor] active endmember order={active}")

    # IMPORTANT:
    # The baseline consumes RNG during burn-in scene generation before scene 0.
    # We must do the same here, even though the ML model does not use burn-in.
    for _ in range(args.burnin):
        H = int(rng.integers(args.scene_min, args.scene_max + 1))
        snr = float(rng.uniform(args.snr_min, args.snr_max))
        alpha = float(rng.uniform(args.alpha_min, args.alpha_max))
        generate_scene(E_true, H, W_fix, snr, alpha, rng)

    rows = []
    total_pixels = 0
    wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Process the same synthetic scenes with the ML model.
    # ------------------------------------------------------------------
    for t in range(args.n_scenes):
        H = int(rng.integers(args.scene_min, args.scene_max + 1))
        snr = float(rng.uniform(args.snr_min, args.snr_max))
        alpha = float(rng.uniform(args.alpha_min, args.alpha_max))

        X_cube, A_true_local, _ = generate_scene(
            E_true, H, W_fix, snr, alpha, rng
        )

        X_flat = X_cube.reshape(-1, B).astype(np.float32, copy=False)
        N = X_flat.shape[0]
        total_pixels += N

        # generate_scene labels abundances in the *active* endmember order.
        # The trained model predicts in the original endmembers.npy order.
        A_true_global = np.zeros((N, K), dtype=np.float32)
        for local_idx, global_idx in enumerate(active):
            A_true_global[:, global_idx] = A_true_local[local_idx]

        # Tucker spectral reduction: X (N x B) -> Z (N x R)
        t0 = time.perf_counter()
        Z = X_flat @ U3_pinv.T
        Z = (Z - z_mean) / z_std
        t_reduce = time.perf_counter() - t0

        # Batched neural-network inference.
        t0 = time.perf_counter()
        abund_sqerr = 0.0
        recon_sqerr = 0.0
        abund_count = 0
        recon_count = 0

        with torch.inference_mode():
            for start in range(0, N, args.batch_size):
                stop = min(start + args.batch_size, N)

                z_batch = torch.from_numpy(
                    np.ascontiguousarray(Z[start:stop])
                ).to(device)

                a_pred, x_hat = model(z_batch)

                a_true = torch.from_numpy(
                    np.ascontiguousarray(A_true_global[start:stop])
                ).to(device)
                x_true = torch.from_numpy(
                    np.ascontiguousarray(X_flat[start:stop])
                ).to(device)

                abund_sqerr += torch.sum((a_pred - a_true) ** 2).item()
                recon_sqerr += torch.sum((x_hat - x_true) ** 2).item()
                abund_count += a_true.numel()
                recon_count += x_true.numel()

        # Synchronize so GPU timing is real rather than asynchronous launch time.
        if device.type == "cuda":
            torch.cuda.synchronize()

        t_infer = time.perf_counter() - t0
        t_total = t_reduce + t_infer

        abundance_rmse = np.sqrt(abund_sqerr / max(abund_count, 1))
        reconstruction_rmse = np.sqrt(recon_sqerr / max(recon_count, 1))

        rows.append({
            "scene": t,
            "H": H,
            "W": W_fix,
            "N": N,
            "snr_db": snr,
            "alpha": alpha,
            "abundance_rmse": abundance_rmse,
            "reconstruction_rmse": reconstruction_rmse,
            "t_reduce": t_reduce,
            "t_infer": t_infer,
            "t_total_ml": t_total,
            "pixels_per_second": N / max(t_total, 1e-12),
        })

        if t < 5 or t % 10 == 0:
            print(
                f"  scene {t:5d}: "
                f"A-RMSE={abundance_rmse:.4f} "
                f"recon-RMSE={reconstruction_rmse:.4f} | "
                f"reduce={t_reduce*1000:.1f}ms "
                f"infer={t_infer*1000:.1f}ms "
                f"total={t_total*1000:.1f}ms"
            )

    wall_seconds = time.perf_counter() - wall_start

    # ------------------------------------------------------------------
    # Save per-scene benchmark results.
    # ------------------------------------------------------------------
    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "scene", "H", "W", "N", "snr_db", "alpha",
        "abundance_rmse", "reconstruction_rmse",
        "t_reduce", "t_infer", "t_total_ml", "pixels_per_second",
    ]

    with results_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    t_reduce = np.array([r["t_reduce"] for r in rows])
    t_infer = np.array([r["t_infer"] for r in rows])
    t_total = np.array([r["t_total_ml"] for r in rows])
    armse = np.array([r["abundance_rmse"] for r in rows])
    rrmse = np.array([r["reconstruction_rmse"] for r in rows])

    print("\n[ml-conveyor summary]")
    print(f"  scenes                  : {len(rows)}")
    print(f"  pixels                  : {total_pixels:,}")
    print(f"  wall time               : {wall_seconds:.3f}s")
    print(f"  mean reduction / scene  : {t_reduce.mean()*1000:.3f} ms")
    print(f"  mean inference / scene  : {t_infer.mean()*1000:.3f} ms")
    print(f"  mean ML total / scene   : {t_total.mean()*1000:.3f} ms")
    print(f"  median ML total / scene : {np.median(t_total)*1000:.3f} ms")
    print(f"  mean abundance RMSE     : {armse.mean():.6f}")
    print(f"  mean reconstruction RMSE: {rrmse.mean():.6f}")
    print(f"  saved                   : {results_path.resolve()}")


if __name__ == "__main__":
    main()
