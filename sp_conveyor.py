#!/usr/bin/env python3
"""
Synthetic HSI conveyor belt using RI-NTD (recursive_update).

Performance fix vs previous version:
  - Core tensor update now uses multiplicative updates (Kim & Choi style)
    instead of scipy.optimize.nnls on the (J0*J1*J2)^2 Kronecker system.
    This is the dominant cost at rank 14 (2744x2744) and the previous
    nnls call took seconds per scene. MU is ~50x faster.

  - Recon error uses only the current scene slice, not the full history.
    This was already in the previous version.

  - U[0] grows by design (linear in samples).

All tensor math uses tensorly's tenalg.mode_dot and multi_mode_dot.
"""

import argparse
import csv
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment, nnls
import tensorly as tl
from tensorly import tenalg
from tensorly.decomposition import non_negative_tucker_hals

# SciPy compat shim
import scipy as sp
for _name in ["dot", "mean", "sum", "amax", "amin", "sqrt", "log10", "log",
              "zeros", "ones", "eye", "vstack", "hstack", "concatenate",
              "absolute", "maximum", "minimum", "argmax", "argmin",
              "reshape", "transpose", "repeat", "inf", "nan", "pi"]:
    if not hasattr(sp, _name):
        setattr(sp, _name, getattr(np, _name))
if not hasattr(sp, "random"):
    sp.random = np.random

import importlib.util
spec = importlib.util.spec_from_file_location("unmixing", "unmixing.py")
um = importlib.util.module_from_spec(spec)
spec.loader.exec_module(um)
sunsal = um.sunsal

tl.set_backend("numpy")
EPS = 1e-12


# ======================================================================
SYNTH_DIR   = r"C:\Users\Marcus\Desktop\TuckerDecomp"
RESULTS_CSV = "conveyor_results.csv"
SUMMARY_PNG = "conveyor_summary.png"

BURNIN_SCENES = 5
HALS_ITERS    = 30
K_INNER       = 10
GAMMA         = 1e-3
VCA_SEED      = 0
SUNSAL_ITERS  = 150

CORE_MU_ITERS = 100        # multiplicative iterations for core update

SCENE_MIN, SCENE_MAX   = 32, 96
SNR_DB_MIN, SNR_DB_MAX = 25.0, 45.0
DIRICHLET_ALPHA_MIN, DIRICHLET_ALPHA_MAX = 0.3, 2.0

DRIFT_COUNT = 3
# ======================================================================


# ----------------------------------------------------------------------
# RI-NTD update (recursive approach)
# ----------------------------------------------------------------------
def recursive_update(X_n, U, G, P, Q, start, nonneg=True,
                      k_inner=10, gamma=1e-3, lam=1,
                      core_mu_iters=CORE_MU_ITERS):
    """
    Incremental NTD (Zdunek & Fonał 2022, recursive approach).

    X_n   : (H_t, W, B)  — new scene slice
    U     : [U0, U1, U2] — U0 pre-allocated with room for new rows
    G     : core tensor
    P, Q  : accumulated auxiliary matrices
    start : row index in U[0] where the new H_t rows are written
    """
    N_modes = len(U)
    block_size = X_n.shape[0]
    last = 0                                    # growth axis

    # ---- solve for the new rows of U[0] via NNLS (cheap) ----
    W = G
    for m in range(N_modes):
        if m != last:
            W = tenalg.mode_dot(W, U[m], mode=m)

    Wn = tl.unfold(W, mode=last)                # (J0, W*B)
    Xn = tl.unfold(X_n, mode=last)              # (H_t, W*B)

    A_last = Wn @ Wn.T
    U_new = np.zeros((block_size, Wn.shape[0]))
    for row in range(block_size):
        rhs = Wn @ Xn[row, :]
        sol, _ = nnls(A_last, rhs)
        U_new[row, :] = sol

    U[last][start:start + block_size, :] = U_new
    U_block_last = U_new

    # ---- Gauss-Seidel updates for U[1], U[2] ----
    for n in range(1, N_modes):
        W = G
        for m in range(N_modes):
            if m != n and m != last:
                W = tenalg.mode_dot(W, U[m], mode=m)
        W = tenalg.mode_dot(W, U_block_last, mode=last)

        Xn_unf = tl.unfold(X_n, mode=n)
        Wn_unf = tl.unfold(W, mode=n)

        P[n] = lam * P[n] + Xn_unf @ Wn_unf.T
        Q[n] = lam * Q[n] + Wn_unf @ Wn_unf.T

        for _ in range(k_inner):
            Q_reg = Q[n] + gamma * np.eye(Q[n].shape[0])
            for j in range(U[n].shape[1]):
                num = P[n][:, j] - U[n] @ Q_reg[:, j]
                U[n][:, j] = U[n][:, j] + num / (Q_reg[j, j] + EPS)
                if nonneg:
                    U[n][:, j] = np.maximum(U[n][:, j], EPS)
        U[n] = U[n] / (U[n].sum(axis=0, keepdims=True) + EPS)

    # ---- core tensor update via multiplicative updates ----
    # Z_core = X_n ×_0 U_t.T ×_1 U1.T ×_2 U2.T
    Z_core = tl.tensor(X_n)
    grams = []
    for n in range(N_modes):
        U_n = U_block_last if n == last else U[n]
        Z_core = tenalg.mode_dot(Z_core, U_n.T, mode=n)
        grams.append(U_n.T @ U_n)

    A_core = grams[0]
    for g in grams[1:]:
        A_core = np.kron(A_core, g)

    z_vec = Z_core.reshape(-1)

    # Multiplicative update for g_vec, warm-started from previous core
    g_vec = np.maximum(G.flatten(), EPS)
    num = A_core.T @ z_vec                       # compute once
    for _ in range(core_mu_iters):
        denom = A_core.T @ (A_core @ g_vec)
        g_vec = g_vec * (num + EPS) / (denom + EPS)
    G_new = g_vec.reshape(G.shape)

    return U, G_new, P, Q


# ----------------------------------------------------------------------
# Per-scene batch metrics
# ----------------------------------------------------------------------
def vca(Y, p, seed=0):
    rng = np.random.default_rng(seed)
    B, N = Y.shape
    U_, S, Vt = np.linalg.svd(Y, full_matrices=False)
    Ud = U_[:, :p]
    Xp = Ud.T @ Y
    A = np.zeros((p, p)); A[-1, 0] = 1.0
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


def sam_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 90.0
    c = float(np.dot(a, b) / (na * nb))
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def hungarian_sad(E_true, E_est):
    K = E_true.shape[1]
    cost = np.array([[sam_deg(E_true[:, i], E_est[:, j])
                      for j in range(K)] for i in range(K)])
    r, c = linear_sum_assignment(cost)
    perm = dict(zip(r.tolist(), c.tolist()))
    sads = np.array([cost[i, perm[i]] for i in range(K)])
    return float(sads.mean()), float(sads.max()), sads, perm


def fcls(X, E, iters=SUNSAL_ITERS):
    A, _, _, _ = sunsal(E, X, positivity=True, addone=True,
                        verbose=False, al_iters=iters)
    return A


def abundance_rmse(A_true, A_est, perm):
    K = A_true.shape[0]
    A_aligned = np.zeros_like(A_true)
    for i in range(K):
        A_aligned[i] = A_est[perm[i]]
    return float(np.sqrt(np.mean((A_true - A_aligned) ** 2)))


# ----------------------------------------------------------------------
# Scene generation
# ----------------------------------------------------------------------
def generate_scene(E_true, H, W, snr_db, alpha, rng):
    K = E_true.shape[1]
    N = H * W
    A_true = rng.dirichlet(alpha * np.ones(K), size=N).T
    X = E_true @ A_true
    sig_pow = np.mean(X ** 2)
    noise_pow = sig_pow / (10 ** (snr_db / 10.0))
    X = X + rng.normal(0, np.sqrt(noise_pow), X.shape)
    X = np.clip(X, 0.0, 1.5)
    X_cube = X.T.reshape(H, W, E_true.shape[0])
    return X_cube, A_true, X


# ----------------------------------------------------------------------
# Conveyor loop
# ----------------------------------------------------------------------
def run_conveyor(args):
    endm = np.load(Path(SYNTH_DIR) / "endmembers.npy").astype(np.float64)
    E_pool = endm.T
    B_pool, K_pool = E_pool.shape
    K = args.K
    r3 = K + args.rank_offset
    ranks = [r3, r3, r3]

    rng = np.random.default_rng(args.seed)
    active = list(rng.choice(K_pool, size=K, replace=False))
    E_true = E_pool[:, active].copy()

    W_fix = args.scene_min + 8

    # ---- burn-in ----
    print(f"[conveyor] burn-in {args.burnin} scenes, K={K}, ranks={ranks}")
    scenes = []
    for _ in range(args.burnin):
        H = int(rng.integers(args.scene_min, args.scene_max + 1))
        snr = float(rng.uniform(args.snr_min, args.snr_max))
        alpha = float(rng.uniform(args.alpha_min, args.alpha_max))
        X_scene, _, _ = generate_scene(E_true, H, W_fix, snr, alpha, rng)
        scenes.append(X_scene)

    X_burn = np.concatenate(scenes, axis=0)
    print(f"  burn-in tensor: {X_burn.shape}")

    t0 = time.perf_counter()
    tucker = non_negative_tucker_hals(
        tl.tensor(X_burn), rank=ranks,
        algorithm="fista", n_iter_max=args.hals_iters, tol=1e-5,
        init="random", random_state=args.seed,
    )
    U = [np.asarray(f) for f in tucker.factors]
    G = np.asarray(tucker.core)
    print(f"  burn-in HALS: {time.perf_counter() - t0:.2f}s")

    # ---- initialize P[1], P[2] from burn-in ----
    P = [None] * 3
    Q = [None] * 3
    for n in [1, 2]:
        W = G
        for m in range(3):
            if m != n:
                W = tenalg.mode_dot(W, U[m], mode=m)
        Xn_unf = tl.unfold(X_burn, mode=n)
        Wn_unf = tl.unfold(W, mode=n)
        P[n] = Xn_unf @ Wn_unf.T
        Q[n] = Wn_unf @ Wn_unf.T

    # ---- pre-allocate U[0] for the entire run ----
    max_total_rows = X_burn.shape[0] + args.n_scenes * args.scene_max
    U[0] = np.vstack([U[0], np.zeros((max_total_rows - U[0].shape[0],
                                       U[0].shape[1]))])
    row_start = X_burn.shape[0]

    # ---- process scenes ----
    rows = []
    print(f"[conveyor] incremental phase ...")

    for t in range(args.n_scenes):
        if args.drift_scene is not None and t == args.drift_scene:
            print(f"[drift] swapping {args.drift_count} at scene {t}")
            swap_out = rng.choice(K, size=args.drift_count, replace=False)
            cand = [k for k in range(K_pool) if k not in active]
            swap_in = rng.choice(cand, size=args.drift_count, replace=False)
            for so, si in zip(swap_out, swap_in):
                active[so] = si
            E_true = E_pool[:, active].copy()

        H = int(rng.integers(args.scene_min, args.scene_max + 1))
        snr = float(rng.uniform(args.snr_min, args.snr_max))
        alpha = float(rng.uniform(args.alpha_min, args.alpha_max))
        X_cube, A_true, _ = generate_scene(E_true, H, W_fix, snr, alpha, rng)

        t0 = time.perf_counter()
        U, G, P, Q = recursive_update(
            X_cube, U, G, P, Q,
            start=row_start, nonneg=True,
            k_inner=K_INNER, gamma=GAMMA, lam=args.lam,
        )
        t_ntd = time.perf_counter() - t0

        # ---- reconstruct only the current scene slice ----
        row_end = row_start + H
        U_t_block = U[0][row_start:row_end, :]
        X_rec_slice = np.asarray(tenalg.multi_mode_dot(
            tl.tensor(G),
            [tl.tensor(U_t_block), tl.tensor(U[1]), tl.tensor(U[2])],
            modes=[0, 1, 2],
        ))
        recon = np.linalg.norm(X_cube - X_rec_slice) / (
            np.linalg.norm(X_cube) + EPS)
        row_start = row_end

        # ---- per-scene batch outputs ----
        Bc = X_cube.shape[2]
        X_flat = X_cube.reshape(-1, Bc).T
        U3 = U[2]
        U3_pinv = np.linalg.pinv(U3)
        Z = U3_pinv @ X_flat
        _, idx_z = vca(Z, K, seed=VCA_SEED)
        E_z = X_flat[:, idx_z]
        sad_mean, sad_max, _, perm = hungarian_sad(E_true, E_z)

        t0 = time.perf_counter()
        A_est = fcls(X_flat, E_z)
        rmse = abundance_rmse(A_true, A_est, perm)
        t_fcls = time.perf_counter() - t0

        rows.append({
            "scene": t, "H": H, "W": W_fix, "N": H * W_fix,
            "snr_db": snr, "alpha": alpha,
            "sad_mean": sad_mean, "sad_max": sad_max, "rmse": rmse,
            "recon_err": recon, "t_ntd": t_ntd, "t_fcls": t_fcls,
        })

        if t % 10 == 0 or t < 5:
            print(f"  scene {t:5d}: SAD={sad_mean:.2f}° RMSE={rmse:.4f} "
                  f"recon={recon:.4f} | NTD {t_ntd*1000:.1f}ms")

    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_scenes", type=int, default=100)
    ap.add_argument("--K", type=int, default=12)
    ap.add_argument("--rank_offset", type=int, default=2)
    ap.add_argument("--burnin", type=int, default=BURNIN_SCENES)
    ap.add_argument("--hals_iters", type=int, default=HALS_ITERS)
    ap.add_argument("--scene_min", type=int, default=SCENE_MIN)
    ap.add_argument("--scene_max", type=int, default=SCENE_MAX)
    ap.add_argument("--snr_min", type=float, default=SNR_DB_MIN)
    ap.add_argument("--snr_max", type=float, default=SNR_DB_MAX)
    ap.add_argument("--alpha_min", type=float, default=DIRICHLET_ALPHA_MIN)
    ap.add_argument("--alpha_max", type=float, default=DIRICHLET_ALPHA_MAX)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--drift_scene", type=int, default=None)
    ap.add_argument("--drift_count", type=int, default=DRIFT_COUNT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--results_csv", default=RESULTS_CSV)
    ap.add_argument("--summary_png", default=SUMMARY_PNG)
    args = ap.parse_args()

    t_start = time.perf_counter()
    rows = run_conveyor(args)
    t_total = time.perf_counter() - t_start

    fields = ["scene", "H", "W", "N", "snr_db", "alpha",
              "sad_mean", "sad_max", "rmse", "recon_err", "t_ntd", "t_fcls"]
    with open(args.results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n[done] {len(rows)} scenes in {t_total:.1f}s "
          f"({t_total/len(rows):.3f}s/scene)")

    sad = np.array([r["sad_mean"] for r in rows])
    rmse = np.array([r["rmse"] for r in rows])
    recon = np.array([r["recon_err"] for r in rows])

    print(f"\n[summary]")
    print(f"  SAD   : mean {sad.mean():.3f}°  median {np.median(sad):.3f}°")
    print(f"  RMSE  : mean {rmse.mean():.4f}  median {np.median(rmse):.4f}")
    print(f"  recon : mean {recon.mean():.4f}")

    n10 = min(10, len(rows))
    print(f"\n[warmup]")
    print(f"  first {n10}: SAD {sad[:n10].mean():.3f}°  RMSE {rmse[:n10].mean():.4f}  recon {recon[:n10].mean():.4f}")
    print(f"  last  {n10}: SAD {sad[-n10:].mean():.3f}°  RMSE {rmse[-n10:].mean():.4f}  recon {recon[-n10:].mean():.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, y, ylabel, color, title in [
        (axes[0], sad,   "SAD (°)",        "tab:blue",   "Endmember SAD"),
        (axes[1], rmse,  "abundance RMSE", "tab:orange", "Abundance RMSE"),
        (axes[2], recon, "recon err",      "tab:green",  "Reconstruction error"),
    ]:
        ax.plot(y, lw=0.8, alpha=0.6, color=color, label="per scene")
        if len(y) >= 20:
            ax.plot(np.convolve(y, np.ones(20)/20, mode="valid"),
                    lw=2, color=color, label="moving avg (20)")
        if args.drift_scene is not None:
            ax.axvline(args.drift_scene, color="red", ls="--", alpha=0.6)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(True, alpha=0.3); ax.legend()

    axes[2].set_xlabel("scene index")
    fig.suptitle(f"RI-NTD conveyor | {len(rows)} scenes | K={args.K} "
                 f"r3={args.K+args.rank_offset} lam={args.lam}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.summary_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {args.summary_png}")


if __name__ == "__main__":
    main()