from pathlib import Path
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

MATERIALS = [
    "Alunite", "Andradite", "Buddingtonite", "Dumortierite",
    "Kaolinite_1", "Kaolinite_2", "Muscovite", "Montmorillonite",
    "Nontronite", "Pyrope", "Sphene", "Chalcedony",
]


class ReducedInputUnmixingAE(nn.Module):
    """Reduced spectral coordinates -> abundances -> full-spectrum reconstruction.

    Input is Marcus's reduced spectral coordinate z (R dimensions).
    Encoder predicts K abundance fractions.
    Decoder remains physically interpretable in the original B-band wavelength space.
    """
    def __init__(self, input_dim: int, n_bands: int, n_endmembers: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, n_endmembers),
        )
        self.endmember_logits = nn.Parameter(torch.zeros(n_endmembers, n_bands))

    def endmembers(self):
        return torch.sigmoid(self.endmember_logits)

    def forward(self, z):
        abundances = torch.softmax(self.encoder(z), dim=1)
        reconstruction = abundances @ self.endmembers()
        return abundances, reconstruction


def spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosang = np.clip(dots / np.maximum(denom, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def reduce_with_u3(X: np.ndarray, U3: np.ndarray) -> np.ndarray:
    """Match Marcus: Z = pinv(U3) @ X_flat, returned as N x R."""
    U3_pinv = np.linalg.pinv(U3)       # R x B
    return X @ U3_pinv.T               # N x R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--basis", required=True,
                    help="Marcus spectral factor U3 saved as .npy, shape (B, R), e.g. (188,14)")
    ap.add_argument("--out-dir", default="cuprite_ml_reduced_out")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--abundance-weight", type=float, default=1.0)
    ap.add_argument("--reconstruction-weight", type=float, default=1.0)
    args = ap.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hsi = np.load(data_dir / "hsi.npy").astype(np.float32)
    abund = np.load(data_dir / "abundances.npy").astype(np.float32)
    true_endmembers = np.load(data_dir / "endmembers.npy").astype(np.float32)
    U3 = np.load(args.basis).astype(np.float32)

    H, W, B = hsi.shape
    _, _, K = abund.shape
    if U3.ndim != 2 or U3.shape[0] != B:
        raise ValueError(f"Expected U3 shape ({B}, R), got {U3.shape}")

    R = U3.shape[1]
    print(f"HSI            : {hsi.shape}")
    print(f"Abundances     : {abund.shape}")
    print(f"Spectral basis : {U3.shape}")
    print(f"Reduced input  : {B} -> {R} dimensions")

    # Spatial holdout identical to the raw-spectrum baseline.
    cut = int(0.8 * H)
    X_train = hsi[:cut].reshape(-1, B)
    A_train = abund[:cut].reshape(-1, K)
    X_test = hsi[cut:].reshape(-1, B)
    A_test = abund[cut:].reshape(-1, K)

    # Marcus projection: z = pinv(U3) x.
    Z_train = reduce_with_u3(X_train, U3).astype(np.float32)
    Z_test = reduce_with_u3(X_test, U3).astype(np.float32)

    # Standardise reduced coordinates using TRAIN statistics only.
    z_mean = Z_train.mean(axis=0, keepdims=True)
    z_std = Z_train.std(axis=0, keepdims=True)
    z_std = np.maximum(z_std, 1e-6)
    Z_train_n = (Z_train - z_mean) / z_std
    Z_test_n = (Z_test - z_mean) / z_std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device         : {device}")

    train_ds = TensorDataset(
        torch.from_numpy(Z_train_n),
        torch.from_numpy(A_train),
        torch.from_numpy(X_train),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ReducedInputUnmixingAE(R, B, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for z, a_true, x_true in train_loader:
            z = z.to(device)
            a_true = a_true.to(device)
            x_true = x_true.to(device)

            a_pred, x_hat = model(z)
            loss_abund = F.mse_loss(a_pred, a_true)
            loss_recon = F.mse_loss(x_hat, x_true)
            loss = (args.abundance_weight * loss_abund +
                    args.reconstruction_weight * loss_recon)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(z)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs} loss={running/len(train_ds):.6f}")

    train_seconds = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        zt = torch.from_numpy(Z_test_n).to(device)
        at = torch.from_numpy(A_test).to(device)
        xt = torch.from_numpy(X_test).to(device)
        A_pred_t, X_hat_t = model(zt)
        abundance_rmse = torch.sqrt(F.mse_loss(A_pred_t, at)).item()
        reconstruction_rmse = torch.sqrt(F.mse_loss(X_hat_t, xt)).item()
        learned_endmembers = model.endmembers().cpu().numpy()

    sad = spectral_angle_deg(learned_endmembers, true_endmembers)

    X_all = hsi.reshape(-1, B)
    Z_all = reduce_with_u3(X_all, U3).astype(np.float32)
    Z_all_n = (Z_all - z_mean) / z_std
    t0 = time.perf_counter()
    with torch.no_grad():
        A_all, _ = model(torch.from_numpy(Z_all_n).to(device))
    inference_seconds = time.perf_counter() - t0

    abundance_maps = A_all.cpu().numpy().reshape(H, W, K)
    scene_average = abundance_maps.mean(axis=(0, 1))

    np.save(out_dir / "predicted_abundances.npy", abundance_maps)
    np.save(out_dir / "learned_endmembers.npy", learned_endmembers)
    np.save(out_dir / "scene_average_abundances.npy", scene_average)
    np.save(out_dir / "spectral_basis_U3.npy", U3)
    np.save(out_dir / "z_mean.npy", z_mean)
    np.save(out_dir / "z_std.npy", z_std)

    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": R,
        "n_bands": B,
        "n_endmembers": K,
        "materials": MATERIALS,
        "basis_U3": U3,
        "z_mean": z_mean,
        "z_std": z_std,
    }, out_dir / "cuprite_reduced_unmixing_model.pt")

    lines = [
        f"HSI shape: {hsi.shape}",
        f"U3 shape: {U3.shape}",
        f"Reduced dimensions: {B} -> {R}",
        f"Train seconds: {train_seconds:.3f}",
        f"Held-out abundance RMSE: {abundance_rmse:.6f}",
        f"Held-out reconstruction RMSE: {reconstruction_rmse:.6f}",
        f"Mean endmember SAD (deg): {sad.mean():.6f}",
        f"Max endmember SAD (deg): {sad.max():.6f}",
        f"Whole-scene inference seconds: {inference_seconds:.6f}",
        f"Pixels per second: {(H*W)/max(inference_seconds, 1e-12):.1f}",
        "",
        "Per-endmember SAD and predicted scene-average abundance:",
    ]
    for i in range(K):
        lines.append(
            f"{i+1:02d} {MATERIALS[i]:18s} SAD={sad[i]:7.3f} deg   "
            f"avg_abundance={scene_average[i]:.5f}"
        )

    metrics = "\n".join(lines)
    print("\n" + metrics)
    (out_dir / "metrics.txt").write_text(metrics, encoding="utf-8")
    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
