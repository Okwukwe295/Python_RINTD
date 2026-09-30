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


class ConstrainedUnmixingAE(nn.Module):
    """Spectral autoencoder with physically interpretable latent abundances.

    Encoder: spectrum -> K abundance fractions (softmax => nonnegative, sum to 1)
    Decoder: abundance fractions -> reconstructed spectrum using trainable endmembers
    Endmembers are constrained to [0, 1] using sigmoid parameterisation.
    """

    def __init__(self, n_bands: int, n_endmembers: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_bands, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_endmembers),
        )
        self.endmember_logits = nn.Parameter(torch.zeros(n_endmembers, n_bands))

    def endmembers(self):
        return torch.sigmoid(self.endmember_logits)

    def forward(self, x):
        abundances = torch.softmax(self.encoder(x), dim=1)
        endmembers = self.endmembers()
        reconstruction = abundances @ endmembers
        return abundances, reconstruction


def spectral_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.sum(a * b, axis=1)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    cosang = np.clip(dots / np.maximum(denom, 1e-12), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Folder containing hsi.npy, abundances.npy, endmembers.npy")
    ap.add_argument("--out-dir", default="cuprite_ml_out")
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

    H, W, B = hsi.shape
    H2, W2, K = abund.shape
    if (H, W) != (H2, W2):
        raise ValueError(f"Spatial mismatch: HSI {hsi.shape}, abundance {abund.shape}")
    if true_endmembers.shape != (K, B):
        raise ValueError(f"Endmember shape mismatch: expected {(K, B)}, got {true_endmembers.shape}")

    print(f"HSI         : {hsi.shape}")
    print(f"Abundances  : {abund.shape}")
    print(f"Endmembers  : {true_endmembers.shape}")
    print(f"Abundance sum mean/min/max: {abund.sum(axis=2).mean():.6f} / "
          f"{abund.sum(axis=2).min():.6f} / {abund.sum(axis=2).max():.6f}")

    # Spatial holdout: train on top 80% of rows, test on bottom 20%.
    # This is preferable to a random pixel split for a spatial scene.
    cut = int(0.8 * H)
    X_train = hsi[:cut].reshape(-1, B)
    A_train = abund[:cut].reshape(-1, K)
    X_test = hsi[cut:].reshape(-1, B)
    A_test = abund[cut:].reshape(-1, K)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(A_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = ConstrainedUnmixingAE(B, K).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for x, a_true in train_loader:
            x = x.to(device)
            a_true = a_true.to(device)

            a_pred, x_hat = model(x)
            loss_abund = F.mse_loss(a_pred, a_true)
            loss_recon = F.mse_loss(x_hat, x)
            loss = args.abundance_weight * loss_abund + args.reconstruction_weight * loss_recon

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(x)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:3d}/{args.epochs}  loss={running/len(train_ds):.6f}")

    train_seconds = time.perf_counter() - start

    # Evaluate on held-out rows.
    model.eval()
    with torch.no_grad():
        X_test_t = torch.from_numpy(X_test).to(device)
        A_test_t = torch.from_numpy(A_test).to(device)
        A_pred_t, X_hat_t = model(X_test_t)
        abundance_rmse = torch.sqrt(F.mse_loss(A_pred_t, A_test_t)).item()
        reconstruction_rmse = torch.sqrt(F.mse_loss(X_hat_t, X_test_t)).item()
        learned_endmembers = model.endmembers().detach().cpu().numpy()

    sad = spectral_angle_deg(learned_endmembers, true_endmembers)

    # Run the whole scene to produce abundance maps.
    X_all = torch.from_numpy(hsi.reshape(-1, B)).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        A_all_pred, _ = model(X_all)
    inference_seconds = time.perf_counter() - t0
    abundance_maps = A_all_pred.cpu().numpy().reshape(H, W, K)
    scene_average = abundance_maps.mean(axis=(0, 1))

    np.save(out_dir / "predicted_abundances.npy", abundance_maps)
    np.save(out_dir / "learned_endmembers.npy", learned_endmembers)
    np.save(out_dir / "scene_average_abundances.npy", scene_average)
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_bands": B,
        "n_endmembers": K,
        "materials": MATERIALS,
    }, out_dir / "cuprite_unmixing_model.pt")

    lines = [
        f"HSI shape: {hsi.shape}",
        f"Abundance shape: {abund.shape}",
        f"Endmember shape: {true_endmembers.shape}",
        f"Train seconds: {train_seconds:.3f}",
        f"Held-out abundance RMSE: {abundance_rmse:.6f}",
        f"Held-out reconstruction RMSE: {reconstruction_rmse:.6f}",
        f"Mean endmember SAD (deg): {sad.mean():.6f}",
        f"Max endmember SAD (deg): {sad.max():.6f}",
        f"Whole-scene inference seconds: {inference_seconds:.6f}",
        f"Pixels per second: {(H*W)/max(inference_seconds,1e-12):.1f}",
        "",
        "Per-endmember SAD and predicted scene-average abundance:",
    ]
    for i in range(K):
        name = MATERIALS[i] if i < len(MATERIALS) else f"Endmember_{i+1}"
        lines.append(f"{i+1:02d} {name:18s} SAD={sad[i]:7.3f} deg   avg_abundance={scene_average[i]:.5f}")

    metrics = "\n".join(lines)
    print("\n" + metrics)
    (out_dir / "metrics.txt").write_text(metrics, encoding="utf-8")

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
