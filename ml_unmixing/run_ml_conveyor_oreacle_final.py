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
import json
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


def compute_endmember_sad(true_endmembers, learned_endmembers):
    """Return per-endmember and mean spectral angle distance in degrees."""
    true_endmembers = np.asarray(true_endmembers, dtype=np.float64)
    learned_endmembers = np.asarray(learned_endmembers, dtype=np.float64)

    if true_endmembers.shape != learned_endmembers.shape:
        raise ValueError(
            f"Endmember shapes do not match: true={true_endmembers.shape}, "
            f"learned={learned_endmembers.shape}"
        )

    numerator = np.sum(true_endmembers * learned_endmembers, axis=1)
    denominator = (
        np.linalg.norm(true_endmembers, axis=1)
        * np.linalg.norm(learned_endmembers, axis=1)
    )
    cosine = numerator / np.maximum(denominator, 1e-12)
    cosine = np.clip(cosine, -1.0, 1.0)
    sad_degrees = np.degrees(np.arccos(cosine))
    return sad_degrees.astype(np.float64), float(np.mean(sad_degrees))


def extract_learned_endmembers(model, checkpoint, K, B):
    """
    Extract the trained K x B endmember matrix used by the decoder.

    Preferred path is an explicit model.endmembers() method.  Fallbacks cover
    common autoencoder implementations where a KxB/BxK decoder/endmember tensor
    is stored as a parameter.  We deliberately avoid guessing from arbitrary
    encoder weights.
    """
    # Best case: the model exposes the physically constrained endmembers itself.
    if hasattr(model, "endmembers") and callable(model.endmembers):
        with torch.no_grad():
            E = model.endmembers()
        if torch.is_tensor(E):
            E = E.detach().cpu().numpy()
        E = np.asarray(E, dtype=np.float32)
        if E.shape == (K, B):
            return E, "model.endmembers()"
        if E.shape == (B, K):
            return E.T, "model.endmembers() [transposed]"

    # Sometimes the final constrained endmembers are saved in the checkpoint.
    for key in (
        "learned_endmembers",
        "endmembers_learned",
        "decoder_endmembers",
        "endmember_matrix",
    ):
        if key in checkpoint:
            E = checkpoint[key]
            if torch.is_tensor(E):
                E = E.detach().cpu().numpy()
            E = np.asarray(E, dtype=np.float32)
            if E.shape == (K, B):
                return E, f"checkpoint['{key}']"
            if E.shape == (B, K):
                return E.T, f"checkpoint['{key}'] [transposed]"

    # Common explicit model attributes.
    for attr in (
        "endmember_matrix",
        "endmember_spectra",
        "endmembers_param",
        "raw_endmembers",
        "E",
    ):
        if hasattr(model, attr):
            E = getattr(model, attr)
            if torch.is_tensor(E):
                E = E.detach().cpu().numpy()
            elif hasattr(E, "weight") and torch.is_tensor(E.weight):
                E = E.weight.detach().cpu().numpy()
            else:
                try:
                    E = np.asarray(E)
                except Exception:
                    continue
            E = np.asarray(E, dtype=np.float32)
            if E.shape == (K, B):
                return E, f"model.{attr}"
            if E.shape == (B, K):
                return E.T, f"model.{attr} [transposed]"

    # Very common linear decoder: abundance K -> spectrum B.
    if hasattr(model, "decoder"):
        decoder = model.decoder
        if hasattr(decoder, "weight") and torch.is_tensor(decoder.weight):
            W = decoder.weight.detach().cpu().numpy().astype(np.float32)
            if W.shape == (B, K):
                return W.T, "model.decoder.weight [transposed]"
            if W.shape == (K, B):
                return W, "model.decoder.weight"

    # Last conservative fallback: only inspect state-dict tensors whose key name
    # clearly indicates decoder/endmember semantics and whose shape is KxB/BxK.
    candidates = []
    for key, value in model.state_dict().items():
        if not torch.is_tensor(value):
            continue
        name = key.lower()
        if not any(token in name for token in ("endmember", "decoder")):
            continue
        shape = tuple(value.shape)
        if shape in ((K, B), (B, K)):
            arr = value.detach().cpu().numpy().astype(np.float32)
            if shape == (B, K):
                arr = arr.T
            candidates.append((key, arr))

    if len(candidates) == 1:
        key, E = candidates[0]
        return E, f"state_dict['{key}']"

    raise RuntimeError(
        "Could not identify the learned KxB endmember matrix for SAD. "
        "Open train_cuprite_unmixing_reduced.py and expose the decoder spectra "
        "through model.endmembers(), or tell me the decoder parameter name."
    )


def save_metrics_figure(rows, mean_endmember_sad, out_path, n_plot=1000):
    """Save one figure with SAD, abundance RMSE and reconstruction RMSE."""
    if not rows:
        return None

    n_plot = min(int(n_plot), len(rows))
    if n_plot <= 0:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  metrics plot             : skipped (matplotlib not installed)")
        return None

    scenes = np.asarray([r["scene"] for r in rows[:n_plot]])
    abundance_rmse = np.asarray([r["abundance_rmse"] for r in rows[:n_plot]])
    reconstruction_rmse = np.asarray(
        [r["reconstruction_rmse"] for r in rows[:n_plot]]
    )
    sad = np.full(n_plot, mean_endmember_sad, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)

    axes[0].plot(scenes, sad, linewidth=1.8)
    axes[0].set_ylabel("Mean SAD (degrees)")
    axes[0].set_title("Endmember Spectral Angle Distance")
    axes[0].grid(True, alpha=0.3)
    axes[0].text(
        0.98,
        0.90,
        f"Mean SAD = {mean_endmember_sad:.4f} deg",
        transform=axes[0].transAxes,
        ha="right",
        va="top",
    )

    axes[1].plot(scenes, abundance_rmse, linewidth=1.0)
    axes[1].set_ylabel("Abundance RMSE")
    axes[1].set_title("Abundance Estimation Error")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(scenes, reconstruction_rmse, linewidth=1.0)
    axes[2].set_ylabel("Reconstruction RMSE")
    axes[2].set_xlabel("Scene")
    axes[2].set_title("Spectral Reconstruction Error")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Oreacle ML Unmixing Performance - First {n_plot} Scenes",
        fontsize=15,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------
# Oreacle demo: mineral -> alteration assemblage interpretation.
# IMPORTANT: these names must match the row order in data/endmembers.npy.
# If the training checkpoint stores names, those are preferred automatically.
# ------------------------------------------------------------------
DEFAULT_MINERAL_NAMES = [
    "Alunite",
    "Andradite",
    "Buddingtonite",
    "Dumortierite",
    "Kaolinite 1",
    "Kaolinite 2",
    "Muscovite",
    "Montmorillonite",
    "Nontronite",
    "Pyrope",
    "Sphene",
    "Chalcedony",
]

ALTERATION_CLASSES = [
    "Advanced Argillic",
    "Argillic / Smectitic",
    "Phyllic / Sericitic",
    "Silicic",
    "Early Hydrothermal",
    "Skarn / Calc-silicate",
]


# ------------------------------------------------------------------
# Oreacle processability rules.
# These are transparent geology-informed heuristics layered on top of the
# alteration map; they are NOT a replacement for metallurgical testwork.
# Risk scale: 1 = low, 2 = moderate, 3 = high.
# ------------------------------------------------------------------
PROCESSABILITY_RULES = {
    "Advanced Argillic": {
        "difficulty": 3,
        "flotation": 3,
        "rheology": 3,
        "dewatering": 2,
        "comminution": 2,
        "leaching": 2,
        "reason": "Alunite/kaolinite-rich alteration can introduce fine clay-rich gangue and variable reagent demand.",
    },
    "Argillic / Smectitic": {
        "difficulty": 3,
        "flotation": 3,
        "rheology": 3,
        "dewatering": 3,
        "comminution": 1,
        "leaching": 2,
        "reason": "Smectite/kaolinite-rich material is the strongest clay-related processing risk in the current mineral set.",
    },
    "Phyllic / Sericitic": {
        "difficulty": 2,
        "flotation": 2,
        "rheology": 2,
        "dewatering": 2,
        "comminution": 1,
        "leaching": 1,
        "reason": "Fine micaceous/sericitic gangue can affect flotation selectivity and solids handling.",
    },
    "Silicic": {
        "difficulty": 2,
        "flotation": 1,
        "rheology": 1,
        "dewatering": 1,
        "comminution": 3,
        "leaching": 1,
        "reason": "Silica-rich material is less clay-dominated, but may shift the main concern toward liberation and grinding demand.",
    },
    "Early Hydrothermal": {
        "difficulty": 2,
        "flotation": 2,
        "rheology": 2,
        "dewatering": 2,
        "comminution": 2,
        "leaching": 1,
        "reason": "Buddingtonite-smectite-silica mixtures can produce mixed handling and separation behaviour.",
    },
    "Skarn / Calc-silicate": {
        "difficulty": 2,
        "flotation": 2,
        "rheology": 1,
        "dewatering": 1,
        "comminution": 3,
        "leaching": 1,
        "reason": "Calc-silicate/garnet-rich material may require more attention to liberation and comminution.",
    },
}

PROCESSABILITY_LEVEL_NAMES = {
    0: "Unknown",
    1: "Low processing risk",
    2: "Moderate processing difficulty",
    3: "Difficult / high processing risk",
}


def _find_mineral_index(names, target):
    """Case-insensitive mineral lookup with a clear error if names do not match."""
    lookup = {str(name).strip().lower(): i for i, name in enumerate(names)}
    key = target.strip().lower()
    if key not in lookup:
        raise ValueError(
            f"Alteration mapper requires mineral '{target}', but checkpoint/endmember "
            f"names are: {list(names)}"
        )
    return lookup[key]


def map_alteration_assemblages(abundance_cube, mineral_names, min_score=0.15):
    """
    Convert ML mineral abundances (H x W x K) into an Oreacle alteration map.

    The scores are intentionally simple, interpretable heuristic assemblage scores.
    They are NOT a separately trained geological classifier.

    Returns
    -------
    label_map : (H, W) int array
        0 = Unclassified; 1..6 correspond to ALTERATION_CLASSES.
    confidence_map : (H, W) float array in [0, 1]
        Relative separation between the best and second-best assemblage score.
    score_cube : (H, W, 6) float array
        Raw alteration assemblage scores.
    """
    if abundance_cube.ndim != 3:
        raise ValueError(
            f"Expected abundance cube H x W x K, got {abundance_cube.shape}"
        )

    idx = lambda name: _find_mineral_index(mineral_names, name)

    alunite = abundance_cube[..., idx("Alunite")]
    andradite = abundance_cube[..., idx("Andradite")]
    buddingtonite = abundance_cube[..., idx("Buddingtonite")]
    kaolinite = (
        abundance_cube[..., idx("Kaolinite 1")]
        + abundance_cube[..., idx("Kaolinite 2")]
    )
    muscovite = abundance_cube[..., idx("Muscovite")]
    montmorillonite = abundance_cube[..., idx("Montmorillonite")]
    nontronite = abundance_cube[..., idx("Nontronite")]
    sphene = abundance_cube[..., idx("Sphene")]
    chalcedony = abundance_cube[..., idx("Chalcedony")]

    # Assemblage scores.  Minerals such as Pyrope and Dumortierite are retained
    # by the ML model but are not forced into a diagnostic hydrothermal class.
    advanced_argillic = (
        1.00 * alunite
        + 0.70 * kaolinite
        + 0.30 * np.minimum(alunite, kaolinite)
    )

    argillic_smectitic = (
        0.75 * kaolinite
        + 1.00 * montmorillonite
        + 0.70 * nontronite
        + 0.20 * np.minimum(kaolinite, montmorillonite + nontronite)
    )

    phyllic_sericitic = 1.00 * muscovite
    silicic = 1.00 * chalcedony

    early_hydrothermal = (
        1.00 * buddingtonite
        + 0.50 * montmorillonite
        + 0.25 * chalcedony
        + 0.25 * np.minimum(buddingtonite, montmorillonite)
    )

    skarn_calc_silicate = (
        1.00 * andradite
        + 0.50 * sphene
        + 0.25 * np.minimum(andradite, sphene)
    )

    score_cube = np.stack(
        [
            advanced_argillic,
            argillic_smectitic,
            phyllic_sericitic,
            silicic,
            early_hydrothermal,
            skarn_calc_silicate,
        ],
        axis=-1,
    ).astype(np.float32)

    best_idx = np.argmax(score_cube, axis=-1)
    best_score = np.max(score_cube, axis=-1)

    # Confidence = how clearly the best alteration separates from runner-up.
    sorted_scores = np.sort(score_cube, axis=-1)
    second_score = sorted_scores[..., -2]
    confidence_map = (best_score - second_score) / np.maximum(best_score, 1e-8)
    confidence_map = np.clip(confidence_map, 0.0, 1.0).astype(np.float32)

    # 0 is explicitly reserved for pixels with weak/non-diagnostic evidence.
    label_map = (best_idx + 1).astype(np.uint8)
    label_map[best_score < min_score] = 0
    confidence_map[best_score < min_score] = 0.0

    return label_map, confidence_map, score_cube


def save_alteration_demo(
    abundance_cube,
    mineral_names,
    scene_idx,
    out_dir,
    min_score=0.15,
):
    """Save a demo-ready Oreacle alteration map and print its interpretation."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_map, confidence_map, score_cube = map_alteration_assemblages(
        abundance_cube,
        mineral_names,
        min_score=min_score,
    )

    np.save(out_dir / f"scene_{scene_idx:04d}_mineral_abundances.npy", abundance_cube)
    np.save(out_dir / f"scene_{scene_idx:04d}_alteration_labels.npy", label_map)
    np.save(out_dir / f"scene_{scene_idx:04d}_alteration_confidence.npy", confidence_map)
    np.save(out_dir / f"scene_{scene_idx:04d}_alteration_scores.npy", score_cube)

    all_classes = ["Unclassified / weak evidence"] + ALTERATION_CLASSES
    counts = np.bincount(label_map.ravel(), minlength=len(all_classes))
    percentages = 100.0 * counts / max(label_map.size, 1)

    # Scene-level summary from mean per-pixel assemblage score.
    scene_scores = score_cube.reshape(-1, score_cube.shape[-1]).mean(axis=0)
    order = np.argsort(scene_scores)[::-1]
    primary_idx = int(order[0])
    secondary_idx = int(order[1])
    primary = ALTERATION_CLASSES[primary_idx]
    secondary = ALTERATION_CLASSES[secondary_idx]
    margin = float(scene_scores[primary_idx] - scene_scores[secondary_idx])

    if scene_scores[primary_idx] < min_score:
        scene_confidence = "LOW"
        primary_display = "Unclassified / weak alteration evidence"
    elif margin >= 0.15:
        scene_confidence = "HIGH"
        primary_display = primary
    elif margin >= 0.05:
        scene_confidence = "MODERATE"
        primary_display = primary
    else:
        scene_confidence = "LOW"
        primary_display = primary

    # Evidence: report the highest mean predicted minerals for the scene.
    mean_minerals = abundance_cube.reshape(-1, abundance_cube.shape[-1]).mean(axis=0)
    top_minerals = np.argsort(mean_minerals)[::-1][:4]
    evidence = ", ".join(
        f"{mineral_names[i]}={100.0 * mean_minerals[i]:.1f}%" for i in top_minerals
    )

    print("\n[Oreacle alteration interpretation]")
    print(f"  mapped scene             : {scene_idx}")
    print(f"  primary alteration       : {primary_display}")
    print(f"  confidence               : {scene_confidence}")
    print(f"  secondary interpretation : {secondary}")
    print(f"  mineral evidence         : {evidence}")
    print("  spatial coverage:")
    for name, pct in zip(all_classes, percentages):
        if pct >= 0.1:
            print(f"    {name:<27}: {pct:6.2f}%")

    # Demo PNG. Keep plotting optional so benchmark inference still runs even if
    # matplotlib is unavailable on the machine.
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch

        colors = [
            "#6b7280",  # unclassified
            "#dc2626",  # advanced argillic
            "#f59e0b",  # argillic/smectitic
            "#7c3aed",  # phyllic/sericitic
            "#2563eb",  # silicic
            "#06b6d4",  # early hydrothermal
            "#16a34a",  # skarn/calc-silicate
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(label_map, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"Oreacle Alteration Assemblage Map - Scene {scene_idx}")
        ax.set_xlabel("Conveyor width (pixels)")
        ax.set_ylabel("Scene height (pixels)")

        legend = [
            Patch(facecolor=colors[i], label=all_classes[i])
            for i in range(len(all_classes))
        ]
        ax.legend(
            handles=legend,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout()

        png_path = out_dir / f"scene_{scene_idx:04d}_alteration_map.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  alteration map PNG       : {png_path.resolve()}")
    except ImportError:
        print("  alteration map PNG       : skipped (matplotlib not installed)")

    return label_map, confidence_map, score_cube


def _risk_level_from_score(score):
    """Convert a 1..3 weighted process risk score to a demo-friendly label."""
    if score >= 2.5:
        return 3, "HIGH"
    if score >= 1.5:
        return 2, "MODERATE"
    if score > 0:
        return 1, "LOW"
    return 0, "UNKNOWN"


def map_processability(label_map):
    """
    Convert the alteration label map into a per-pixel processability map.

    Returns
    -------
    process_map : (H, W) uint8
        0 unknown, 1 low risk, 2 moderate, 3 difficult/high risk.
    risk_maps : dict[str, (H,W) float32]
        Per-pixel risk layers for flotation, rheology, dewatering,
        comminution and leaching.
    """
    if label_map.ndim != 2:
        raise ValueError(f"Expected 2-D alteration label map, got {label_map.shape}")

    process_map = np.zeros(label_map.shape, dtype=np.uint8)
    risk_maps = {
        name: np.zeros(label_map.shape, dtype=np.float32)
        for name in ["flotation", "rheology", "dewatering", "comminution", "leaching"]
    }

    for label_id, alteration_name in enumerate(ALTERATION_CLASSES, start=1):
        mask = label_map == label_id
        rule = PROCESSABILITY_RULES[alteration_name]
        process_map[mask] = int(rule["difficulty"])
        for risk_name in risk_maps:
            risk_maps[risk_name][mask] = float(rule[risk_name])

    return process_map, risk_maps


def save_processability_demo(label_map, confidence_map, scene_idx, out_dir):
    """
    Save and print the Oreacle scene-level processability prediction.

    This is deliberately a rule-based inference from alteration mineralogy.
    It predicts likely processing behaviour/risks; it does not claim measured
    recovery, Bond work index, reagent consumption or plant performance.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    process_map, risk_maps = map_processability(label_map)
    np.save(out_dir / f"scene_{scene_idx:04d}_processability_labels.npy", process_map)
    for risk_name, risk_map in risk_maps.items():
        np.save(out_dir / f"scene_{scene_idx:04d}_{risk_name}_risk.npy", risk_map)

    classified = label_map > 0
    classified_fraction = float(np.mean(classified)) if label_map.size else 0.0

    # Use classified pixels only. Unclassified pixels reduce confidence instead
    # of making the ore appear easier to process.
    if np.any(classified):
        overall_score = float(np.mean(process_map[classified]))
        risk_scores = {
            name: float(np.mean(risk_map[classified]))
            for name, risk_map in risk_maps.items()
        }
        mean_mapping_confidence = float(np.mean(confidence_map[classified]))
    else:
        overall_score = 0.0
        risk_scores = {name: 0.0 for name in risk_maps}
        mean_mapping_confidence = 0.0

    process_level, process_label = _risk_level_from_score(overall_score)

    # Find the alteration phase contributing most to the processability result.
    counts = np.bincount(label_map.ravel(), minlength=len(ALTERATION_CLASSES) + 1)
    alteration_percentages = 100.0 * counts / max(label_map.size, 1)

    driver_scores = []
    for label_id, alteration_name in enumerate(ALTERATION_CLASSES, start=1):
        coverage_fraction = counts[label_id] / max(label_map.size, 1)
        difficulty = PROCESSABILITY_RULES[alteration_name]["difficulty"]
        driver_scores.append(coverage_fraction * difficulty)

    if max(driver_scores, default=0.0) > 0:
        driver_idx = int(np.argmax(driver_scores))
        driver_name = ALTERATION_CLASSES[driver_idx]
        driver_reason = PROCESSABILITY_RULES[driver_name]["reason"]
    else:
        driver_name = "Unclassified / weak alteration evidence"
        driver_reason = (
            "The scene does not contain enough diagnostic alteration evidence "
            "for a reliable processability inference."
        )

    combined_conf = classified_fraction * mean_mapping_confidence
    if combined_conf >= 0.45:
        prediction_confidence = "HIGH"
    elif combined_conf >= 0.20:
        prediction_confidence = "MODERATE"
    else:
        prediction_confidence = "LOW"

    # Operational considerations are intentionally phrased as things to evaluate,
    # rather than claiming that mineralogy alone determines a final plant flowsheet.
    considerations = []
    if risk_scores["rheology"] >= 2.2:
        considerations.append(
            "Evaluate clay/rheology management and desliming before downstream separation."
        )
    if risk_scores["flotation"] >= 2.2:
        considerations.append(
            "Expect flotation sensitivity; evaluate slime coating, entrainment and reagent conditions."
        )
    if risk_scores["dewatering"] >= 2.2:
        considerations.append(
            "Plan thickening/filtration testwork for fine or clay-rich material."
        )
    if risk_scores["comminution"] >= 2.2:
        considerations.append(
            "Check grindability and liberation; grinding demand may be elevated."
        )
    if risk_scores["leaching"] >= 2.2:
        considerations.append(
            "Validate permeability and reagent/acid consumption with leach testwork."
        )
    if not considerations:
        considerations.append(
            "No single severe alteration-driven risk dominates; validate the conventional flowsheet with metallurgical testwork."
        )

    print("\n[Oreacle processability prediction]")
    print(f"  mapped scene             : {scene_idx}")
    print(f"  predicted processability : {PROCESSABILITY_LEVEL_NAMES[process_level]}")
    print(f"  process risk score       : {overall_score:.2f} / 3.00")
    print(f"  prediction confidence    : {prediction_confidence}")
    print(f"  classified scene area    : {100.0 * classified_fraction:.2f}%")
    print(f"  main alteration driver   : {driver_name}")
    print(f"  why it matters           : {driver_reason}")
    print("  risk profile (1=low, 2=moderate, 3=high):")
    for name, score in risk_scores.items():
        _, label = _risk_level_from_score(score)
        print(f"    {name:<12}: {score:.2f}/3  ({label})")
    print("  processing considerations:")
    for item in considerations:
        print(f"    - {item}")

    summary = {
        "scene": int(scene_idx),
        "predicted_processability": PROCESSABILITY_LEVEL_NAMES[process_level],
        "process_risk_score_1_to_3": overall_score,
        "prediction_confidence": prediction_confidence,
        "classified_scene_fraction": classified_fraction,
        "main_alteration_driver": driver_name,
        "driver_reason": driver_reason,
        "risk_profile_1_to_3": risk_scores,
        "processing_considerations": considerations,
        "alteration_coverage_percent": {
            ALTERATION_CLASSES[i - 1]: float(alteration_percentages[i])
            for i in range(1, len(ALTERATION_CLASSES) + 1)
        },
    }

    json_path = out_dir / f"scene_{scene_idx:04d}_processability_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.patches import Patch

        colors = ["#6b7280", "#22c55e", "#f59e0b", "#dc2626"]
        names = [
            "Unknown",
            "Low processing risk",
            "Moderate processing difficulty",
            "Difficult / high processing risk",
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)

        fig, ax = plt.subplots(figsize=(10, 7))
        ax.imshow(process_map, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"Oreacle Predicted Ore Processability - Scene {scene_idx}")
        ax.set_xlabel("Conveyor width (pixels)")
        ax.set_ylabel("Scene height (pixels)")
        legend = [Patch(facecolor=colors[i], label=names[i]) for i in range(4)]
        ax.legend(
            handles=legend,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
        )
        fig.tight_layout()

        png_path = out_dir / f"scene_{scene_idx:04d}_processability_map.png"
        fig.savefig(png_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"  processability map PNG   : {png_path.resolve()}")
    except ImportError:
        print("  processability map PNG   : skipped (matplotlib not installed)")

    print(f"  processability summary   : {json_path.resolve()}")
    return process_map, risk_maps, summary


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

    # Performance figure: fixed trained-model SAD + per-scene errors.
    ap.add_argument(
        "--metrics-plot-scenes",
        type=int,
        default=1000,
        help="Number of initial scenes to include in the 3-panel metrics figure.",
    )
    ap.add_argument(
        "--metrics-plot",
        default=str(ROOT / "results" / "ml" / "ml_first_1000_metrics.png"),
        help="Path for the SAD / abundance-RMSE / reconstruction-RMSE figure.",
    )

    # Oreacle demo output: map one scene spatially instead of writing 3000 PNGs.
    ap.add_argument(
        "--alteration-map-scene",
        type=int,
        default=0,
        help="Scene index to save as a mineral/alteration map; use -1 to disable.",
    )
    ap.add_argument(
        "--alteration-outdir",
        default=str(ROOT / "results" / "ml" / "alteration_demo"),
    )
    ap.add_argument(
        "--alteration-min-score",
        type=float,
        default=0.15,
        help="Minimum assemblage score before a pixel is called altered.",
    )

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

    # Prefer names saved with the model, but support the current 12-mineral model.
    mineral_names = checkpoint.get(
        "endmember_names",
        checkpoint.get("mineral_names", DEFAULT_MINERAL_NAMES),
    )
    mineral_names = [str(name) for name in mineral_names]
    if len(mineral_names) != K:
        raise ValueError(
            f"Need exactly {K} mineral names for alteration mapping, got "
            f"{len(mineral_names)}: {mineral_names}"
        )

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
    # Fixed trained-model endmember SAD.
    # The ML endmembers do not change scene-by-scene during inference, so this
    # value is plotted as a horizontal line across the first N scenes.
    # ------------------------------------------------------------------
    learned_endmembers, endmember_source = extract_learned_endmembers(
        model, checkpoint, K, B
    )
    sad_per_endmember, mean_endmember_sad = compute_endmember_sad(
        endm, learned_endmembers
    )

    print("\n[ML endmember SAD]")
    print(f"  learned endmember source: {endmember_source}")
    for k, sad_value in enumerate(sad_per_endmember):
        mineral = mineral_names[k] if k < len(mineral_names) else f"Endmember {k}"
        print(f"  {mineral:<20}: {sad_value:.4f} deg")
    print(f"  mean SAD             : {mean_endmember_sad:.4f} deg")

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

    demo_abundance_cube = None
    demo_scene_idx = None

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

        # Only retain predictions for the selected demo scene. This keeps memory
        # usage small during long 3000-scene benchmark runs.
        capture_demo = (t == args.alteration_map_scene)
        demo_batches = [] if capture_demo else None

        with torch.inference_mode():
            for start in range(0, N, args.batch_size):
                stop = min(start + args.batch_size, N)

                z_batch = torch.from_numpy(
                    np.ascontiguousarray(Z[start:stop])
                ).to(device)

                a_pred, x_hat = model(z_batch)

                if capture_demo:
                    demo_batches.append(a_pred.detach().cpu().numpy())

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

        if capture_demo:
            A_pred_scene = np.concatenate(demo_batches, axis=0).astype(
                np.float32, copy=False
            )
            if A_pred_scene.shape != (N, K):
                raise ValueError(
                    f"Expected captured abundance shape {(N, K)}, "
                    f"got {A_pred_scene.shape}"
                )
            demo_abundance_cube = A_pred_scene.reshape(H, W_fix, K)
            demo_scene_idx = t

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

    metrics_plot_path = save_metrics_figure(
        rows,
        mean_endmember_sad,
        args.metrics_plot,
        n_plot=args.metrics_plot_scenes,
    )

    print("\n[ml-conveyor summary]")
    print(f"  scenes                  : {len(rows)}")
    print(f"  pixels                  : {total_pixels:,}")
    print(f"  wall time               : {wall_seconds:.3f}s")
    print(f"  mean reduction / scene  : {t_reduce.mean()*1000:.3f} ms")
    print(f"  mean inference / scene  : {t_infer.mean()*1000:.3f} ms")
    print(f"  mean ML total / scene   : {t_total.mean()*1000:.3f} ms")
    print(f"  median ML total / scene : {np.median(t_total)*1000:.3f} ms")
    print(f"  mean endmember SAD      : {mean_endmember_sad:.6f} deg")
    print(f"  mean abundance RMSE     : {armse.mean():.6f}")
    print(f"  mean reconstruction RMSE: {rrmse.mean():.6f}")
    print(f"  saved                   : {results_path.resolve()}")
    if metrics_plot_path is not None:
        print(f"  metrics figure          : {metrics_plot_path.resolve()}")

    # ------------------------------------------------------------------
    # Oreacle demo: mineral abundances -> alteration assemblage map.
    # ------------------------------------------------------------------
    if args.alteration_map_scene >= 0:
        if demo_abundance_cube is None:
            print(
                f"\n[Oreacle alteration] requested scene "
                f"{args.alteration_map_scene}, but only scenes "
                f"0..{args.n_scenes - 1} were processed."
            )
        else:
            label_map, confidence_map, score_cube = save_alteration_demo(
                demo_abundance_cube,
                mineral_names,
                demo_scene_idx,
                args.alteration_outdir,
                min_score=args.alteration_min_score,
            )

            # Final Oreacle requirement:
            # alteration assemblages -> predicted ore processability.
            save_processability_demo(
                label_map,
                confidence_map,
                demo_scene_idx,
                args.alteration_outdir,
            )


if __name__ == "__main__":
    main()
