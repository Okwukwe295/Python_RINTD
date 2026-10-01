#!/usr/bin/env python3
"""
Oreacle - Interactive 50-scene conveyor demo
=============================================

Purpose
-------
Judge-facing Streamlit wrapper around the existing Oreacle pipeline.

It DOES NOT retrain the model and DOES NOT replace the existing algorithms.
It reuses:
    - baseline/sp_conveyor.py -> synthetic scene generation
    - ml_unmixing/train_cuprite_unmixing_reduced.py -> trained AE definition
    - ml_unmixing/run_ml_conveyor_oreacle_final.py -> alteration + processability logic
    - results/ml/cuprite_reduced_unmixing_model.pt -> trained model
    - data/endmembers.npy -> 12-mineral spectral library

Demo flow
---------
Synthetic conveyor scene
    -> Tucker spectral reduction
    -> trained autoencoder inference
    -> 12 mineral abundance maps
    -> alteration assemblage map
    -> processability prediction
    -> recommended processing action / routing decision

Run
---
From the project root:

    pip install streamlit matplotlib
    streamlit run demo/oreacle_demo.py

The demo uses a staged 50-scene synthetic conveyor sequence so judges can see
different ore conditions. The stage label is used ONLY to choose which known
endmember spectra are mixed to create the synthetic sensor input. It is never
passed to the trained ML model as a label.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve()

# Expected location: <project>/demo/oreacle_demo.py
if (HERE.parent.parent / "ml_unmixing").exists():
    ROOT = HERE.parent.parent
elif (HERE.parent / "ml_unmixing").exists():
    # Also allow the file to live directly in the project root.
    ROOT = HERE.parent
else:
    ROOT = HERE.parent.parent

ML_DIR = ROOT / "ml_unmixing"
BASELINE_DIR = ROOT / "baseline"

PIPELINE_FILE = ML_DIR / "run_ml_conveyor_oreacle_final.py"
TRAIN_FILE = ML_DIR / "train_cuprite_unmixing_reduced.py"
BASELINE_FILE = BASELINE_DIR / "sp_conveyor.py"

MODEL_FILE = ROOT / "results" / "ml" / "cuprite_reduced_unmixing_model.pt"
ENDMEMBER_FILE = ROOT / "data" / "endmembers.npy"

DEMO_RESULTS_DIR = ROOT / "results" / "ml" / "oreacle_live_demo"


# ---------------------------------------------------------------------
# Demo settings
# ---------------------------------------------------------------------
N_DEMO_SCENES = 50
DEMO_HEIGHT = 64
DEMO_WIDTH = 96
DEMO_SEED = 20261001
DEFAULT_BATCH_SIZE = 8192
DEFAULT_MIN_SCORE = 0.15

# These are the 12 minerals used by the current Oreacle model.
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


# ---------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module '{name}' from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_project_files():
    required = {
        "Oreacle pipeline": PIPELINE_FILE,
        "Training/model definition": TRAIN_FILE,
        "Conveyor generator": BASELINE_FILE,
        "Trained model": MODEL_FILE,
        "Endmember library": ENDMEMBER_FILE,
    }

    missing = {name: path for name, path in required.items() if not path.exists()}
    if missing:
        lines = ["Missing required project files:"]
        for name, path in missing.items():
            lines.append(f"  - {name}: {path}")
        raise FileNotFoundError("\n".join(lines))


# ---------------------------------------------------------------------
# Load trained model only once.
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained Oreacle encoder...")
def load_oreacle():
    validate_project_files()

    pipeline = load_module("oreacle_pipeline_demo", PIPELINE_FILE)
    baseline = load_module("oreacle_sp_conveyor_demo", BASELINE_FILE)
    train_mod = load_module("oreacle_train_mod_demo", TRAIN_FILE)

    generate_scene = baseline.generate_scene
    ReducedInputUnmixingAE = train_mod.ReducedInputUnmixingAE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=False,
    )

    R = int(checkpoint["input_dim"])
    B = int(checkpoint["n_bands"])
    K = int(checkpoint["n_endmembers"])

    mineral_names = checkpoint.get(
        "endmember_names",
        checkpoint.get("mineral_names", DEFAULT_MINERAL_NAMES),
    )
    mineral_names = [str(x) for x in mineral_names]

    if len(mineral_names) != K:
        raise ValueError(
            f"Model expects {K} minerals but found {len(mineral_names)} names."
        )

    model = ReducedInputUnmixingAE(R, B, K).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    U3 = np.asarray(checkpoint["basis_U3"], dtype=np.float32)
    z_mean = np.asarray(checkpoint["z_mean"], dtype=np.float32)
    z_std = np.asarray(checkpoint["z_std"], dtype=np.float32)
    U3_pinv = np.linalg.pinv(U3).astype(np.float32)

    endm = np.load(ENDMEMBER_FILE).astype(np.float32)
    if endm.ndim != 2 or endm.shape != (K, B):
        raise ValueError(
            f"Expected endmembers.npy shape {(K, B)}, got {endm.shape}"
        )

    # B x K: each column is one mineral spectrum.
    E_pool = endm.T

    return {
        "pipeline": pipeline,
        "generate_scene": generate_scene,
        "device": device,
        "checkpoint": checkpoint,
        "model": model,
        "R": R,
        "B": B,
        "K": K,
        "U3_pinv": U3_pinv,
        "z_mean": z_mean,
        "z_std": z_std,
        "endm": endm,
        "E_pool": E_pool,
        "mineral_names": mineral_names,
    }


# ---------------------------------------------------------------------
# 50-scene demonstration plan
#
# Important:
# These profile names are NOT given to the model.
# They only select which known mineral spectra are mixed by the synthetic
# conveyor generator. The model receives only the generated spectral cube.
# ---------------------------------------------------------------------
def build_demo_plan(mineral_names):
    name_to_idx = {name.lower(): i for i, name in enumerate(mineral_names)}

    def ids(*names):
        found = []
        for name in names:
            key = name.lower()
            if key not in name_to_idx:
                raise ValueError(f"Demo plan mineral '{name}' not found.")
            found.append(name_to_idx[key])
        return found

    stages = [
        {
            "start": 0,
            "stop": 9,
            "profile": "Silica-rich segment",
            "minerals": ids(
                "Chalcedony",
                "Muscovite",
                "Kaolinite 1",
            ),
            "snr": (34.0, 44.0),
            "alpha": (0.7, 1.5),
        },
        {
            "start": 10,
            "stop": 19,
            "profile": "Advanced-argillic segment",
            "minerals": ids(
                "Alunite",
                "Kaolinite 1",
                "Kaolinite 2",
                "Chalcedony",
            ),
            "snr": (31.0, 42.0),
            "alpha": (0.45, 1.25),
        },
        {
            "start": 20,
            "stop": 29,
            "profile": "Clay-rich argillic segment",
            "minerals": ids(
                "Kaolinite 1",
                "Kaolinite 2",
                "Montmorillonite",
                "Nontronite",
            ),
            "snr": (29.0, 40.0),
            "alpha": (0.35, 1.0),
        },
        {
            "start": 30,
            "stop": 39,
            "profile": "Skarn / calc-silicate segment",
            "minerals": ids(
                "Andradite",
                "Sphene",
                "Pyrope",
            ),
            "snr": (33.0, 44.0),
            "alpha": (0.6, 1.4),
        },
        {
            "start": 40,
            "stop": 44,
            "profile": "Early-hydrothermal segment",
            "minerals": ids(
                "Buddingtonite",
                "Montmorillonite",
                "Chalcedony",
            ),
            "snr": (30.0, 41.0),
            "alpha": (0.45, 1.1),
        },
        {
            "start": 45,
            "stop": 49,
            "profile": "Mixed ore segment",
            "minerals": list(range(len(mineral_names))),
            "snr": (27.0, 43.0),
            "alpha": (0.3, 1.8),
        },
    ]

    plan = []
    for scene_idx in range(N_DEMO_SCENES):
        stage = next(
            s for s in stages
            if s["start"] <= scene_idx <= s["stop"]
        )

        # Independent deterministic RNG per scene.
        rng = np.random.default_rng(DEMO_SEED + scene_idx)
        snr = float(rng.uniform(*stage["snr"]))
        alpha = float(rng.uniform(*stage["alpha"]))

        plan.append(
            {
                "scene": scene_idx,
                "profile": stage["profile"],
                "mineral_indices": list(stage["minerals"]),
                "snr_db": snr,
                "alpha": alpha,
                "seed": DEMO_SEED + scene_idx,
            }
        )

    return plan


# ---------------------------------------------------------------------
# Scene inference
# ---------------------------------------------------------------------
def run_scene(scene_idx: int, batch_size=DEFAULT_BATCH_SIZE):
    env = load_oreacle()
    plan = build_demo_plan(env["mineral_names"])
    spec = plan[scene_idx]

    model = env["model"]
    device = env["device"]
    K = env["K"]
    B = env["B"]

    rng = np.random.default_rng(spec["seed"])

    # Only these minerals are used to synthesize this test section of the belt.
    active = spec["mineral_indices"]
    E_true_local = env["E_pool"][:, active].copy()

    X_cube, A_true_local, _ = env["generate_scene"](
        E_true_local,
        DEMO_HEIGHT,
        DEMO_WIDTH,
        spec["snr_db"],
        spec["alpha"],
        rng,
    )

    X_flat = X_cube.reshape(-1, B).astype(np.float32, copy=False)
    N = X_flat.shape[0]

    # Map synthetic ground truth back into the global 12-mineral ordering.
    A_true_global = np.zeros((N, K), dtype=np.float32)
    for local_idx, global_idx in enumerate(active):
        A_true_global[:, global_idx] = A_true_local[local_idx]

    # Tucker spectral reduction.
    t0 = time.perf_counter()
    Z = X_flat @ env["U3_pinv"].T
    Z = (Z - env["z_mean"]) / env["z_std"]
    t_reduce = time.perf_counter() - t0

    pred_batches = []
    abund_sqerr = 0.0
    recon_sqerr = 0.0
    abund_count = 0
    recon_count = 0

    t0 = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)

            z_batch = torch.from_numpy(
                np.ascontiguousarray(Z[start:stop])
            ).to(device)

            a_pred, x_hat = model(z_batch)

            pred_batches.append(a_pred.detach().cpu().numpy())

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

    if device.type == "cuda":
        torch.cuda.synchronize()

    t_infer = time.perf_counter() - t0
    t_total = t_reduce + t_infer

    A_pred = np.concatenate(pred_batches, axis=0).astype(np.float32)
    abundance_cube = A_pred.reshape(DEMO_HEIGHT, DEMO_WIDTH, K)

    abundance_rmse = float(
        np.sqrt(abund_sqerr / max(abund_count, 1))
    )
    reconstruction_rmse = float(
        np.sqrt(recon_sqerr / max(recon_count, 1))
    )

    # Alteration inference from predicted abundance maps.
    label_map, confidence_map, score_cube = (
        env["pipeline"].map_alteration_assemblages(
            abundance_cube,
            env["mineral_names"],
            min_score=DEFAULT_MIN_SCORE,
        )
    )

    alteration_summary = summarize_alteration(
        abundance_cube,
        label_map,
        confidence_map,
        score_cube,
        env["mineral_names"],
        env["pipeline"].ALTERATION_CLASSES,
    )

    process_map, risk_maps = env["pipeline"].map_processability(label_map)

    process_summary = summarize_processability(
        label_map,
        confidence_map,
        process_map,
        risk_maps,
        env["pipeline"].ALTERATION_CLASSES,
        env["pipeline"].PROCESSABILITY_RULES,
        env["pipeline"].PROCESSABILITY_LEVEL_NAMES,
    )

    result = {
        "scene_idx": scene_idx,
        "profile": spec["profile"],
        "active_indices": active,
        "active_names": [env["mineral_names"][i] for i in active],
        "snr_db": spec["snr_db"],
        "alpha": spec["alpha"],
        "X_cube": X_cube,
        "A_true_global": A_true_global.reshape(
            DEMO_HEIGHT, DEMO_WIDTH, K
        ),
        "abundance_cube": abundance_cube,
        "label_map": label_map,
        "confidence_map": confidence_map,
        "score_cube": score_cube,
        "process_map": process_map,
        "risk_maps": risk_maps,
        "alteration": alteration_summary,
        "processability": process_summary,
        "abundance_rmse": abundance_rmse,
        "reconstruction_rmse": reconstruction_rmse,
        "t_reduce": t_reduce,
        "t_infer": t_infer,
        "t_total": t_total,
        "pixels_per_second": N / max(t_total, 1e-12),
    }

    # Print a proper terminal output too.
    print(console_report(result))

    return result


# ---------------------------------------------------------------------
# Scene summaries
# ---------------------------------------------------------------------
def summarize_alteration(
    abundance_cube,
    label_map,
    confidence_map,
    score_cube,
    mineral_names,
    alteration_classes,
):
    mean_minerals = abundance_cube.reshape(
        -1, abundance_cube.shape[-1]
    ).mean(axis=0)

    top_idx = np.argsort(mean_minerals)[::-1]

    scene_scores = score_cube.reshape(
        -1, score_cube.shape[-1]
    ).mean(axis=0)

    score_order = np.argsort(scene_scores)[::-1]
    primary_idx = int(score_order[0])
    secondary_idx = int(score_order[1])

    primary = alteration_classes[primary_idx]
    secondary = alteration_classes[secondary_idx]

    counts = np.bincount(
        label_map.ravel(),
        minlength=len(alteration_classes) + 1,
    )
    coverage = {
        alteration_classes[i - 1]: float(
            100.0 * counts[i] / max(label_map.size, 1)
        )
        for i in range(1, len(alteration_classes) + 1)
    }

    classified = label_map > 0
    classified_fraction = float(np.mean(classified))

    if np.any(classified):
        mean_conf = float(np.mean(confidence_map[classified]))
    else:
        mean_conf = 0.0

    combined = classified_fraction * mean_conf

    if combined >= 0.45:
        confidence = "HIGH"
    elif combined >= 0.20:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    return {
        "primary": primary,
        "secondary": secondary,
        "confidence": confidence,
        "coverage_percent": coverage,
        "mean_mineral_abundance": {
            mineral_names[i]: float(mean_minerals[i])
            for i in range(len(mineral_names))
        },
        "top_minerals": [
            (mineral_names[i], float(mean_minerals[i]))
            for i in top_idx[:5]
        ],
    }


def summarize_processability(
    label_map,
    confidence_map,
    process_map,
    risk_maps,
    alteration_classes,
    rules,
    level_names,
):
    classified = label_map > 0
    classified_fraction = float(np.mean(classified))

    if np.any(classified):
        overall_score = float(np.mean(process_map[classified]))
        risk_scores = {
            name: float(np.mean(risk_map[classified]))
            for name, risk_map in risk_maps.items()
        }
        mean_mapping_conf = float(
            np.mean(confidence_map[classified])
        )
    else:
        overall_score = 0.0
        risk_scores = {name: 0.0 for name in risk_maps}
        mean_mapping_conf = 0.0

    if overall_score >= 2.5:
        level = 3
    elif overall_score >= 1.5:
        level = 2
    elif overall_score > 0:
        level = 1
    else:
        level = 0

    counts = np.bincount(
        label_map.ravel(),
        minlength=len(alteration_classes) + 1,
    )

    driver_scores = []
    for label_id, alteration_name in enumerate(
        alteration_classes, start=1
    ):
        fraction = counts[label_id] / max(label_map.size, 1)
        difficulty = rules[alteration_name]["difficulty"]
        driver_scores.append(fraction * difficulty)

    if max(driver_scores, default=0.0) > 0:
        driver_idx = int(np.argmax(driver_scores))
        driver = alteration_classes[driver_idx]
        reason = rules[driver]["reason"]
    else:
        driver = "Unclassified / weak alteration evidence"
        reason = (
            "Insufficient diagnostic alteration evidence for "
            "a strong processability prediction."
        )

    combined_conf = classified_fraction * mean_mapping_conf
    if combined_conf >= 0.45:
        confidence = "HIGH"
    elif combined_conf >= 0.20:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    route, action, yield_logic = choose_processing_route(
        risk_scores,
        overall_score,
        driver,
    )

    considerations = []

    if risk_scores["rheology"] >= 2.2:
        considerations.append(
            "Manage clay/slurry rheology and evaluate desliming."
        )
    if risk_scores["flotation"] >= 2.2:
        considerations.append(
            "Use flotation conditioning and optimise reagent conditions."
        )
    if risk_scores["dewatering"] >= 2.2:
        considerations.append(
            "Plan for thickening/filtration of fine or clay-rich material."
        )
    if risk_scores["comminution"] >= 2.2:
        considerations.append(
            "Check grindability and liberation before downstream separation."
        )
    if risk_scores["leaching"] >= 2.2:
        considerations.append(
            "Validate permeability and reagent/acid demand with leach testwork."
        )

    if not considerations:
        considerations.append(
            "No severe alteration-driven processing risk dominates; "
            "continue on the standard route with monitoring."
        )

    return {
        "label": level_names[level],
        "level": level,
        "score": overall_score,
        "confidence": confidence,
        "classified_fraction": classified_fraction,
        "driver": driver,
        "reason": reason,
        "risk_scores": risk_scores,
        "route": route,
        "action": action,
        "yield_logic": yield_logic,
        "considerations": considerations,
    }


def choose_processing_route(risk_scores, overall_score, driver):
    """
    Demo-level operational routing rule.

    This is intentionally a process recommendation, not a claim that HSI alone
    determines the final metallurgical flowsheet.
    """
    rheology = risk_scores["rheology"]
    flotation = risk_scores["flotation"]
    dewatering = risk_scores["dewatering"]
    comminution = risk_scores["comminution"]
    leaching = risk_scores["leaching"]

    if max(rheology, flotation, dewatering) >= 2.5:
        return (
            "CLAY-MANAGEMENT / FLOTATION-CONDITIONING ROUTE",
            "Deslime where appropriate, control slurry conditions, "
            "then feed controlled flotation.",
            "Avoid sending clay-rich ore through unchanged plant settings; "
            "conditioning can reduce the risk of poor selectivity, entrainment "
            "and recovery losses.",
        )

    if comminution >= 2.5:
        return (
            "COMMINUTION / LIBERATION-CONTROL ROUTE",
            "Prioritise controlled crushing and grinding, then verify liberation "
            "before separation.",
            "Adapt grinding effort to harder or silica/calc-silicate-rich ore "
            "instead of over- or under-grinding the whole feed.",
        )

    if leaching >= 2.5:
        return (
            "LEACH-CONTROL ROUTE",
            "Validate permeability and reagent demand, then adjust leach conditions.",
            "Route material according to expected leach behaviour instead of "
            "using one reagent/permeability assumption for all ore.",
        )

    if overall_score >= 2.5:
        return (
            "HIGH-RISK ORE HOLD / CONDITIONING ROUTE",
            "Flag the material for controlled conditioning before normal processing.",
            "Prevent a difficult ore parcel from destabilising downstream recovery.",
        )

    if overall_score >= 1.5:
        return (
            "STANDARD ROUTE + ADAPTIVE CONTROL",
            f"Continue processing while adapting settings for {driver}.",
            "Use Oreacle as feed-forward information so plant settings can respond "
            "before the ore reaches the next unit operation.",
        )

    return (
        "STANDARD PROCESSING ROUTE",
        "Continue on the normal route with routine monitoring.",
        "Avoid unnecessary intervention when alteration-driven risk is low.",
    )


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def fig_abundance_map(abundance_cube, mineral_names, mineral_name):
    idx = mineral_names.index(mineral_name)
    data = abundance_cube[..., idx]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    im = ax.imshow(
        data,
        vmin=0,
        vmax=max(0.35, float(np.percentile(data, 99))),
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(f"{mineral_name} predicted abundance")
    ax.set_xlabel("Conveyor width")
    ax.set_ylabel("Pushbroom scan direction")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Predicted abundance")
    fig.tight_layout()
    return fig


def fig_dominant_mineral_map(abundance_cube, mineral_names):
    dominant = np.argmax(abundance_cube, axis=-1)

    cmap = plt.get_cmap("tab20", len(mineral_names))

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.imshow(
        dominant,
        cmap=cmap,
        vmin=-0.5,
        vmax=len(mineral_names) - 0.5,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Dominant predicted mineral")
    ax.set_xlabel("Conveyor width")
    ax.set_ylabel("Pushbroom scan direction")

    handles = [
        Patch(
            facecolor=cmap(i),
            label=mineral_names[i],
        )
        for i in range(len(mineral_names))
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=7,
        borderaxespad=0,
    )

    fig.tight_layout()
    return fig


def fig_alteration_map(label_map, alteration_classes):
    names = ["Unclassified"] + alteration_classes
    colors = [
        "#6b7280",
        "#dc2626",
        "#f59e0b",
        "#7c3aed",
        "#2563eb",
        "#06b6d4",
        "#16a34a",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(colors) + 0.5, 1),
        cmap.N,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.imshow(
        label_map,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Predicted alteration assemblage")
    ax.set_xlabel("Conveyor width")
    ax.set_ylabel("Pushbroom scan direction")

    handles = [
        Patch(facecolor=colors[i], label=names[i])
        for i in range(len(names))
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        borderaxespad=0,
    )

    fig.tight_layout()
    return fig


def fig_processability_map(process_map):
    names = [
        "Unknown",
        "Low risk",
        "Moderate",
        "Difficult / high risk",
    ]
    colors = [
        "#6b7280",
        "#22c55e",
        "#f59e0b",
        "#dc2626",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, 4.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.imshow(
        process_map,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title("Predicted processability")
    ax.set_xlabel("Conveyor width")
    ax.set_ylabel("Pushbroom scan direction")

    handles = [
        Patch(facecolor=colors[i], label=names[i])
        for i in range(4)
    ]

    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=8,
        borderaxespad=0,
    )

    fig.tight_layout()
    return fig


def fig_mean_abundances(mean_abundance, mineral_names):
    order = np.argsort(mean_abundance)[::-1]
    names = [mineral_names[i] for i in order]
    values = 100.0 * mean_abundance[order]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    ax.bar(names, values)
    ax.set_ylabel("Mean predicted abundance (%)")
    ax.set_title("Scene-average mineral abundance")
    ax.tick_params(axis="x", rotation=55)
    fig.tight_layout()
    return fig


def fig_risk_profile(risk_scores):
    names = [
        "Flotation",
        "Rheology",
        "Dewatering",
        "Comminution",
        "Leaching",
    ]
    values = [
        risk_scores["flotation"],
        risk_scores["rheology"],
        risk_scores["dewatering"],
        risk_scores["comminution"],
        risk_scores["leaching"],
    ]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.bar(names, values)
    ax.set_ylim(0, 3.1)
    ax.set_ylabel("Predicted risk (1 low - 3 high)")
    ax.set_title("Ore processability risk profile")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# Terminal / judge-facing report
# ---------------------------------------------------------------------
def console_report(result):
    alt = result["alteration"]
    proc = result["processability"]

    mineral_lines = "\n".join(
        f"    {name:<20} {100.0 * value:6.2f}%"
        for name, value in alt["top_minerals"]
    )

    risk_lines = "\n".join(
        f"    {name:<12} {score:.2f}/3"
        for name, score in proc["risk_scores"].items()
    )

    consideration_lines = "\n".join(
        f"    - {item}" for item in proc["considerations"]
    )

    return f"""
============================================================
OREACLE - REAL-TIME ORE CHARACTERISATION
============================================================

Scene                  : {result['scene_idx'] + 1:02d} / {N_DEMO_SCENES}
Synthetic belt segment : {result['profile']}
Sensor cube             : {DEMO_HEIGHT} x {DEMO_WIDTH} x {result['X_cube'].shape[-1]}

MODEL TIMING
------------------------------------------------------------
Tucker reduction        : {result['t_reduce'] * 1000:8.2f} ms
ML inference            : {result['t_infer'] * 1000:8.2f} ms
Total ML pipeline       : {result['t_total'] * 1000:8.2f} ms
Throughput              : {result['pixels_per_second']:,.0f} pixels/s

VALIDATION FOR THIS SYNTHETIC SCENE
------------------------------------------------------------
Abundance RMSE          : {result['abundance_rmse']:.5f}
Reconstruction RMSE     : {result['reconstruction_rmse']:.5f}

TOP PREDICTED MINERALS
------------------------------------------------------------
{mineral_lines}

ALTERATION INTERPRETATION
------------------------------------------------------------
Primary alteration      : {alt['primary']}
Secondary alteration    : {alt['secondary']}
Confidence              : {alt['confidence']}

PROCESSABILITY PREDICTION
------------------------------------------------------------
Prediction              : {proc['label']}
Risk score              : {proc['score']:.2f} / 3.00
Confidence              : {proc['confidence']}
Main alteration driver  : {proc['driver']}

PROCESSING RISK PROFILE
------------------------------------------------------------
{risk_lines}

RECOMMENDED PROCESSING ROUTE
------------------------------------------------------------
{proc['route']}

Recommended action:
    {proc['action']}

Why this can protect yield:
    {proc['yield_logic']}

Additional considerations:
{consideration_lines}

NOTE
------------------------------------------------------------
This is a geology-informed processability prediction from alteration
mineralogy. It is feed-forward decision support, not a replacement for
metallurgical testwork or a measured recovery value.
============================================================
""".strip()


# ---------------------------------------------------------------------
# Save current scene outputs
# ---------------------------------------------------------------------
def save_scene_outputs(result, mineral_names, alteration_classes):
    out_dir = DEMO_RESULTS_DIR / f"scene_{result['scene_idx'] + 1:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "predicted_abundances.npy", result["abundance_cube"])
    np.save(out_dir / "alteration_labels.npy", result["label_map"])
    np.save(out_dir / "processability_labels.npy", result["process_map"])

    report = console_report(result)
    (out_dir / "oreacle_report.txt").write_text(report, encoding="utf-8")

    mean_abundance = np.array(
        [
            result["alteration"]["mean_mineral_abundance"][name]
            for name in mineral_names
        ],
        dtype=float,
    )

    figures = {
        "dominant_mineral_map.png": fig_dominant_mineral_map(
            result["abundance_cube"], mineral_names
        ),
        "alteration_map.png": fig_alteration_map(
            result["label_map"], alteration_classes
        ),
        "processability_map.png": fig_processability_map(
            result["process_map"]
        ),
        "mean_abundance.png": fig_mean_abundances(
            mean_abundance, mineral_names
        ),
        "risk_profile.png": fig_risk_profile(
            result["processability"]["risk_scores"]
        ),
    }

    for filename, fig in figures.items():
        fig.savefig(
            out_dir / filename,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    return out_dir


# ---------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Oreacle Conveyor Demo",
    page_icon="⛏️",
    layout="wide",
)

st.title("⛏️ OREACLE")
st.caption(
    "Real-time mineral characterisation → alteration mapping → "
    "ore processability prediction"
)

try:
    env = load_oreacle()
except Exception as exc:
    st.error("Oreacle could not load the current project pipeline.")
    st.code(str(exc))
    st.stop()

plan = build_demo_plan(env["mineral_names"])

if "scene_idx" not in st.session_state:
    st.session_state.scene_idx = 0

if "result" not in st.session_state:
    st.session_state.result = None

if "history" not in st.session_state:
    st.session_state.history = {}


# Sidebar controls.
with st.sidebar:
    st.header("Demo controls")

    selected_mineral = st.selectbox(
        "Abundance map",
        env["mineral_names"],
        index=0,
    )

    display_delay = st.slider(
        "Auto-run display delay",
        min_value=0.0,
        max_value=1.5,
        value=0.35,
        step=0.05,
        help=(
            "This delay is only for the judges to see each scene. "
            "The displayed inference time is the real model runtime."
        ),
    )

    st.divider()

    st.write(f"**Device:** {env['device']}")
    st.write(f"**Model minerals:** {env['K']}")
    st.write(f"**Spectral bands:** {env['B']}")
    st.write(f"**Demo scenes:** {N_DEMO_SCENES}")
    st.write(f"**Scene size:** {DEMO_HEIGHT} × {DEMO_WIDTH}")

    with st.expander("How the demo is generated"):
        st.write(
            "The 50-scene belt is synthetic. Each segment mixes spectra "
            "from the same 12-mineral library. The segment identity is not "
            "given to the trained encoder; the network receives only the "
            "generated hyperspectral pixels."
        )


# Top controls.
b1, b2, b3, b4, b5, b6 = st.columns(6)

if b1.button("⚙️ Generate 50 scenes", use_container_width=True):
    st.session_state.scene_idx = 0
    st.session_state.result = None
    st.session_state.history = {}
    st.success("50-scene synthetic conveyor sequence prepared.")

if b2.button("◀ Previous", use_container_width=True):
    st.session_state.scene_idx = max(0, st.session_state.scene_idx - 1)
    st.session_state.result = run_scene(st.session_state.scene_idx)

if b3.button("▶ Run current", use_container_width=True):
    idx = st.session_state.scene_idx
    result = run_scene(idx)
    st.session_state.result = result
    st.session_state.history[idx] = {
        "abundance_rmse": result["abundance_rmse"],
        "reconstruction_rmse": result["reconstruction_rmse"],
        "t_total": result["t_total"],
        "process_score": result["processability"]["score"],
    }

if b4.button("Next ▶", use_container_width=True):
    st.session_state.scene_idx = min(
        N_DEMO_SCENES - 1,
        st.session_state.scene_idx + 1,
    )
    result = run_scene(st.session_state.scene_idx)
    st.session_state.result = result
    st.session_state.history[st.session_state.scene_idx] = {
        "abundance_rmse": result["abundance_rmse"],
        "reconstruction_rmse": result["reconstruction_rmse"],
        "t_total": result["t_total"],
        "process_score": result["processability"]["score"],
    }

auto_run = b5.button("⏩ Auto run", use_container_width=True)

if b6.button("↺ Reset", use_container_width=True):
    st.session_state.scene_idx = 0
    st.session_state.result = None
    st.session_state.history = {}
    st.rerun()


# -------------------------------------------------------------
# Auto-run mode
# -------------------------------------------------------------
if auto_run:
    progress = st.progress(0)
    status = st.empty()
    auto_metrics = st.empty()
    auto_map = st.empty()
    auto_decision = st.empty()

    start_idx = st.session_state.scene_idx

    for idx in range(start_idx, N_DEMO_SCENES):
        result = run_scene(idx)

        st.session_state.scene_idx = idx
        st.session_state.result = result
        st.session_state.history[idx] = {
            "abundance_rmse": result["abundance_rmse"],
            "reconstruction_rmse": result["reconstruction_rmse"],
            "t_total": result["t_total"],
            "process_score": result["processability"]["score"],
        }

        progress.progress((idx + 1) / N_DEMO_SCENES)

        status.markdown(
            f"### Conveyor scene {idx + 1:02d}/{N_DEMO_SCENES}"
        )

        proc = result["processability"]

        auto_metrics.markdown(
            f"""
**Inference:** `{result['t_total'] * 1000:.1f} ms` &nbsp;&nbsp; | &nbsp;&nbsp;
**Alteration:** `{result['alteration']['primary']}` &nbsp;&nbsp; | &nbsp;&nbsp;
**Processability:** `{proc['label']}`
"""
        )

        fig = fig_processability_map(result["process_map"])
        auto_map.pyplot(fig, clear_figure=True)
        plt.close(fig)

        auto_decision.info(
            f"**Route:** {proc['route']}\n\n"
            f"**Action:** {proc['action']}"
        )

        if display_delay > 0:
            time.sleep(display_delay)

    st.success("50-scene conveyor demonstration complete.")


# -------------------------------------------------------------
# Main current-scene view
# -------------------------------------------------------------
result = st.session_state.result

if result is None:
    st.info(
        "Press **Run current** to process Scene 1, or press "
        "**Auto run** to play through the 50-scene conveyor."
    )

    # Show sequence plan without showing expected model outputs.
    st.subheader("Synthetic conveyor sequence")
    stage_rows = []
    last_profile = None
    start = 0

    for item in plan:
        if last_profile is None:
            last_profile = item["profile"]
            start = item["scene"] + 1
        elif item["profile"] != last_profile:
            stage_rows.append(
                {
                    "Scenes": f"{start}-{item['scene']}",
                    "Synthetic test segment": last_profile,
                }
            )
            last_profile = item["profile"]
            start = item["scene"] + 1

    stage_rows.append(
        {
            "Scenes": f"{start}-{N_DEMO_SCENES}",
            "Synthetic test segment": last_profile,
        }
    )

    st.table(stage_rows)
    st.stop()


proc = result["processability"]
alt = result["alteration"]

# Headline metrics.
m1, m2, m3, m4, m5 = st.columns(5)

m1.metric(
    "Scene",
    f"{result['scene_idx'] + 1:02d}/{N_DEMO_SCENES}",
)

m2.metric(
    "ML pipeline time",
    f"{result['t_total'] * 1000:.1f} ms",
)

m3.metric(
    "Abundance RMSE",
    f"{result['abundance_rmse']:.4f}",
)

m4.metric(
    "Reconstruction RMSE",
    f"{result['reconstruction_rmse']:.4f}",
)

m5.metric(
    "Process risk",
    f"{proc['score']:.2f}/3",
)

st.caption(
    f"Synthetic belt segment: **{result['profile']}** · "
    f"SNR: **{result['snr_db']:.1f} dB** · "
    f"Throughput: **{result['pixels_per_second']:,.0f} pixels/s**"
)


# Strong judge-facing decision banner.
st.markdown("---")

left, right = st.columns([1.05, 1.95])

with left:
    st.subheader("Ore interpretation")
    st.write(f"**Primary alteration:** {alt['primary']}")
    st.write(f"**Secondary:** {alt['secondary']}")
    st.write(f"**Alteration confidence:** {alt['confidence']}")
    st.write(f"**Processability:** {proc['label']}")
    st.write(f"**Prediction confidence:** {proc['confidence']}")

with right:
    st.subheader("Operational decision")
    st.success(f"**{proc['route']}**")
    st.write(f"**Recommended action:** {proc['action']}")
    st.write(f"**Why this protects yield:** {proc['yield_logic']}")


# Visual outputs.
st.markdown("---")
st.subheader("What Oreacle sees")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Mineral abundance",
        "Dominant mineral map",
        "Alteration map",
        "Processability map",
    ]
)

with tab1:
    fig = fig_abundance_map(
        result["abundance_cube"],
        env["mineral_names"],
        selected_mineral,
    )
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    fig = fig_dominant_mineral_map(
        result["abundance_cube"],
        env["mineral_names"],
    )
    st.pyplot(fig)
    plt.close(fig)

with tab3:
    fig = fig_alteration_map(
        result["label_map"],
        env["pipeline"].ALTERATION_CLASSES,
    )
    st.pyplot(fig)
    plt.close(fig)

with tab4:
    fig = fig_processability_map(result["process_map"])
    st.pyplot(fig)
    plt.close(fig)


# Scene-wide abundance and risk profile.
st.markdown("---")

c1, c2 = st.columns(2)

mean_abundance = np.array(
    [
        alt["mean_mineral_abundance"][name]
        for name in env["mineral_names"]
    ],
    dtype=float,
)

with c1:
    fig = fig_mean_abundances(
        mean_abundance,
        env["mineral_names"],
    )
    st.pyplot(fig)
    plt.close(fig)

with c2:
    fig = fig_risk_profile(proc["risk_scores"])
    st.pyplot(fig)
    plt.close(fig)


# Top mineral table.
st.subheader("Top predicted minerals")

top_rows = [
    {
        "Mineral": name,
        "Mean abundance": f"{100.0 * value:.2f}%",
    }
    for name, value in alt["top_minerals"]
]
st.table(top_rows)


# Technical detail / recommendations.
with st.expander("Processing considerations"):
    for item in proc["considerations"]:
        st.write(f"• {item}")

    st.caption(
        "The processability layer is a transparent geology-informed rule "
        "engine based on the predicted alteration assemblage. It predicts "
        "likely processing behaviour; it does not claim measured recovery."
    )


# Terminal-style output.
st.markdown("---")
st.subheader("Oreacle terminal output")
st.code(console_report(result), language="text")


# Save current scene.
save_col, _ = st.columns([1, 3])

if save_col.button("💾 Save current scene outputs"):
    out_dir = save_scene_outputs(
        result,
        env["mineral_names"],
        env["pipeline"].ALTERATION_CLASSES,
    )
    st.success(f"Saved to: {out_dir}")


# History chart if more than one scene has been processed.
if len(st.session_state.history) >= 2:
    st.markdown("---")
    st.subheader("Processed-scene history")

    history_indices = sorted(st.session_state.history)
    scenes = np.array(history_indices) + 1
    risk = np.array(
        [
            st.session_state.history[i]["process_score"]
            for i in history_indices
        ]
    )

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(scenes, risk, marker="o")
    ax.set_ylim(0, 3.1)
    ax.set_xlabel("Scene")
    ax.set_ylabel("Processability risk (1-3)")
    ax.set_title("Conveyor processability trend")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
# Hope and pray