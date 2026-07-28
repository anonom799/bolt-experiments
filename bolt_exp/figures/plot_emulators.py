#!/usr/bin/env python3
"""Visualize emulators for HPO and DM problems for a conference paper.

Four figures:
  1. Rank-rank scatter   — emulator accuracy on held-out val sets
  2. Landscape slice     — HPO 2D heatmap + DM ternary (one per phase group)
  3. Multi-fidelity      — step fidelity error curve + model fidelity histogram
  4. Pareto front        — 3 pairwise 2D scatter plots with 3rd objective as color

Usage:
  python plot_emulators.py
  python plot_emulators.py --out_dir /path/to/out

Data path args default to the raw eval data committed under `data/`.
"""

import argparse
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import spearmanr

import bolt

from bolt_exp import REPO_ROOT

matplotlib.rcParams.update({"font.size": 11})

NOISE_FEATURES = ["if_prop1", "math_prop1", "math_prop2"]

HPO_TARGET_OPTIONS = [
    "q_proj,v_proj",
    "q_proj,v_proj,k_proj,o_proj",
    "gate_proj,up_proj,down_proj",
    "all-linear",
]

sns.set_theme(style="white", context="paper", font_scale=1.3)

# ============================================================
# Shared helpers
# ============================================================


def encode_hpo_parquet(df: pd.DataFrame) -> torch.Tensor:
    """Encode raw HPO parquet rows into bolt HPO 7-dim input format."""
    rows = []
    for _, row in df.iterrows():
        lr = (row["learning_rate"] - 1e-6) / (1e-3 - 1e-6)
        batch = float(np.log2(row["per_device_train_batch_size"]))
        rank = float(np.log2(row["lora_r"]))
        alpha = float(np.log2(row["lora_alpha"]))
        dropout = row["lora_dropout"] / 0.1
        s = row["lora_layers"]
        layers = 30.0 if s == "all" else float(s.split("_")[-1])
        target = float(HPO_TARGET_OPTIONS.index(row["lora_target_modules"]))
        rows.append([lr, batch, rank, alpha, dropout, layers, target])
    return torch.tensor(rows, dtype=torch.double)


def rank_normalize(arr: np.ndarray) -> np.ndarray:
    """Fractional ranks in [0, 1]."""
    n = len(arr)
    ranks = np.empty(n)
    order = np.argsort(arr)
    ranks[order] = np.arange(n) / max(n - 1, 1)
    return ranks


def scatter_with_kde(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    scatter_color,
    kde_cmap="Blues",
    s=20,
) -> None:
    """Scatter plot with 2D KDE contours overlaid."""
    ax.scatter(
        x, y, color=scatter_color, s=s, alpha=0.5, edgecolors="none", zorder=1
    )
    clip = ((0, 1), (0, 1))
    sns.kdeplot(
        x=x,
        y=y,
        ax=ax,
        cmap=kde_cmap,
        fill=True,
        alpha=0.4,
        levels=6,
        clip=clip,
        zorder=2,
    )
    sns.kdeplot(
        x=x,
        y=y,
        ax=ax,
        color="k",
        linewidths=0.4,
        alpha=0.4,
        levels=6,
        clip=clip,
        zorder=3,
    )


def simplex_grid(n: int = 60) -> np.ndarray:
    """Triangular grid of (n+1)*(n+2)/2 points on the 2-simplex (a+b+c=1)."""
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append([i / n, j / n, 1.0 - i / n - j / n])
    return np.array(pts)


def ternary_to_cart(abc: np.ndarray) -> np.ndarray:
    """(N,3) simplex → (N,2) Cartesian.  Vertices: a→(0,0), b→(1,0), c→(0.5, √3/2)."""
    x = abc[:, 1] + 0.5 * abc[:, 2]
    y = np.sqrt(3) / 2 * abc[:, 2]
    return np.column_stack([x, y])


def draw_ternary_frame(ax: plt.Axes, labels=("IF", "Math", "Code")) -> None:
    h = np.sqrt(3) / 2
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], "k-", lw=1)
    ax.text(-0.03, -0.03, labels[0], ha="right", va="top", fontsize=12)
    ax.text(1.03, -0.03, labels[1], ha="left", va="top", fontsize=12)
    ax.text(0.5, h + 0.01, labels[2], ha="center", va="bottom", fontsize=12)
    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.1, h + 0.15)
    ax.set_aspect("equal")
    ax.axis("off")


def sample_hpo_configs(n: int, seed: int = 0) -> torch.Tensor:
    """Random HPO configs in bolt encoding (no fidelity dim)."""
    torch.manual_seed(seed)
    X = torch.zeros(n, 7, dtype=torch.double)
    X[:, 0] = torch.rand(n, dtype=torch.double)  # lr
    X[:, 1] = torch.randint(2, 5, (n,)).double()  # batch log2
    X[:, 2] = torch.randint(2, 6, (n,)).double()  # rank log2
    X[:, 3] = torch.randint(2, 6, (n,)).double()  # alpha log2
    X[:, 4] = torch.rand(n, dtype=torch.double)  # dropout
    X[:, 5] = torch.randint(1, 31, (n,)).double()  # layers
    X[:, 6] = torch.randint(0, 4, (n,)).double()  # target
    return X


# ============================================================
# SECTION 1: RANK-RANK SCATTER
# ============================================================


def plot_rank_rank_scatter(
    ax_hpo: plt.Axes,
    axes_dm,
    prob_hpo: bolt.HPO,
    prob_dmo: bolt.DMCurriculumMO,
    hpo_val_path: Path,
    dm_val_path: Path,
) -> None:
    """Predicted vs actual rank scatter for HPO (qwen8b) and DM per-objective (qwen4b) val sets."""

    # --- HPO: one obs per run = row with most training tokens ---
    df_hpo = pd.read_parquet(hpo_val_path)
    df_hpo_best = df_hpo.loc[
        df_hpo.groupby("run_name")["train/num_input_tokens_seen"].idxmax()
    ]
    X_hpo = encode_hpo_parquet(df_hpo_best)
    y_actual_hpo = df_hpo_best["eval/math500/accuracy/mean"].values

    with torch.no_grad():
        y_pred_hpo = prob_hpo._evaluate_true(X_hpo).squeeze(-1).numpy()

    rho_hpo, _ = spearmanr(y_actual_hpo, y_pred_hpo)
    rx_hpo = rank_normalize(y_actual_hpo)
    ry_hpo = rank_normalize(y_pred_hpo)
    scatter_with_kde(
        ax_hpo,
        rx_hpo,
        ry_hpo,
        scatter_color="#0072B2",
        kde_cmap="Blues",
        s=18,
    )
    ax_hpo.plot([0, 1], [0, 1], "k--", lw=1)
    ax_hpo.set_xlabel("Actual rank")
    ax_hpo.set_ylabel("Predicted rank")
    ax_hpo.set_title(f"HPO (ρ = {rho_hpo:.3f})")
    ax_hpo.set_xlim(0, 1)
    ax_hpo.set_ylim(0, 1)

    # --- DM: per-objective rank-rank using DMCurriculumMO ---
    df_dm = pd.read_parquet(dm_val_path)
    _obj_cols = [
        "eval/ifeval/accuracy/mean",
        "eval/minerva_math500/accuracy/mean",
        "eval/mbpp_plus_instruct/accuracy/mean",
    ]
    _obj_labels = ["DM IFEval", "DM MATH-500", "DM MBPP+"]
    X_dm = torch.tensor(
        df_dm[
            [
                "if_prop1",
                "math_prop1",
                "code_prop1",
                "if_prop2",
                "math_prop2",
                "code_prop2",
            ]
        ].values,
        dtype=torch.double,
    )
    with torch.no_grad():
        y_pred_all = prob_dmo._evaluate_true(X_dm).numpy()  # (N, 3)

    for ax, col, label in zip(axes_dm, _obj_cols, _obj_labels):
        y_actual = df_dm[col].values
        y_pred = y_pred_all[:, _obj_cols.index(col)]
        rho, _ = spearmanr(y_actual, y_pred)
        rx = rank_normalize(y_actual)
        ry = rank_normalize(y_pred)
        scatter_with_kde(
            ax, rx, ry, scatter_color="#E69F00", kde_cmap="Oranges", s=18
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("Actual rank")
        ax.set_ylabel("Predicted rank")
        ax.set_title(f"{label} (ρ = {rho:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)


# ============================================================
# SECTION 2: LANDSCAPE SLICE
# ============================================================


def plot_hpo_landscape(
    ax: plt.Axes, prob_hpo: bolt.HPO, cmap="viridis"
) -> None:
    """2D heatmap of HPO: sweep lr × dropout, fix all other dims at known optimizer."""
    # _optimizers = (lr=0.311, batch=2, rank=4, alpha=2, dropout=0.871, layers=30, target=1)
    n = 80
    lr_vals = np.linspace(0, 1, n)
    dr_vals = np.linspace(0, 1, n)
    LR, DR = np.meshgrid(lr_vals, dr_vals)

    X_grid = torch.zeros(n * n, 7, dtype=torch.double)
    X_grid[:, 0] = torch.tensor(LR.ravel())
    X_grid[:, 1] = 2.0  # batch = log2(4)
    X_grid[:, 2] = 4.0  # rank  = log2(16)
    X_grid[:, 3] = 2.0  # alpha = log2(4)
    X_grid[:, 4] = torch.tensor(DR.ravel())
    X_grid[:, 5] = 30.0  # layers = all
    X_grid[:, 6] = 1.0  # target = qvko

    with torch.no_grad():
        Z = prob_hpo._evaluate_true(X_grid).squeeze(-1).numpy().reshape(n, n)

    im = ax.contourf(LR, DR, Z, levels=30, cmap=cmap)
    plt.colorbar(im, ax=ax, label="Math-500 acc.")
    ax.set_xlabel("lr (normalized [1e-6, 1e-3])", fontsize=12)
    ax.set_ylabel("lora_dropout (normalized [0, 0.1])", fontsize=12)
    ax.set_title(
        "HPO landscape\n(batch=4, rank=16, α=4, layers=all, target=qvko)",
        fontsize=11,
        pad=2,
    )


def _dmo_ternary_Y(
    prob_dm,
    phase_idx: int,
    fixed_other: np.ndarray,
    n_grid: int = 60,
    obj_idx=0,
) -> tuple:
    """Compute emulator outputs for a DM ternary slice. Returns (grid, Y)."""
    grid = simplex_grid(n_grid)
    X_full = np.zeros((len(grid), 6))
    if phase_idx == 0:
        X_full[:, :3] = grid
        X_full[:, 3:] = fixed_other
    else:
        X_full[:, :3] = fixed_other
        X_full[:, 3:] = grid
    with torch.no_grad():
        Y_raw = prob_dm._evaluate_true(
            torch.tensor(X_full, dtype=torch.double)
        ).numpy()
    if obj_idx is None:
        Y = (
            Y_raw.mean(axis=1)
            if Y_raw.ndim > 1 and Y_raw.shape[1] > 1
            else Y_raw.ravel()
        )
    else:
        Y = (
            Y_raw.ravel()
            if Y_raw.ndim == 1 or Y_raw.shape[1] == 1
            else Y_raw[:, obj_idx]
        )
    return grid, Y


def _dmo_ternary_draw(
    ax: plt.Axes,
    grid: np.ndarray,
    Y: np.ndarray,
    phase_idx: int,
    fixed_other: np.ndarray,
    vmin=None,
    vmax=None,
    cmap="plasma",
):
    """Draw pre-computed DM ternary values onto ax. Returns im."""
    cart = ternary_to_cart(grid)
    triang = mtri.Triangulation(cart[:, 0], cart[:, 1])
    im = ax.tricontourf(triang, Y, levels=30, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_ternary_frame(ax)
    other_label = f"Phase {2 - phase_idx} @ [{fixed_other[0]:.2f}, {fixed_other[1]:.2f}, {fixed_other[2]:.2f}]"
    ax.set_title(
        f"DM Phase {phase_idx + 1} landscape ({other_label})",
        fontsize=11,
        pad=2,
    )
    return im


def plot_dmo_ternary(
    ax: plt.Axes,
    prob_dm,
    phase_idx: int,
    fixed_other: np.ndarray,
    n_grid: int = 60,
    obj_idx=0,
    obj_label: str = "Score",
    cmap="plasma",
) -> None:
    """Ternary heatmap for one DM phase group; fix the other at best observed point."""
    grid, Y = _dmo_ternary_Y(prob_dm, phase_idx, fixed_other, n_grid, obj_idx)
    im = _dmo_ternary_draw(ax, grid, Y, phase_idx, fixed_other, cmap=cmap)
    plt.colorbar(
        im, ax=ax, label=obj_label, fraction=0.046, pad=0.0, shrink=0.7
    )


# ============================================================
# SECTION 3: MULTI-FIDELITY CORRELATION
# ============================================================


def plot_mf_step_error(
    ax: plt.Axes, n_samples: int = 300, seed: int = 0
) -> None:
    """Mean relative error vs fidelity level for HPOMultiFidelityToken."""
    prob = bolt.HPOMultiFidelityToken(noise_std=None)
    X_base = sample_hpo_configs(n_samples, seed=seed)
    X_8d = torch.cat(
        [X_base, torch.zeros(n_samples, 1, dtype=torch.double)], dim=1
    )

    # Reference: full fidelity
    X_ref = X_8d.clone()
    X_ref[:, 7] = 1.0
    with torch.no_grad():
        y_ref = prob._evaluate_true(X_ref).squeeze(-1).numpy()

    fidelity_bins = np.linspace(0.05, 1.0, 20)
    mean_err, std_err = [], []
    for s in fidelity_bins:
        X_s = X_8d.clone()
        X_s[:, 7] = s
        with torch.no_grad():
            y_s = prob._evaluate_true(X_s).squeeze(-1).numpy()
        # cummax guarantees y_s <= y_ref, so (y_ref - y_s) / y_ref >= 0
        rel_err = (y_ref - y_s) / np.abs(y_ref).clip(1e-8)
        mean_err.append(rel_err.mean())
        std_err.append(rel_err.std())

    mean_err, std_err = np.array(mean_err), np.array(std_err)
    ax.plot(fidelity_bins, mean_err, color="#0072B2", lw=2)
    ax.fill_between(
        fidelity_bins,
        mean_err - std_err,
        mean_err + std_err,
        alpha=0.25,
        color="#0072B2",
    )
    ax.set_xlabel("Fidelity (normalized training tokens)")
    ax.set_ylabel("Relative error")
    ax.set_title("HPO-MF-Cont")
    ax.set_xlim(0, 1)


def plot_mf_model_histogram(
    ax: plt.Axes, n_samples: int = 300, seed: int = 0
) -> None:
    """Histogram of (HF − LF) / |HF| for HPOMultiFidelityModel (4B vs 8B)."""
    prob = bolt.HPOMultiFidelityModel(noise_std=None)
    X_base = sample_hpo_configs(n_samples, seed=seed)
    X_8d = torch.cat(
        [X_base, torch.zeros(n_samples, 1, dtype=torch.double)], dim=1
    )

    X_lf = X_8d.clone()
    X_lf[:, 7] = 0.0  # 4B (low fidelity)
    X_hf = X_8d.clone()
    X_hf[:, 7] = 1.0  # 8B (high fidelity)

    with torch.no_grad():
        y_lf = prob._evaluate_true(X_lf).squeeze(-1).numpy()
        y_hf = prob._evaluate_true(X_hf).squeeze(-1).numpy()

    norm_diff = (y_hf - y_lf) / np.abs(y_hf).clip(1e-8)
    sns.histplot(norm_diff, bins=30, color="#E69F00", ax=ax, alpha=0.85)
    ax.axvline(0, color="k", lw=1, linestyle="--")
    ax.set_xlabel("Normalized difference")
    ax.set_ylabel("Count")
    ax.set_title("HPO-MF-Disc: 4B vs 8B")


# ============================================================
# SECTION 4: PARETO FRONT
# ============================================================


def plot_pareto_front(axes, pf_path: Path) -> None:
    """3 pairwise scatter plots of the Pareto front with 3rd objective as color."""
    df = pd.read_csv(pf_path)
    s_if = df["score_if"].values
    s_math = df["score_math"].values
    s_code = df["score_code"].values

    configs = [
        ("IFEval", "MATH-500", "MBPP+", s_if, s_math, s_code, "viridis"),
        ("IFEval", "MBPP+", "MATH-500", s_if, s_code, s_math, "plasma"),
        ("MATH-500", "MBPP+", "IFEval", s_math, s_code, s_if, "cividis"),
    ]
    for ax, (xl, yl, cl, xs, ys, cs, cmap) in zip(axes, configs):
        sc = ax.scatter(xs, ys, c=cs, cmap=cmap, s=5, alpha=0.5, linewidths=0)
        plt.colorbar(sc, ax=ax, label=cl, fraction=0.046, pad=0.04)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
    axes[1].set_title("DM Pareto front (optimal, from emulator)")


# ============================================================
# SECTION 5: NOISE MODEL DIAGNOSTICS
# ============================================================


def _noise_predict(
    prob_dmhet: bolt.DMCurriculumHet, noise_df: pd.DataFrame
) -> np.ndarray:
    """Run the heteroscedastic noise emulator on noise_df rows, return (N,) std_math array."""
    prop_cols = [
        "if_prop1",
        "math_prop1",
        "code_prop1",
        "if_prop2",
        "math_prop2",
        "code_prop2",
    ]
    X = torch.tensor(noise_df[prop_cols].values, dtype=torch.double)
    with torch.no_grad():
        return prob_dmhet._evaluate_noise(X).squeeze(-1).numpy() / 0.1


def plot_noise_scatter(
    noise_df: pd.DataFrame,
    prob_dmhet: bolt.DMCurriculumHet,
    out_dir: Path,
    cmap="viridis",
) -> None:
    """Scatter of actual (and predicted) std_math: if_prop1 × math_prop1, size=math_prop2."""
    df = noise_df.copy()

    s_min, s_max = 20, 300
    m2 = df["math_prop2"].values
    sizes = s_min + (s_max - s_min) * (m2 - m2.min()) / (
        m2.max() - m2.min() + 1e-9
    )

    actual = df["std_math"].values
    predicted = _noise_predict(prob_dmhet, df)

    vmin = min(actual.min(), predicted.min())
    vmax = max(actual.max(), predicted.max())
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), squeeze=False)

    def _scatter_panel(ax, colors, title):
        sc = ax.scatter(
            df["if_prop1"],
            df["math_prop1"],
            c=colors,
            cmap=cmap,
            norm=norm,
            s=sizes,
            alpha=0.75,
            linewidths=0,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("std_math", fontsize=11)
        for m2_val in [0.0, 0.25, 0.50, 0.75, 1.0]:
            s = s_min + (s_max - s_min) * (m2_val - m2.min()) / (
                m2.max() - m2.min() + 1e-9
            )
            ax.scatter([], [], s=s, c="gray", alpha=0.7, label=f"{m2_val:.2f}")
        ax.legend(
            title="math_prop2",
            fontsize=8,
            title_fontsize=8,
            loc="upper right",
            framealpha=0.7,
        )
        ax.set_xlabel("if_prop1", fontsize=12)
        ax.set_ylabel("math_prop1", fontsize=12)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(title, fontsize=12)

    _scatter_panel(
        axes[0, 0],
        actual,
        # "actual std_math: if_prop1 × math_prop1  (size = math_prop2)",
        "Actual std_math",
    )
    _scatter_panel(
        axes[0, 1],
        predicted,
        # "predicted std_math: if_prop1 × math_prop1  (size = math_prop2)",
        "Predicted std_math",
    )

    plt.tight_layout()
    out_path = out_dir / "emulator_noise_scatter.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  saved emulator_noise_scatter.pdf")
    plt.close(fig)


def plot_pred_vs_actual(
    noise_df: pd.DataFrame,
    prob_dmhet: bolt.DMCurriculumHet,
    out_dir: Path,
) -> None:
    """Predicted vs actual std_math scatter (noise emulator)."""
    y = noise_df["std_math"].values.astype(float)
    y_pred = _noise_predict(prob_dmhet, noise_df)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(y, y_pred, s=40, alpha=0.75, linewidths=0, color="#E69F00")

    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    pad = (hi - lo) * 0.05
    ax.plot(
        [lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1, label="y = x"
    )

    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("actual std_math", fontsize=12)
    ax.set_ylabel("predicted std_math", fontsize=12)
    ax.set_title("Noise emulator on training set", fontsize=12)
    # ax.legend(fontsize=12)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path = out_dir / "emulator_pred_vs_actual.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  saved emulator_pred_vs_actual.pdf")
    plt.close(fig)


# ============================================================
# Main
# ============================================================


def parse_args() -> argparse.Namespace:
    # The diagnostics data ships with this repo under `data/`.
    _data_root = REPO_ROOT / "data"
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--out_dir", type=Path, default=None)
    p.add_argument(
        "--hpo_val_path",
        type=Path,
        default=_data_root / "hyperparam_opt/out_val_lhc_qwen8b/data.parquet",
    )
    p.add_argument(
        "--dm_val_path",
        type=Path,
        default=_data_root / "data_mixture/out_val_sobol_qwen4b/data.parquet",
    )
    p.add_argument(
        "--pf_path",
        type=Path,
        default=_data_root / "data_mixture/out_qwen4b/pareto_front.csv",
    )
    p.add_argument(
        "--noise_targets_path",
        type=Path,
        default=_data_root
        / "data_mixture/out_qwen4b_samples/noise_targets.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (
        args.out_dir if args.out_dir is not None else REPO_ROOT
    )

    # Load problems once to avoid repeated HF hub downloads
    print("Loading emulators...")
    prob_hpo = bolt.HPO(noise_std=None)
    prob_dm = bolt.DMCurriculum(noise_std=None)
    prob_dmo = bolt.DMCurriculumMO(noise_std=None)
    prob_dmhet = bolt.DMCurriculumHet(noise_std=None)

    # Find DM val points to fix as frozen phase in ternaries
    df_dm = pd.read_parquet(args.dm_val_path)
    _obj_cols = [
        "eval/ifeval/accuracy/mean",
        "eval/minerva_math500/accuracy/mean",
        "eval/mbpp_plus_instruct/accuracy/mean",
    ]

    # best score
    best_dm = df_dm.loc[df_dm[_obj_cols].mean(axis=1).idxmax()]

    # # 90 percentile
    # mean_scores = df_dm[_obj_cols].mean(axis=1)
    # best_dm = df_dm.loc[
    #     (mean_scores - mean_scores.quantile(0.90)).abs().idxmin()
    # ]

    # sample from top 10%
    mean_scores = df_dm[_obj_cols].mean(axis=1)
    top10 = df_dm[mean_scores >= mean_scores.quantile(0.80)]
    sample_rows = [best_dm] + [
        row for _, row in top10.sample(3, random_state=0).iterrows()
    ]

    # --- Figure 1: Rank-rank scatter ---
    print("Plotting rank-rank scatter...")
    fig1, axes1 = plt.subplots(1, 4, figsize=(13, 3))
    plot_rank_rank_scatter(
        axes1[0],
        axes1[1:],
        prob_hpo,
        prob_dmo,
        args.hpo_val_path,
        args.dm_val_path,
    )
    fig1.tight_layout()
    fig1.savefig(out_dir / "emulator_rank_rank.pdf", bbox_inches="tight")
    plt.close(fig1)
    print("  saved emulator_rank_rank.pdf")

    # --- Figure 2: HPO landscape (standalone) ---
    print("Plotting HPO landscape...")

    cmap = sns.cubehelix_palette(
        start=3,
        rot=-0.3,
        hue=1.7,
        dark=0.15,
        light=0.8,
        gamma=1.3,
        as_cmap=True,
        reverse=True,
    )

    fig_hpo_land, ax_hpo_land = plt.subplots(1, 1, figsize=(5, 4.5))
    plot_hpo_landscape(ax_hpo_land, prob_hpo, cmap=cmap)

    fig_hpo_land.tight_layout()
    fig_hpo_land.savefig(
        out_dir / "emulator_landscape_hpo.pdf", bbox_inches="tight"
    )
    plt.close(fig_hpo_land)

    print("  saved emulator_landscape_hpo.pdf")
    # cmap = sns.color_palette("viridis", as_cmap=True)

    obj_names = ["ifeval", "math", "code"]
    obj_labels = ["IFEval", "MATH-500", "MBPP+"]

    for s_idx, best_dm in enumerate(sample_rows):
        fixed_phase1 = best_dm[
            ["if_prop1", "math_prop1", "code_prop1"]
        ].values.astype(float)
        fixed_phase2 = best_dm[
            ["if_prop2", "math_prop2", "code_prop2"]
        ].values.astype(float)
        phase_configs = [(0, fixed_phase2, "p1"), (1, fixed_phase1, "p2")]
        tag = f"_s{s_idx}"

        # --- Figure 2b: Combined landscape (HPO + DM ternaries) ---
        print(f"Plotting combined landscape slices (sample {s_idx})...")
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
        plot_hpo_landscape(axes2[0], prob_hpo)
        plot_dmo_ternary(
            axes2[1], prob_dm, phase_idx=0, fixed_other=fixed_phase2
        )
        plot_dmo_ternary(
            axes2[2], prob_dm, phase_idx=1, fixed_other=fixed_phase1
        )
        fig2.tight_layout()
        fig2.savefig(
            out_dir / f"emulator_landscape{tag}.pdf", bbox_inches="tight"
        )
        plt.close(fig2)
        print(f"  saved emulator_landscape{tag}.pdf")

        # --- Figure 2c–e: Per-objective ternary plots ---
        for obj_idx, (obj_name, obj_label) in enumerate(
            zip(obj_names, obj_labels)
        ):
            # Paired figure
            fig_t, axes_t = plt.subplots(1, 2, figsize=(10, 4.5))
            plot_dmo_ternary(
                axes_t[0],
                prob_dmo,
                phase_idx=0,
                fixed_other=fixed_phase2,
                obj_idx=obj_idx,
                obj_label=obj_label,
                cmap=cmap,
            )
            plot_dmo_ternary(
                axes_t[1],
                prob_dmo,
                phase_idx=1,
                fixed_other=fixed_phase1,
                obj_idx=obj_idx,
                obj_label=obj_label,
                cmap=cmap,
            )
            fig_t.suptitle(f"DM landscape — {obj_label}", fontsize=12, y=1.0)
            fig_t.tight_layout()
            fname = f"emulator_ternary_{obj_name}{tag}.pdf"
            fig_t.savefig(out_dir / fname, bbox_inches="tight")
            plt.close(fig_t)
            print(f"  saved {fname}")

            # Individual figures (one per phase)
            for phase_idx, fixed_other, phase_tag in phase_configs:
                fig_i, ax_i = plt.subplots(1, 1, figsize=(5, 4.5))
                plot_dmo_ternary(
                    ax_i,
                    prob_dmo,
                    phase_idx=phase_idx,
                    fixed_other=fixed_other,
                    obj_idx=obj_idx,
                    obj_label=obj_label,
                    cmap=cmap,
                )
                fig_i.tight_layout()
                fname_i = f"emulator_ternary_{obj_name}_{phase_tag}{tag}.pdf"
                fig_i.savefig(out_dir / fname_i, bbox_inches="tight")
                plt.close(fig_i)
                print(f"  saved {fname_i}")

        # --- Mean-objective ternary (average of IFEval, MATH-500, MBPP+) ---
        fig_m, axes_m = plt.subplots(1, 2, figsize=(10, 4.5))
        plot_dmo_ternary(
            axes_m[0],
            prob_dmo,
            phase_idx=0,
            fixed_other=fixed_phase2,
            obj_idx=None,
            obj_label="Mean score",
            cmap=cmap,
        )
        plot_dmo_ternary(
            axes_m[1],
            prob_dmo,
            phase_idx=1,
            fixed_other=fixed_phase1,
            obj_idx=None,
            obj_label="Mean score",
            cmap=cmap,
        )
        fig_m.suptitle("DM landscape — Mean objective", fontsize=12, y=1.0)
        fig_m.tight_layout()
        fname_m = f"emulator_ternary_mean{tag}.pdf"
        fig_m.savefig(out_dir / fname_m, bbox_inches="tight")
        plt.close(fig_m)
        print(f"  saved {fname_m}")

        for phase_idx, fixed_other, phase_tag in phase_configs:
            fig_mi, ax_mi = plt.subplots(1, 1, figsize=(5, 4.5))
            plot_dmo_ternary(
                ax_mi,
                prob_dmo,
                phase_idx=phase_idx,
                fixed_other=fixed_other,
                obj_idx=None,
                obj_label="Mean score",
            )
            fig_mi.tight_layout()
            fname_mi = f"emulator_ternary_mean_{phase_tag}{tag}.pdf"
            fig_mi.savefig(out_dir / fname_mi, bbox_inches="tight")
            plt.close(fig_mi)
            print(f"  saved {fname_mi}")

    # --- Figure 3: Multi-fidelity ---
    print("Plotting multi-fidelity figures...")
    fig3, (ax_step, ax_model) = plt.subplots(1, 2, figsize=(6.5, 3))
    plot_mf_step_error(ax_step)
    plot_mf_model_histogram(ax_model)
    fig3.tight_layout()
    fig3.savefig(out_dir / "emulator_multifidelity.pdf", bbox_inches="tight")
    plt.close(fig3)
    print("  saved emulator_multifidelity.pdf")

    # --- Figure 4: Pareto front ---
    print("Plotting Pareto front...")
    fig4, axes4 = plt.subplots(1, 3, figsize=(13, 4))
    plot_pareto_front(axes4, args.pf_path)
    fig4.tight_layout()
    fig4.savefig(out_dir / "emulator_pareto.pdf", bbox_inches="tight")
    plt.close(fig4)
    print("  saved emulator_pareto.pdf")

    # --- Figure 5: Noise model diagnostics ---
    print("Plotting noise model diagnostics...")
    noise_df = pd.read_csv(args.noise_targets_path)
    plot_noise_scatter(noise_df, prob_dmhet, out_dir, cmap=cmap)
    plot_pred_vs_actual(noise_df, prob_dmhet, out_dir)

    print(f"\nAll figures saved to {out_dir}")


if __name__ == "__main__":
    main()
