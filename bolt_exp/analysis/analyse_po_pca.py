"""PCA / eigenvalue analysis for the PO prompt-embedding search space.

Loads the tabular dataset (5014 prompts × 768-dim embeddings + scores) and
runs two complementary analyses for each PO variant (128 / 256 / 512 / 768):

1. Unsupervised: cumulative explained variance of input space (how many PCs
   are needed to reconstruct the embeddings).
2. Supervised: cumulative R² of OLS regression on PCs → score (how many PCs
   are needed to predict y), and per-PC |correlation| with score.

Usage:
    python analyse_po_pca.py [--save] [--no_show]

Outputs (saved to the repo root if --save):
    po_pca_variance.png   — cumulative explained-variance curves (all dims)
    po_pca_r2.png         — cumulative R² curves (supervised)
    po_pca_corr.png       — per-PC |correlation| with score (first 50 PCs)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from bolt_exp import REPO_ROOT

PROBLEMS = {
    "PO-128": 128,
    "PO-256": 256,
    "PO-512": 512,
    "PO-768": 768,
}

THRESHOLDS = [0.80, 0.90, 0.95, 0.99]

# Okabe-Ito palette entries for the four PO variants
OKABE_ITO = ["#56B4E9", "#009E73", "#E69F00", "#0072B2"]


def load_data():
    ds = load_dataset("anonom799/po_qwen14b_tabular_data", split="train")
    X_full = np.array(ds["embedding"], dtype=np.float64)  # (5014, 768)
    y = np.array(ds["score"], dtype=np.float64)  # (5014,)
    return X_full, y


def run_pca(X: np.ndarray, max_dim: int) -> PCA:
    """Fit PCA keeping all components up to max_dim."""
    X_sub = X[:, :max_dim]
    scaler = StandardScaler(
        with_std=False
    )  # center only; embeddings already ~unit scale
    X_c = scaler.fit_transform(X_sub)
    pca = PCA(n_components=max_dim, svd_solver="full")
    pca.fit(X_c)
    return pca, scaler


def cumulative_r2(
    pca: PCA, scaler, X: np.ndarray, y: np.ndarray, max_dim: int
) -> np.ndarray:
    """R² of OLS(PCs[:k] → y) for k = 1 … max_dim."""
    X_sub = X[:, :max_dim]
    X_c = scaler.transform(X_sub)
    Z = pca.transform(X_c)  # (N, max_dim)
    r2 = np.zeros(max_dim)
    for k in range(1, max_dim + 1):
        reg = LinearRegression().fit(Z[:, :k], y)
        r2[k - 1] = reg.score(Z[:, :k], y)
    return r2


def pc_score_corr(
    pca: PCA, scaler, X: np.ndarray, y: np.ndarray, max_dim: int, n: int = 50
) -> np.ndarray:
    """Absolute Pearson correlation of each PC with score, for first n PCs."""
    X_sub = X[:, :max_dim]
    X_c = scaler.transform(X_sub)
    Z = pca.transform(X_c)
    n = min(n, max_dim)
    corr = np.array([abs(np.corrcoef(Z[:, k], y)[0, 1]) for k in range(n)])
    return corr


def dims_for_threshold(cumvar: np.ndarray, threshold: float) -> int:
    idx = np.searchsorted(cumvar, threshold)
    return int(idx) + 1  # 1-indexed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save figures")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save figures (default: script directory)",
    )
    parser.add_argument(
        "--no_show", action="store_true", help="Don't display figures"
    )
    args = parser.parse_args()
    out_dir = (
        args.out_dir if args.out_dir is not None else REPO_ROOT
    )

    print("Loading dataset…")
    X_full, y = load_data()
    print(
        f"  X: {X_full.shape}, y: {y.shape}, score range [{y.min():.3f}, {y.max():.3f}]"
    )

    results = {}
    for name, dim in PROBLEMS.items():
        print(f"\nRunning PCA for {name} (dim={dim})…")
        pca, scaler = run_pca(X_full, dim)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        print(f"  Dims needed for explained variance thresholds:")
        for t in THRESHOLDS:
            d = dims_for_threshold(cumvar, t)
            print(f"    {int(t*100)}%: {d} dims  ({d/dim*100:.1f}% of {dim})")

        print(
            f"  Computing supervised R² (this may take a moment for large dims)…"
        )
        r2 = cumulative_r2(pca, scaler, X_full, y, dim)
        corr = pc_score_corr(pca, scaler, X_full, y, dim, n=50)

        print(f"  Max R² (all {dim} PCs): {r2[-1]:.4f}")
        for t in [0.50, 0.70, 0.80, 0.90]:
            if r2[-1] >= t:
                d = int(np.searchsorted(r2, t)) + 1
                print(f"  R²≥{t}: {d} PCs")

        results[name] = dict(dim=dim, pca=pca, cumvar=cumvar, r2=r2, corr=corr)

    # --- Plot 1: cumulative explained variance ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for name, res in results.items():
        xs = np.arange(1, res["dim"] + 1)
        ax.plot(xs / res["dim"] * 100, res["cumvar"] * 100, label=name)
    for t in THRESHOLDS:
        ax.axhline(t * 100, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("% of dimensions used")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA: Unsupervised (input space)")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 101)

    ax = axes[1]
    for name, res in results.items():
        xs = np.arange(1, res["dim"] + 1)
        ax.plot(xs, res["cumvar"] * 100, label=name)
    for t in THRESHOLDS:
        ax.axhline(t * 100, color="gray", lw=0.8, ls="--")
        for name, res in results.items():
            d = dims_for_threshold(res["cumvar"], t)
            ax.axvline(d, color="gray", lw=0.4, ls=":")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA: Unsupervised (absolute dims)")
    ax.legend()
    ax.set_xlim(0)

    fig.tight_layout()
    if args.save:
        path = out_dir / "po_pca_variance.png"
        fig.savefig(path, dpi=150)
        print(f"\nSaved {path}")

    # --- Plot 1b: standalone cumulative variance (% dims, Okabe-Ito, paper style) ---
    MARKERS = {"PO-128": "o", "PO-256": "s", "PO-512": "^", "PO-768": "D"}
    MARKER_SIZES = {"PO-128": 10, "PO-256": 9, "PO-512": 12, "PO-768": 8}
    THRESHOLD_LABELS = {0.80: "80%", 0.90: "90%", 0.95: "95%", 0.99: "99%"}

    sns.set_theme(style="white", context="paper", font_scale=1.8)
    fig1b, ax1b = plt.subplots(figsize=(4, 3.5))

    for (name, res), color in zip(results.items(), OKABE_ITO):
        xs_pct = np.arange(1, res["dim"] + 1) / res["dim"] * 100
        marker = MARKERS[name]
        ms = MARKER_SIZES[name]
        ax1b.plot(xs_pct, res["cumvar"] * 100, color=color, linewidth=2)
        # legend handle: line + marker combined
        ax1b.plot(
            [],
            [],
            color=color,
            linewidth=2,
            marker=marker,
            markersize=ms,
            markeredgewidth=1.5,
            markeredgecolor="white",
            label=name,
        )
        for t in THRESHOLDS:
            d = dims_for_threshold(res["cumvar"], t)
            x_cross = d / res["dim"] * 100
            y_cross = res["cumvar"][d - 1] * 100
            ax1b.plot(
                x_cross,
                y_cross,
                marker=marker,
                color=color,
                markersize=ms,
                markeredgewidth=1.5,
                markeredgecolor="white",
                zorder=5,
                linestyle="none",
            )

    for t, tlabel in THRESHOLD_LABELS.items():
        ax1b.axhline(t * 100, color="gray", lw=0.8, ls="--")
        ax1b.text(
            101,
            t * 100,
            tlabel,
            va="center",
            ha="left",
            fontsize=plt.rcParams["font.size"] * 0.7,
            color="gray",
        )

    ax1b.set_xlabel("% of dimensions used")
    ax1b.set_ylabel("Cumulative explained variance (%)")
    ax1b.set_xlim(0, 100)
    ax1b.set_ylim(0, 101)
    ax1b.legend(title=None, framealpha=0.9, handlelength=2.0, edgecolor="0.7")
    plt.tight_layout(pad=0.4)

    if args.save:
        path = out_dir / "po_pca_variance_standalone.pdf"
        fig1b.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.08)
        print(f"Saved {path}")

    # --- Plot 2: cumulative R² (supervised) ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes2[0]
    for name, res in results.items():
        xs = np.arange(1, res["dim"] + 1)
        ax.plot(xs / res["dim"] * 100, res["r2"] * 100, label=name)
    ax.set_xlabel("% of PCs used")
    ax.set_ylabel("Cumulative R² (%)")
    ax.set_title("Supervised: R²(PCs → score), % of dims")
    ax.legend()
    ax.set_xlim(0, 100)

    ax = axes2[1]
    for name, res in results.items():
        xs = np.arange(1, res["dim"] + 1)
        ax.plot(xs, res["r2"] * 100, label=name)
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative R² (%)")
    ax.set_title("Supervised: R²(PCs → score), absolute dims")
    ax.legend()

    fig2.tight_layout()
    if args.save:
        path = out_dir / "po_pca_r2.png"
        fig2.savefig(path, dpi=150)
        print(f"Saved {path}")

    # --- Plot 3: per-PC |correlation| with score ---
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, res) in zip(axes3.flat, results.items()):
        xs = np.arange(1, len(res["corr"]) + 1)
        ax.bar(xs, res["corr"], color="steelblue", alpha=0.7)
        ax.set_xlabel("PC index")
        ax.set_ylabel("|Pearson r| with score")
        ax.set_title(f"{name}: per-PC correlation with score (first 50 PCs)")
        ax.set_xlim(0.5, len(xs) + 0.5)
    fig3.tight_layout()
    if args.save:
        path = out_dir / "po_pca_corr.png"
        fig3.savefig(path, dpi=150)
        print(f"Saved {path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
