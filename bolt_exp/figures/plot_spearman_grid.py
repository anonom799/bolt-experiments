"""One 2x1 figure stacking the real-vs-emulated Spearman curves of two families.

plot_real_vs_emulated.py draws a Spearman panel per problem family, each in
its own file. This script draws the same curves for several families as rows
of a single figure (top panel first on the command line, e.g. HPO above DMO),
so the document embeds one graphic with one caption.

Everything but the layout is shared with plot_real_vs_emulated.py: the panels
are built from that module's loading, normalisation and ranking helpers, so a
row here is the same curve as that family's standalone `_spearman` figure.

Usage:
    python -m bolt_exp.figures.plot_spearman_grid \
        --panel plot_configs/hpo_real_vs_emulated.yaml:plot_configs/hpo_okabe_ito.yaml:HPO \
        --panel plot_configs/dm_real_vs_emulated.yaml:plot_configs/dm_okabe_ito.yaml:DMO \
        --out pics/emulator/real_vs_emulated_normalized_regret_rank_spearman_grid.pdf
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml

from bolt_exp.figures.plot_real_vs_emulated import (
    load_regret_frame,
    mean_rank_matrix,
    rank_frame,
    rho_per_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", action="append", required=True, metavar="SPEC",
                        help="One row of the grid, as "
                             "METHODS_CONFIG:STYLE_CONFIG[:PANEL_TITLE]. Repeat "
                             "for each row, top row first. PANEL_TITLE defaults "
                             "to the methods config's `title` up to its first "
                             "colon (so \"HPO: real vs emulated\" -> \"HPO\").")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output path; \"_spearman_grid\" is appended to the stem.")
    parser.add_argument("--shared-scale", action="store_true",
                        help="Pass through to the per-family normalisation (see "
                             "plot_real_vs_emulated.py --shared-scale). Ranks are "
                             "invariant to it, so this only affects the printed bounds.")
    parser.add_argument("--row-height", type=float, default=0.42, metavar="FRAC",
                        help="Height of each subplot as a fraction of the style "
                             "config's figsize height (default: 0.42; plot_grid's "
                             "rows are 0.62).")
    parser.add_argument("--title", default=None,
                        help="Figure-level title. Default: no suptitle.")
    return parser.parse_args()


def _parse_panel(spec: str) -> tuple[Path, Path, str | None]:
    parts = spec.split(":")
    if len(parts) == 2:
        methods, style, panel_title = *parts, None
    elif len(parts) == 3:
        methods, style, panel_title = parts
    else:
        raise ValueError(
            f"--panel {spec!r} should be METHODS_CONFIG:STYLE_CONFIG[:PANEL_TITLE]"
        )
    return Path(methods), Path(style), panel_title


def _short(source_label: str) -> str:
    """Panel-legend form of a source label: "Emulated (DMO-Het)" -> "DMO-Het".

    The full labels are sized for a standalone figure. Here the panel is one
    column of a grid and its title already says the comparison is against the
    emulator, so the wrapper is redundant — and at this width it is what makes
    the legend wide enough to overlap the curves.
    """
    stripped = source_label.removeprefix("Emulated").strip()
    if stripped.startswith("(") and stripped.endswith(")"):
        stripped = stripped[1:-1].strip()
    return stripped or source_label


def _panel_curves(methods_path: Path, shared_scale: bool):
    """Per-emulator (label, steps, rho) curves for one problem family."""
    with methods_path.open() as f:
        methods_config = yaml.safe_load(f)

    sources = tuple(methods_config["sources"].keys())
    source_labels = methods_config.get(
        "source_labels", {name: name.title() for name in sources}
    )
    methods = [entry["label"] for entry in methods_config["methods"]]

    df = load_regret_frame(methods_config, shared_scale)
    rank_df = rank_frame(df, sources)

    steps = np.sort(rank_df["step"].unique())
    real_mean = mean_rank_matrix(rank_df, "real", steps, methods)

    curves = []
    for source in (s for s in sources if s != "real"):
        rho = rho_per_step(real_mean, mean_rank_matrix(rank_df, source, steps, methods))
        curves.append((source_labels[source], steps, rho))
        print(f"Spearman rho ({source_labels[source]} vs real) at iteration "
              f"{steps[-1]}: {rho[-1]:.3f}")
    if not curves:
        raise ValueError(f"{methods_path} has no non-real source to compare against.")

    default_title = str(methods_config.get("title", methods_path.stem)).split(":")[0].strip()
    return curves, methods_config["n_iterations"], default_title


def main() -> None:
    args = parse_args()
    panels = [_parse_panel(spec) for spec in args.panel]

    sns.set_theme(style="white", context="paper", font_scale=1.8)

    # Panel width and spacing follow plot_real_vs_emulated.py's plot_grid (its
    # width formula at one column, its tight_layout pads), so a subplot here is
    # as wide as a subplot there. Rows are shorter than that figure's, though:
    # a rho curve spans a fixed [-1, 1] and needs less height than the regret
    # panels, and stacking full-height rows would make the figure very tall.
    with panels[0][1].open() as f:
        panel_w, panel_h = yaml.safe_load(f)["figsize"]
    n_rows = len(panels)
    # constrained_layout (rather than tight_layout) because it measures the
    # supylabel's own bbox when placing it, so it sits right against the tick
    # labels instead of leaving a gap sized by a guessed rect margin.
    fig, axes = plt.subplots(
        n_rows, 1, squeeze=False,
        figsize=(panel_w * 0.45 + 0.5, panel_h * args.row_height * n_rows),
        layout="constrained",
    )

    for r, (methods_path, _style_path, panel_title) in enumerate(panels):
        ax = axes[r][0]
        curves, n_iterations, default_title = _panel_curves(methods_path, args.shared_scale)
        palette = sns.color_palette(n_colors=len(curves))
        for color, (label, steps, rho) in zip(palette, curves):
            ax.plot(steps, rho, color=color, linewidth=2, label=_short(label))
        # Each family compares against a different set of emulators, so the
        # legend belongs to the panel rather than the figure. A panel with a
        # single emulator needs none at all: its title already names the only
        # comparison drawn, and at this panel width a legend box would sit on
        # top of the curve.
        if len(curves) > 1:
            ax.legend(frameon=False, fontsize="xx-small", loc="lower right",
                      handlelength=1.4, borderaxespad=0.2, labelspacing=0.2)
        ax.set_xlim(0, n_iterations)
        ax.set_title(panel_title or default_title, fontsize="small")
        ax.set_xlabel("BO Iteration" if r == n_rows - 1 else "", fontsize="small")
        # Smaller ticks than the axis labels, matching plot_grid: stacked
        # panels sit close together and full-size tick text would collide
        # across the seam between rows.
        ax.tick_params(labelsize="x-small")

    # One shared y-label instead of one per panel: every row plots the same
    # quantity, and repeating "Spearman's rho" on each row wastes the width
    # this figure doesn't have at panel-grid scale.
    fig.supylabel("Spearman's rho", fontsize="small")

    # Small, tight pads: stacked panels should sit close together.
    fig.get_layout_engine().set(w_pad=0.03, h_pad=0.03)
    if args.title:
        fig.suptitle(args.title, y=1.06)

    out_path = args.out.with_name(args.out.stem + "_spearman_grid" + args.out.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
