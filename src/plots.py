from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DISPLAY_NAMES = {
    "CC": "TE",
    "PCC": "WE",
    "PACC": "PACC",
    "EMQ": "EMQ",
    "KDEyML": "KDEyML",
}


def plot_fairness_estimation_error(
    results_df: pd.DataFrame,
    prevalence_col: str = "true-prev",
    error_col: str = "dd_e",
    quantifier_order: list[str] | None = None,
    natural_prev: float | None = None,
):
    """
    Reproduce the estimation-error boxplot for fairness experiments.

    Parameters
    ----------
    results_df
        DataFrame produced by the fairness evaluation pipeline.
    prevalence_col
        Column containing binary prevalence vectors.
    error_col
        Error column to plot, e.g. 'dd_e' or 'tpr_delta_e'.
    quantifier_order
        Order of quantifiers in the plot and legend.
    natural_prev
        Optional natural prevalence of the positive control label in D3.
        If provided, a vertical reference line is added.
    """
    base_fontsize = 16

    plt.rcParams.update(
        {
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize * 1.1,
            "axes.labelsize": base_fontsize * 1.1,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.75,
        }
    )

    results_df = results_df.copy()
    results_df["s1_prevalence"] = results_df[prevalence_col].apply(
        lambda x: x[1]
    )

    quantifier_order = quantifier_order or [
        "CC",
        "PCC",
        "PACC",
        "EMQ",
        "KDEyML",
    ]
    colors = sns.hls_palette(
        n_colors=len(quantifier_order), h=0.1, l=0.55, s=0.5
    )
    palette = dict(zip(quantifier_order, colors))

    fig, ax = plt.subplots(figsize=(12, 8))

    sns.boxplot(
        x="s1_prevalence",
        y=error_col,
        hue="quantifier",
        data=results_df,
        ax=ax,
        palette=palette,
        hue_order=quantifier_order,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 3, "alpha": 0.5},
        width=0.8,
    )

    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.3, which="major", axis="both")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1)

    ax.set_xlabel(r"Pr$(Y=\oplus, S=\mathrm{Female})$")
    ax.set_ylabel("Estimation Error")

    y_ticks = np.arange(-0.25, 0.25 + 0.05, 0.05)
    ax.set_yticks(y_ticks)

    unique_prevs = sorted(results_df["s1_prevalence"].unique())
    ax.set_xticks(range(len(unique_prevs)))
    ax.set_xticklabels([f"{p:.2f}" for p in unique_prevs])

    ax.set_xlim(-0.5, len(unique_prevs) - 0.5)
    ax.set_ylim(-0.25, 0.25)

    if natural_prev is not None:
        if natural_prev in unique_prevs:
            vline_pos = unique_prevs.index(natural_prev)
        else:
            vline_pos = int(
                np.argmin(np.abs(np.array(unique_prevs) - natural_prev))
            )

        ax.axvline(
            x=vline_pos,
            color="green",
            linestyle="dotted",
            linewidth=2,
            alpha=0.75,
            label=r"$p_{D_3}^{\oplus}(S=\mathrm{Female})$",
        )

    if ax.get_legend() is not None:
        ax.get_legend().remove()

    legend_elements = [
        plt.Line2D([0], [0], color="black", linestyle="--", label="optimal")
    ]

    for quantifier, color in palette.items():
        if quantifier in results_df["quantifier"].unique():
            legend_elements.append(
                plt.Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor=color,
                    alpha=0.7,
                    label=DISPLAY_NAMES.get(quantifier, quantifier),
                )
            )

    if natural_prev is not None:
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                color="green",
                linewidth=2,
                label=r"$p(Y=\oplus, S=\mathrm{Female})$",
            )
        )

    ax.legend(
        handles=legend_elements,
        loc="upper right",
        bbox_to_anchor=(1.325, 1),
        framealpha=0.9,
    )

    fig.subplots_adjust(right=0.78)
    return fig, ax


def summarize_adversarial_results(
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate raw adversarial attack results across runs.

    Expected input columns:
    - quantifier
    - n
    - b
    - macro_f1

    Returns a dataframe with:
    - quantifier
    - n
    - b
    - mean
    - max
    - min
    """
    required_cols = {"quantifier", "n", "b", "macro_f1"}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for adversarial summary: {sorted(missing)}"
        )

    summary = (
        results_df.groupby(["quantifier", "n", "b"])["macro_f1"]
        .agg(["mean", "max", "min"])
        .reset_index()
    )
    return summary


def plot_adversarial_attack(
    results_df: pd.DataFrame,
    quantifier_order: list[str] | None = None,
    n_order: list[int] | None = None,
    use_tex: bool = False,
    ylim: tuple[float, float] = (0.7, 1.0),
):
    """
    Create the adversarial differencing-attack plot for UCI Adult.

    This function accepts either:
    1. raw per-run results with a 'macro_f1' column, or
    2. an already-aggregated dataframe with 'mean', 'max', and 'min' columns.

    Parameters
    ----------
    results_df
        Raw or aggregated adversarial results dataframe.
    quantifier_order
        Order of quantifiers in the legend and plotting.
    n_order
        Order of background sample sizes shown across facets.
    use_tex
        Whether to enable LaTeX text rendering.
    ylim
        y-axis limits for all facets.
    """
    base_fontsize = 18

    sns.set_style(
        "whitegrid",
        {
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "-",
            "axes.edgecolor": ".2",
            "axes.linewidth": 1.2,
        },
    )

    plt.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "sans-serif",
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize * 1.1,
            "axes.labelsize": base_fontsize * 1.1,
            "xtick.labelsize": base_fontsize * 0.9,
            "ytick.labelsize": base_fontsize * 0.9,
            "legend.fontsize": base_fontsize * 0.75,
        }
    )

    df = results_df.copy()

    if {"mean", "max", "min"}.issubset(df.columns):
        plot_df = df
    else:
        plot_df = summarize_adversarial_results(df)

    quantifier_order = quantifier_order or [
        "CC",
        "PCC",
        "PACC",
        "EMQ",
        "KDEyML",
    ]
    n_order = n_order or sorted(plot_df["n"].unique())

    plot_df["quantifier_display"] = plot_df["quantifier"].map(DISPLAY_NAMES)
    display_order = [DISPLAY_NAMES[q] for q in quantifier_order]

    colors = sns.hls_palette(
        n_colors=len(display_order),
        h=0.1,
        l=0.55,
        s=0.5,
    )
    palette = dict(zip(display_order, colors))

    g = sns.FacetGrid(
        plot_df,
        col="n",
        col_wrap=len(n_order),
        height=5,
        aspect=0.9,
        sharex=False,
        sharey=True,
        col_order=n_order,
    )

    def plot_facet(data, **kwargs):
        ax = plt.gca()

        b_values = sorted(data["b"].unique())
        x_positions = {b: i for i, b in enumerate(b_values)}

        n_quantifiers = len(display_order)
        dodge_width = 0.35
        dodge_positions = np.linspace(
            -dodge_width / 2,
            dodge_width / 2,
            n_quantifiers,
        )

        markers = ["o", "s", "D", "^", "v"]

        for q_idx, qid in enumerate(display_order):
            df_q = data[data["quantifier_display"] == qid]
            if len(df_q) == 0:
                continue

            x_vals = []
            means = []
            yerr_lower = []
            yerr_upper = []

            for _, row in df_q.iterrows():
                x_pos = x_positions[row["b"]] + dodge_positions[q_idx]
                x_vals.append(x_pos)

                mean_val = row["mean"]
                means.append(mean_val)

                yerr_lower.append(abs(mean_val - row["min"]))
                yerr_upper.append(abs(row["max"] - mean_val))

            ax.errorbar(
                x_vals,
                means,
                yerr=[yerr_lower, yerr_upper],
                fmt=markers[q_idx],
                label=qid,
                color=palette[qid],
                capsize=4,
                capthick=1.5,
                markersize=8,
                elinewidth=1.5,
                markeredgewidth=1.5,
                markeredgecolor="white",
                alpha=0.9,
            )

        ax.set_xticks(list(x_positions.values()))
        ax.set_xticklabels([f"$B={b}$" for b in b_values])
        ax.set_xlabel(r"Attack Budget ($b$)")
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f"{x:.2f}")
        )

        ax.grid(True, alpha=0.75, linestyle=":", linewidth=1.0, axis="y")
        ax.set_axisbelow(True)

        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_linewidth(1.0)
            ax.spines[spine].set_color("#333333")

        ax.tick_params(axis="both", which="major", length=5, width=1.0)

    g.map_dataframe(plot_facet)
    g.set_axis_labels("", "Macro F1 Score")

    for ax, n_val in zip(g.axes.flat, n_order):
        ax.set_title(f"Background Sample Size: $N={n_val}$", pad=10)

    handles, labels = g.axes.flat[0].get_legend_handles_labels()
    g.fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.525, 1.075),
        frameon=False,
        fancybox=False,
        shadow=False,
        framealpha=0.9,
        edgecolor="#cccccc",
        borderpad=0.5,
        ncol=len(display_order),
    )

    for ax in g.axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
        ax.set_ylim(*ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1] + 0.001, 0.05))

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.025)

    return g.fig, g.axes
