# src/llama_bench_tuner/viz.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

DEFAULT_COMBO = {"ngl": 16, "b": 8, "fa": 0}


def _apply_gray_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.5)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize llama-bench tuning results")
    p.add_argument("--summary", type=Path, required=True, help="Path to summary_* .csv")
    p.add_argument("--outdir", type=Path, default=Path("outfile"))
    return p.parse_args()


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.summary)

    sns.set_theme(style="whitegrid", context="notebook")

    # Ensure numeric typing for relevant columns (CSV may be strings)
    for col in ["ngl", "b", "fa", "decode_tps", "prefill_tps"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ranking by decode tok/s
    rank = df.sort_values(["decode_tps", "prefill_tps"], ascending=[False, False]).reset_index(drop=True)
    rank.to_csv(args.outdir / "ranking_decode.csv", index=False)

    # Best per ngl (for p/n fixed in runs)
    best_per_ngl = (rank.groupby("ngl", as_index=False)
                        .first()
                        .sort_values("ngl"))
    # Save table
    best_per_ngl.to_csv(args.outdir / "best_per_ngl.csv", index=False)

    # Plot decode vs ngl using scatter panels per flash-attn value
    fa_values = sorted(rank["fa"].dropna().unique()) if "fa" in rank else []
    n_cols = max(1, min(2, len(fa_values)))
    n_rows = (len(fa_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 4.2 * n_rows), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    palette = sns.color_palette()
    legend_handles = {}

    targets = fa_values or [None]
    for idx, fa_val in enumerate(targets):
        ax = axes[idx]
        if fa_val is None:
            sub = rank
            title = "All flash-attn"
        else:
            sub = rank[rank["fa"] == fa_val]
            title = f"flash-attn = {int(fa_val)}"
        palette = sns.color_palette(n_colors=sub["b"].nunique() or 1)
        sns.scatterplot(
            data=sub,
            x="ngl",
            y="decode_tps",
            hue="b",
            palette=palette,
            s=85,
            linewidth=0.4,
            edgecolor="black",
            alpha=0.9,
            ax=ax,
        )
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _apply_gray_grid(ax)
        ax.set_xlabel("ngl")
        if idx % n_cols == 0:
            ax.set_ylabel("decode tok/s")
        else:
            ax.set_ylabel("")
        ax.set_title(title)

        # highlight default/best if they appear in this subset
        default_row = sub[
            (sub["ngl"] == DEFAULT_COMBO["ngl"]) &
            (sub["b"] == DEFAULT_COMBO["b"])
        ].head(1)
        if fa_val == DEFAULT_COMBO["fa"] and not default_row.empty:
            ax.scatter(
                default_row["ngl"],
                default_row["decode_tps"],
                s=170,
                color="#2ca02c",
                edgecolor="black",
                linewidth=0.6,
                marker="o",
                label="default params",
                zorder=6,
            )

        if not sub.empty and rank.index[0] in sub.index:
            # best row belongs to this subset
            best_row = sub.loc[[rank.index[0]]]
            ax.scatter(
                best_row["ngl"],
                best_row["decode_tps"],
                s=230,
                color="#e31a1c",
                edgecolor="black",
                linewidth=0.6,
                marker="*",
                label="best decode",
                zorder=7,
            )

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            for handle, label in zip(handles, labels):
                if label and label not in legend_handles:
                    legend_handles[label] = handle
        ax.legend_.remove() if ax.legend_ else None

    # remove unused axes
    for extra_ax in axes[len(targets):]:
        extra_ax.remove()

    if legend_handles:
        fig.legend(
            handles=legend_handles.values(),
            labels=legend_handles.keys(),
            title="batch size",
            loc="upper center",
            ncol=min(4, len(legend_handles)),
            frameon=True,
        )

    fig.suptitle("Decode tok/s by ngl")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    plt.savefig(args.outdir / "decode_vs_ngl.png")
    plt.close()

    # Scatter for ngl/batch/flash-attn combinations (color = decode tok/s)
    scatter_cols = {"ngl", "b", "fa", "decode_tps"}
    scatter_path = args.outdir / "grid_decode_scatter_ngl_batch_fa.png"
    scatter_written = False
    if scatter_cols.issubset(rank.columns):
        fig, ax = plt.subplots(figsize=(7, 5))
        scatter = sns.scatterplot(
            data=rank,
            x="ngl",
            y="b",
            hue="fa",
            style="fa",
            size="decode_tps",
            sizes=(40, 260),
            palette="deep",
            linewidth=0.6,
            edgecolor="white",
            alpha=0.9,
            ax=ax,
        )
        norm = plt.Normalize(rank["decode_tps"].min(), rank["decode_tps"].max())
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("decode tok/s")
        scatter.legend(title="flash-attn", loc="best")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        _apply_gray_grid(ax)
        plt.xlabel("ngl (GPU layers)")
        plt.ylabel("batch size")
        plt.title("Decode tok/s by (ngl, batch, flash-attn)")
        plt.tight_layout()
        plt.savefig(scatter_path)
        scatter_written = True
        plt.close()

    # Heatmaps per flash-attn value for decode tok/s
    heatmap_cols = {"ngl", "b", "fa", "decode_tps"}
    heatmap_dir = args.outdir / "heatmaps"
    heatmap_paths = []
    if heatmap_cols.issubset(rank.columns):
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        for fa_val, sub in rank.groupby("fa"):
            pivot = sub.pivot_table(index="ngl", columns="b", values="decode_tps", aggfunc="max")
            if pivot.empty:
                continue
            plt.figure(figsize=(6, 4.5))
            sns.heatmap(
                pivot.sort_index(),
                annot=True,
                fmt=".2f",
                cmap="mako",
                cbar_kws={"label": "decode tok/s"},
                linewidths=0.5,
                linecolor="#e0e0e0",
            )
            ax = plt.gca()
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#b0b0b0")
                spine.set_linewidth(1.0)
            plt.xlabel("batch size")
            plt.ylabel("ngl (GPU layers)")
            plt.title(f"Decode heatmap (fa={int(fa_val)})")
            plt.tight_layout()
            heat_path = heatmap_dir / f"decode_heatmap_fa{int(fa_val)}.png"
            plt.savefig(heat_path)
            heatmap_paths.append(heat_path)
            plt.close()

    print(f"Saved: {args.outdir/'ranking_decode.csv'}")
    print(f"Saved: {args.outdir/'best_per_ngl.csv'}")
    print(f"Saved: {args.outdir/'decode_vs_ngl.png'}")
    if scatter_written:
        print(f"Saved: {scatter_path}")
    for path in heatmap_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
