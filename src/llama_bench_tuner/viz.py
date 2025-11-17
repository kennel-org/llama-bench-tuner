# src/llama_bench_tuner/viz.py
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


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

    # Plot decode vs ngl for all evaluated configurations
    plt.figure(figsize=(7, 4.5))
    ax = plt.gca()
    sns.scatterplot(
        data=rank,
        x="ngl",
        y="decode_tps",
        hue="b",
        style="fa",
        palette="viridis",
        s=90,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
        ax=ax,
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _apply_gray_grid(ax)
    ax.set_xlabel("ngl (GPU layers)")
    ax.set_ylabel("decode tok/s")
    ax.set_title("Decode tok/s per ngl (all configs)")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(title="batch (style = flash-attn)", loc="best")
    plt.tight_layout()
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
