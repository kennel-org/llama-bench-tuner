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
    for path in heatmap_paths:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
