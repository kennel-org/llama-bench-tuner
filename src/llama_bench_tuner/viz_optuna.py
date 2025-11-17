# src/llama_bench_tuner/viz_optuna.py
import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

DEFAULT_COMBO = {"ngl": 16, "batch": 8, "fa": 0}


def _apply_gray_grid(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color="#cccccc", linewidth=0.6, alpha=0.5)

def parse_args():
    p = argparse.ArgumentParser(description="Visualize Optuna tuning results for llama-bench")
    p.add_argument("--trials", type=Path, default=Path("outfile/optuna_trials.csv"),
                   help="Path to Optuna trials CSV (created by llama-tune-optuna)")
    p.add_argument("--best", type=Path, default=Path("outfile/optuna_best.json"),
                   help="Path to Optuna best JSON (created by llama-tune-optuna)")
    p.add_argument("--outdir", type=Path, default=Path("outfile"),
                   help="Output directory for figures and tables")
    return p.parse_args()

def _save_rank_tables(df: pd.DataFrame, outdir: Path):
    """Save sorted ranking tables to CSV."""
    # Decode ranking (primary)
    rank = df.sort_values(["value", "prefill_tps"], ascending=[False, False]).reset_index(drop=True)
    rank.to_csv(outdir / "optuna_ranking_decode.csv", index=False)

    # Best per ngl (handy overview)
    best_per_ngl = (rank.groupby("ngl", as_index=False).first().sort_values("ngl"))
    best_per_ngl.to_csv(outdir / "optuna_best_per_ngl.csv", index=False)
    return rank, best_per_ngl

def _plot_decode_vs_trial(df: pd.DataFrame, outdir: Path):
    """Plot decode tok/s over trial order."""
    ordered = df.sort_values("number") if "number" in df.columns else df
    plt.figure()
    ax = plt.gca()
    ax.plot(
        ordered["number"],
        ordered["value"],
        marker="o",
        linestyle="-",
        label="trial results",
        zorder=1,
    )

    default_mask = (
        (ordered.get("ngl") == DEFAULT_COMBO["ngl"]) &
        (ordered.get("batch") == DEFAULT_COMBO["batch"]) &
        (ordered.get("fa") == DEFAULT_COMBO["fa"]) if {"ngl","batch","fa"}.issubset(ordered.columns)
        else pd.Series(False, index=ordered.index)
    )
    if default_mask.any():
        default_pts = ordered[default_mask]
        ax.scatter(
            default_pts["number"],
            default_pts["value"],
            color="green",
            s=100,
            marker="o",
            label="default params",
            zorder=2,
        )

    if not ordered["value"].isna().all():
        best_idx = ordered["value"].idxmax()
        best = ordered.loc[[best_idx]]
        ax.scatter(
            best["number"],
            best["value"],
            color="red",
            s=180,
            marker="*",
            label="best decode",
            zorder=3,
        )

    _apply_gray_grid(ax)
    plt.xlabel("trial number")
    plt.ylabel("decode tok/s")
    plt.title("Decode tok/s over trials")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "optuna_decode_vs_trial.png")
    plt.close()

def _plot_decode_vs_ngl(df: pd.DataFrame, outdir: Path):
    """Bar plot: best decode per ngl."""
    best_per_ngl = (df.sort_values("value", ascending=False)
                      .groupby("ngl", as_index=False)
                      .first()
                      .sort_values("ngl"))
    if best_per_ngl.empty:
        return
    plt.figure()
    ax = plt.gca()
    ax.bar(best_per_ngl["ngl"].astype(int), best_per_ngl["value"])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _apply_gray_grid(ax)
    plt.xlabel("ngl (GPU layers)")
    plt.ylabel("best decode tok/s")
    plt.title("Best decode tok/s per ngl")
    plt.tight_layout()
    plt.savefig(outdir / "optuna_best_decode_per_ngl.png")
    plt.close()

def _scatter_decode_by_params(df: pd.DataFrame, outdir: Path):
    """Scatter: decode vs ngl, marker size by batch, marker style by fa."""
    if not {"ngl", "batch", "fa", "value"}.issubset(set(df.columns)):
        return
    plt.figure()
    ax = plt.gca()
    # Create a simple mapping for FA marker styles
    markers = {0: "o", 1: "s"}
    for fa_val in sorted(df["fa"].dropna().unique()):
        sub = df[df["fa"] == fa_val]
        sizes = (sub["batch"].astype(float) * 10.0).clip(lower=10.0)
        ax.scatter(sub["ngl"], sub["value"], s=sizes, marker=markers.get(fa_val, "o"),
                    label=f"fa={fa_val}", alpha=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    _apply_gray_grid(ax)
    plt.xlabel("ngl")
    plt.ylabel("decode tok/s")
    plt.title("Decode tok/s by ngl (size ~ batch, marker ~ fa)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "optuna_decode_scatter_ngl_batch_fa.png")
    plt.close()

def _heatmaps_by_fa(df: pd.DataFrame, outdir: Path):
    required = {"ngl", "batch", "fa", "value"}
    if not required.issubset(df.columns):
        return
    heatmap_dir = outdir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fa_val, sub in df.groupby("fa"):
        pivot = sub.pivot_table(index="ngl", columns="batch", values="value", aggfunc="max")
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
        plt.title(f"Optuna decode heatmap (fa={int(fa_val)})")
        plt.tight_layout()
        heat_path = heatmap_dir / f"optuna_decode_heatmap_fa{int(fa_val)}.png"
        plt.savefig(heat_path)
        plt.close()
        paths.append(heat_path)
    return paths

def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    # Load trials CSV
    df = pd.read_csv(args.trials)
    # Ensure numeric
    for c in ["number", "value", "ngl", "batch", "fa", "prefill_tps"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Save rank tables
    rank, best_per_ngl = _save_rank_tables(df, args.outdir)

    # Load best JSON (optional but nice to print)
    best_info = None
    if args.best.exists():
        try:
            best_info = json.loads(args.best.read_text())
            (args.outdir / "optuna_best_copy.json").write_text(json.dumps(best_info, indent=2))
        except Exception:
            best_info = None

    # Plots
    _plot_decode_vs_trial(df, args.outdir)
    _plot_decode_vs_ngl(rank, args.outdir)
    _scatter_decode_by_params(rank, args.outdir)
    heatmap_paths = _heatmaps_by_fa(rank, args.outdir)

    # Console summary
    print("[OK] Saved:")
    print(f" - {args.outdir/'optuna_ranking_decode.csv'}")
    print(f" - {args.outdir/'optuna_best_per_ngl.csv'}")
    print(f" - {args.outdir/'optuna_decode_vs_trial.png'}")
    print(f" - {args.outdir/'optuna_best_decode_per_ngl.png'}")
    print(f" - {args.outdir/'optuna_decode_scatter_ngl_batch_fa.png'}")
    if heatmap_paths:
        for path in heatmap_paths:
            print(f" - {path}")
    if best_info:
        print(f" - {args.outdir/'optuna_best_copy.json'}")
        print(f"Best (decode tok/s): {best_info.get('value')}, params={best_info.get('params')}")
    else:
        print("Best JSON not found or failed to parse.")

if __name__ == "__main__":
    main()
