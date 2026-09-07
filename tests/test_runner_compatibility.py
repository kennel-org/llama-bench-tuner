import csv
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from llama_bench_tuner import optuna_tune, tune, viz, viz_optuna


LEGACY_GRID_FIELDS = [
    "ok", "ngl", "b", "ub", "fa", "decode_tps", "prefill_tps", "csv", "stderr", "start", "end", "elapsed_sec",
]
LEGACY_OPTUNA_FIELDS = ["number", "value", "state", "ngl", "batch", "fa", "prefill_tps", "csv", "stderr"]


class RunnerCompatibilityTests(unittest.TestCase):
    def _fake_bench(self, root: Path) -> Path:
        binary = root / "llama-bench"
        binary.write_text(
            "#!/bin/sh\n"
            "printf '%s\\n' 'build_commit,n_gpu_layers,no_kv_offload,split_mode,n_prompt,n_gen,n_depth,avg_ts'\n"
            "printf '%s\\n' 'test,16,0,layer,64,0,1024,100.0'\n"
            "printf '%s\\n' 'test,16,0,layer,0,16,1024,50.0'\n"
        )
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        return binary

    def _call(self, func, argv):
        old_argv = sys.argv
        try:
            sys.argv = argv
            func()
        finally:
            sys.argv = old_argv

    def test_grid_optuna_and_visualizations_accept_append_only_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = self._fake_bench(root)
            model = root / "model.gguf"
            model.touch()
            outdir = root / "out"
            tmpdir = root / "tmp"
            grid_run = outdir / "grid" / "stable"
            self._call(tune.main, [
                "llama-tune", "--llama-bench", str(binary), "--model", str(model),
                "--ngl", "16", "--batch", "8", "--flash-attn", "0", "--out-dir", str(outdir),
                "--tmp-dir", str(tmpdir), "--run-dir", str(grid_run),
            ])
            summary = grid_run / "summary_stable.csv"
            with summary.open(newline="") as f:
                grid_rows = list(csv.DictReader(f))
                self.assertEqual(LEGACY_GRID_FIELDS, list(grid_rows[0])[:len(LEGACY_GRID_FIELDS)])
                self.assertEqual("success", grid_rows[0]["status"])
                self.assertNotEqual(grid_rows[0]["requested_placement"], grid_rows[0]["observed_placement"])
            grid_viz = root / "grid-viz"
            self._call(viz.main, ["llama-tune-viz", "--summary", str(summary), "--outdir", str(grid_viz)])
            self.assertTrue((grid_viz / "ranking_decode.csv").exists())
            self.assertTrue((grid_viz / "heatmaps" / "decode_heatmap_fa0.png").exists())

            self._call(optuna_tune.main, [
                "llama-tune-optuna", "--llama-bench", str(binary), "--model", str(model),
                "--ngl-min", "16", "--ngl-max", "16", "--batch-min", "8", "--batch-max", "8",
                "--flash-attn", "0", "--n-trials", "1", "--storage", "none", "--out-dir", str(outdir),
                "--tmp-dir", str(tmpdir),
            ])
            trials = next((outdir / "optuna").glob("*/optuna_trials.csv"))
            best = trials.parent / "optuna_best.json"
            with trials.open(newline="") as f:
                trial_rows = list(csv.DictReader(f))
                self.assertEqual(LEGACY_OPTUNA_FIELDS, list(trial_rows[0])[:len(LEGACY_OPTUNA_FIELDS)])
                self.assertEqual("success", trial_rows[0]["status"])
            opt_viz = root / "opt-viz"
            self._call(viz_optuna.main, ["llama-tune-viz-opt", "--trials", str(trials), "--best", str(best), "--outdir", str(opt_viz)])
            self.assertTrue((opt_viz / "optuna_ranking_decode.csv").exists())
            self.assertTrue((opt_viz / "heatmaps" / "optuna_decode_heatmap_fa0.png").exists())


if __name__ == "__main__":
    unittest.main()
