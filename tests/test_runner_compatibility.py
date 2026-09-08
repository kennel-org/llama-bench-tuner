import csv
import json
import os
import stat
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

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
                "--tmp-dir", str(tmpdir), "--in-dir", str(root / "in"), "--run-dir", str(grid_run),
            ])
            summary = grid_run / "summary_stable.csv"
            with summary.open(newline="") as f:
                grid_rows = list(csv.DictReader(f))
                self.assertEqual(LEGACY_GRID_FIELDS, list(grid_rows[0])[:len(LEGACY_GRID_FIELDS)])
                self.assertEqual("success", grid_rows[0]["status"])
                self.assertNotEqual(grid_rows[0]["requested_placement"], grid_rows[0]["native_reported_placement"])
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

    def test_resume_rejects_a_changed_benchmark_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = self._fake_bench(root)
            model = root / "model.gguf"
            model.touch()
            run_dir = root / "run"
            base_args = [
                "llama-tune", "--llama-bench", str(binary), "--model", str(model),
                "--ngl", "16", "--batch", "8", "--flash-attn", "0", "--out-dir", str(root / "out"),
                "--tmp-dir", str(root / "tmp"), "--in-dir", str(root / "in"), "--run-dir", str(run_dir),
            ]
            self._call(tune.main, base_args)
            metadata = json.loads((run_dir / "run_metadata.json").read_text())
            checkpoint = json.loads((run_dir / "checkpoint.json").read_text())
            self.assertEqual(metadata["benchmark_fingerprint"], checkpoint["benchmark_fingerprint"])

            with self.assertRaisesRegex(SystemExit, "fingerprint differs"):
                self._call(tune.main, base_args + ["--resume", "--prompt", "128"])

    def test_resume_rejects_a_binary_rebuilt_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = self._fake_bench(root)
            model = root / "model.gguf"
            model.touch()
            run_dir = root / "run"
            base_args = [
                "llama-tune", "--llama-bench", str(binary), "--model", str(model),
                "--ngl", "16", "--batch", "8", "--flash-attn", "0", "--out-dir", str(root / "out"),
                "--tmp-dir", str(root / "tmp"), "--in-dir", str(root / "in"), "--run-dir", str(run_dir),
            ]
            self._call(tune.main, base_args)

            # Same path, different content: a rebuilt binary must not be
            # accepted as the same identity, even though the resolved path
            # is unchanged.
            binary.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' 'build_commit,n_gpu_layers,no_kv_offload,split_mode,n_prompt,n_gen,n_depth,avg_ts'\n"
                "printf '%s\\n' 'rebuilt,16,0,layer,64,0,1024,999.0'\n"
                "printf '%s\\n' 'rebuilt,16,0,layer,0,16,1024,888.0'\n"
            )
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

            with self.assertRaisesRegex(SystemExit, "fingerprint differs"):
                self._call(tune.main, base_args + ["--resume"])

    def test_resume_rejects_a_model_replaced_at_the_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = self._fake_bench(root)
            model = root / "model.gguf"
            model.write_bytes(b"a")
            run_dir = root / "run"
            base_args = [
                "llama-tune", "--llama-bench", str(binary), "--model", str(model),
                "--ngl", "16", "--batch", "8", "--flash-attn", "0", "--out-dir", str(root / "out"),
                "--tmp-dir", str(root / "tmp"), "--in-dir", str(root / "in"), "--run-dir", str(run_dir),
            ]
            self._call(tune.main, base_args)

            # Same path, different size: a replaced model must not be
            # accepted as the same identity.
            model.write_bytes(b"a different model")

            with self.assertRaisesRegex(SystemExit, "fingerprint differs"):
                self._call(tune.main, base_args + ["--resume"])

    def test_checkpoint_uses_atomic_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            with mock.patch("llama_bench_tuner.tune.os.replace", wraps=os.replace) as replace:
                tune.write_checkpoint(
                    checkpoint,
                    completed_items=["ngl=16,b=8,fa=0"],
                    rows=[{"ok": True}],
                    started=datetime.now(timezone.utc),
                    benchmark_fingerprint_value="fingerprint",
                )
            self.assertEqual(checkpoint, replace.call_args.args[1])
            state = json.loads(checkpoint.read_text())
            self.assertEqual("fingerprint", state["benchmark_fingerprint"])
            self.assertEqual([], list(checkpoint.parent.glob(".checkpoint.json.*.tmp")))

    def test_checkpoint_replace_failure_preserves_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            checkpoint.write_text('{"previous": true}\n')
            with mock.patch("llama_bench_tuner.tune.os.replace", side_effect=OSError("disk error")):
                with self.assertRaisesRegex(OSError, "disk error"):
                    tune.write_checkpoint(
                        checkpoint,
                        completed_items=["ngl=16,b=8,fa=0"],
                        rows=[{"ok": True}],
                        started=datetime.now(timezone.utc),
                        benchmark_fingerprint_value="fingerprint",
                    )
            self.assertEqual({"previous": True}, json.loads(checkpoint.read_text()))
            self.assertEqual([], list(checkpoint.parent.glob(".checkpoint.json.*.tmp")))


if __name__ == "__main__":
    unittest.main()
