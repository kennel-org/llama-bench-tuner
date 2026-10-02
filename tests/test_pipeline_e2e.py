import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from llama_bench_tuner import pipeline

FAKE = r'''#!/bin/sh
# Fake llama-bench with a smooth performance model so every pipeline stage has something to optimise.
depth=0; kv=f16; ub=512; fa=1; ngl=99; n=64
while [ $# -gt 0 ]; do
  case "$1" in
    --help) printf '%s\n' '-d, --n-depth <n>' '-ctk, --cache-type-k <t>' '-ctv, --cache-type-v <t>' '-ncmoe, --n-cpu-moe' '-fa, --flash-attn' '-sm, --split-mode'; exit 0;;
    -d) depth=$2; shift;; -ctk) kv=$2; shift;; -ub) ub=$2; shift;; -fa) fa=$2; shift;; -ngl) ngl=$2; shift;; -n) n=$2; shift;;
  esac
  shift
done
if [ "$kv" = "f16" ] && [ "$depth" -ge 65536 ]; then echo "CUDA error: out of memory" >&2; exit 1; fi
python3 - "$depth" "$kv" "$ub" "$fa" "$ngl" <<'PY'
import sys
d, kv, ub, fa, ngl = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
kvf = {"f16": 1.0, "q8_0": 0.95, "q4_0": 0.9}[kv]
tg = 30 * kvf / (1 + d / 65536) * (1.0 if ngl >= 99 else 0.5) * (1.0 if fa else 0.8)
pp = 1000 * (ub / 512) ** 0.3 / (1 + d / 16384) * (1.0 if fa else 0.7)
print("build_commit,n_prompt,n_gen,n_depth,avg_ts")
print(f"fake123,512,0,{d},{pp}")
print(f"fake123,0,64,{d},{tg}")
PY
'''


class PipelineEndToEnd(unittest.TestCase):
    def _setup(self, root: Path):
        binary = root / "llama-bench"
        binary.write_text(FAKE)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        server = root / "llama-server"
        server.write_text("#!/bin/sh\necho '-fa, --flash-attn [on|off|auto]'\n")
        server.chmod(server.stat().st_mode | stat.S_IXUSR)
        model = root / "Swift-1.5-Qwen3.8-27B-IQ4_XS.gguf"
        model.write_bytes(b"x")
        space = root / "space.json"
        space.write_text(json.dumps({"batch": [512, 2048], "ubatch": [256, 512], "flash_attn": [0, 1]}))
        return binary, model, space

    def _argv(self, root, binary, model, space, *extra):
        return ["all", "--llama-bench", str(binary), "--model", str(model), "--out-dir", str(root / "out"),
                "--tmp-dir", str(root / "tmp"), "--name", "t", "--depths", "0,8k,32k,64k",
                "--kv", "f16,q8_0,q4_0", "--per-run-timeout", "30", "--grid-space", str(space),
                "--grid-depths", "8k,32k", "--optuna-depths", "32k", "--trials", "8", "--seed", "1",
                "--validate-depths", "8k,32k,64k", "--validate-reps", "3", "--top-k", "3",
                "--fast-depth", "8k", "--balanced-depth", "32k", *extra]

    def test_full_workflow_produces_profiles_pareto_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model, space = self._setup(root)
            with mock.patch.object(pipeline, "detect_backend", return_value=None):
                pipeline.main(self._argv(root, binary, model, space))
            run = root / "out" / "pipeline" / "t"
            cap = json.loads((run / "capacity" / "capacity.json").read_text())
            self.assertEqual(cap["per_kv"]["f16"]["first_failure"]["status"], "oom")
            self.assertEqual(cap["per_kv"]["f16"]["capacity_max_depth"], 32768)
            self.assertEqual(cap["per_kv"]["q8_0"]["capacity_max_depth"], 65536)
            # 64K completes for q8_0 but prefill falls below the practical ratio: capacity != practical
            self.assertEqual(cap["per_kv"]["q8_0"]["practical_candidate_max_depth"], 32768)

            grid = json.loads((run / "grid" / "grid.json").read_text())
            self.assertTrue(grid["promising"])
            self.assertIn("flash_attn", grid["axis_effects"])
            self.assertIn("next_space", grid)

            opt = json.loads((run / "optuna" / "d32768" / "optuna.json").read_text())
            self.assertEqual(opt["mode"], "pareto")
            self.assertTrue(opt["best"])
            self.assertIn("space", opt)

            val = json.loads((run / "validation" / "validation.json").read_text())
            self.assertEqual(val["reps"], 3)
            for cand in val["candidates"]:
                self.assertEqual(cand["depths"]["32768"]["n_runs"], 3)
            self.assertNotIn("65536", {d for c in val["candidates"] for kv, d in [(c["config"]["kv"], d)
                                                                                   for d in c["depths"]]
                                       if c["config"]["kv"] == "f16"})

            profiles = sorted((run / "profiles").glob("*.json"))
            self.assertEqual(len(profiles), 3)
            by = {json.loads(p.read_text())["profile"]: json.loads(p.read_text()) for p in profiles}
            for name in ("fast", "balanced", "long"):
                self.assertTrue(by[name]["available"], name)
                self.assertIn("llama-server", by[name]["commands"]["llama_server"])
                self.assertIn("-fa on", by[name]["commands"]["llama_server"])  # server --help style probed
                self.assertEqual(by[name]["model"]["quant"], "IQ4_XS")
                self.assertGreater(by[name]["expected"]["tg_tps"], 0)
            self.assertLessEqual(by["fast"]["context"], by["balanced"]["context"])
            self.assertLessEqual(by["balanced"]["context"], by["long"]["context"])
            # LONG = deepest *practical* validated depth; 64K completed but is not practical
            self.assertEqual(by["long"]["context"], 32768)
            self.assertEqual(by["long"]["evidence"]["selection"].split("depth ")[1].split(" ")[0], "32768")
            q8_64k = [c["depths"]["65536"] for c in val["candidates"]
                      if c["config"]["kv"] == "q8_0" and "65536" in c["depths"]]
            for summary in q8_64k:
                self.assertFalse(summary["practical"])
                self.assertIn("pp ratio", summary["practical_reason"])
            self.assertTrue(list((run / "pareto").glob("pareto_d*.json")))
            fails = json.loads((run / "failure_summary.json").read_text())
            self.assertTrue(any(e["status"] == "oom" for e in fails["examples"]))

    def test_resume_skips_finished_stages_and_fresh_rerun_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model, space = self._setup(root)
            with mock.patch.object(pipeline, "detect_backend", return_value=None):
                pipeline.main(["capacity", "--llama-bench", str(binary), "--model", str(model), "--out-dir",
                               str(root / "out"), "--tmp-dir", str(root / "tmp"), "--name", "t",
                               "--depths", "0,8k", "--kv", "f16"])
                with self.assertRaises(SystemExit):
                    pipeline.main(["capacity", "--llama-bench", str(binary), "--model", str(model), "--out-dir",
                                   str(root / "out"), "--tmp-dir", str(root / "tmp"), "--name", "t",
                                   "--depths", "0,8k", "--kv", "f16"])
                calls = []
                orig = pipeline.run_capacity
                pipeline.run_capacity = lambda *a, **k: calls.append(1) or orig(*a, **k)
                try:
                    pipeline.main(["capacity", "--llama-bench", str(binary), "--model", str(model), "--out-dir",
                                   str(root / "out"), "--tmp-dir", str(root / "tmp"), "--name", "t",
                                   "--depths", "0,8k", "--kv", "f16", "--resume"])
                finally:
                    pipeline.run_capacity = orig
                self.assertEqual(calls, [])

    def test_gpu_busy_stops_cleanly_with_exit_3(self):
        class Busy:
            name = "nvidia-smi"

            def list_gpus(self):
                from llama_bench_tuner.telemetry import GpuInfo
                return [GpuInfo(0, "gpu0", 24000, "1", self.name)]

            def free_mib(self, i): return 500
            def used_mib(self, i): return 23000
            def sample(self, i): return {}
            def compute_processes(self, i): return ["99, llama-server, 21000 MiB"]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model, space = self._setup(root)
            with mock.patch.object(pipeline, "detect_backend", return_value=Busy()):
                with self.assertRaises(SystemExit) as cm:
                    pipeline.main(["capacity", "--llama-bench", str(binary), "--model", str(model), "--out-dir",
                                   str(root / "out"), "--tmp-dir", str(root / "tmp"), "--name", "t",
                                   "--depths", "0,8k", "--kv", "f16", "--min-free-vram-gib", "18", "--gpu-index", "0"])
                self.assertEqual(cm.exception.code, 3)
            cap_dir = root / "out/pipeline/t/capacity"
            # an interrupted sweep must NOT leave a capacity.json that --resume would treat as complete
            self.assertFalse((cap_dir / "capacity.json").exists())
            partial = json.loads((cap_dir / "capacity_partial.json").read_text())
            self.assertEqual(partial["status_counts"], {"gpu_busy": 1})
            self.assertFalse(partial["complete"])


if __name__ == "__main__":
    unittest.main()
