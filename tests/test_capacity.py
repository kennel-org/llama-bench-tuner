import csv
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from llama_bench_tuner import capacity
from llama_bench_tuner.capacity import CapacityConfig, PracticalCriteria, parse_depth
from llama_bench_tuner.executor import run_monitored
from llama_bench_tuner.status import BenchStatus

FAKE = r'''#!/bin/sh
# Fake llama-bench: behaviour depends on -d and -ctk so every failure class is exercised.
depth=0; kv=f16
for a in "$@"; do :; done
while [ $# -gt 0 ]; do
  case "$1" in
    --help) printf '%s\n' '-d, --n-depth <n>' '-ctk, --cache-type-k <t>' '-ctv, --cache-type-v <t>' '-ncmoe, --n-cpu-moe' '-fa, --flash-attn'; exit 0;;
    -d) depth=$2; shift;;
    -ctk) kv=$2; shift;;
  esac
  shift
done
if [ "$kv" = "q4_0" ] && [ "$depth" -ge 8192 ]; then
  echo "ggml-cuda.cu:107: ROCm error" >&2; echo "rocdevice.cpp: HW Exception Error" >&2; exit 134
fi
if [ "$depth" -ge 131072 ]; then echo "CUDA error: out of memory" >&2; exit 1; fi
if [ "$kv" = "q8_0" ] && [ "$depth" -ge 65536 ]; then sleep 30; fi
python3 - "$depth" <<'PY'
import sys
d = int(sys.argv[1])
pp = 1000 / (1 + d / 8192)
tg = 20 / (1 + d / 32768)
print("build_commit,n_prompt,n_gen,n_depth,avg_ts")
print(f"abc1234,512,0,{d},{pp}")
print(f"abc1234,0,64,{d},{tg}")
PY
'''


class CapacityTests(unittest.TestCase):
    def _env(self, root: Path):
        binary = root / "llama-bench"
        binary.write_text(FAKE)
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        model = root / "m.gguf"
        model.write_bytes(b"x")
        return binary, model

    def _run(self, root, **overrides):
        binary, model = self._env(root)
        cfg = CapacityConfig(llama_bench=binary, model=model, depths=(0, 4096, 8192, 65536, 131072),
                             kv_types=("f16", "q8_0", "q4_0"), per_run_timeout=5, **overrides)
        rows, code = capacity.run_capacity(cfg, root / "run", root / "tmp", backend=None, runner=run_monitored)
        return cfg, rows, code

    def test_parse_depth_suffixes(self):
        self.assertEqual(parse_depth("4k"), 4096)
        self.assertEqual(parse_depth("128K"), 131072)
        self.assertEqual(parse_depth("1m"), 1048576)
        self.assertEqual(parse_depth("512"), 512)
        with self.assertRaises(ValueError):
            parse_depth("x")

    def test_failure_classification_and_skip_after_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows, code = self._run(Path(tmp), stop_after_fail=True)
            by = {(r["kv_type"], r["context_depth"]): r for r in rows}
            self.assertEqual(code, 0)
            self.assertEqual(by[("f16", 0)]["status"], "success")
            # ROCm HW exception is a runtime abort, never OOM
            self.assertEqual(by[("q4_0", 8192)]["status"], BenchStatus.RUNTIME_ABORT.value)
            self.assertEqual(by[("q4_0", 4096)]["status"], "success")
            self.assertEqual(by[("q4_0", 65536)]["status"], BenchStatus.SKIPPED_AFTER_FAIL.value)
            self.assertEqual(by[("q4_0", 131072)]["status"], BenchStatus.SKIPPED_AFTER_FAIL.value)
            # real OOM is still OOM
            self.assertEqual(by[("f16", 131072)]["status"], BenchStatus.OOM.value)
            self.assertFalse(by[("f16", 131072)]["capacity_ok"])

    def test_timeout_is_recorded_and_stops_that_kv(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, model = self._env(Path(tmp))
            cfg = CapacityConfig(llama_bench=binary, model=model, depths=(0, 65536, 131072),
                                 kv_types=("q8_0",), per_run_timeout=1)
            rows, _ = capacity.run_capacity(cfg, Path(tmp) / "run", Path(tmp) / "tmp", backend=None)
            by = {r["context_depth"]: r["status"] for r in rows}
            self.assertEqual(by, {0: "success", 65536: "timeout", 131072: "skipped_after_fail"})

    def test_depth_rows_are_not_collapsed(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, rows, _ = self._run(Path(tmp))
            f16 = {r["context_depth"]: r for r in rows if r["kv_type"] == "f16" and r["capacity_ok"]}
            self.assertGreater(f16[0]["tg_tps"], f16[65536]["tg_tps"])
            self.assertGreater(f16[0]["pp_tps"], f16[8192]["pp_tps"])
            self.assertEqual(f16[8192]["context"], 8192 + 512 + 64)
            self.assertAlmostEqual(f16[0]["tg_ratio_vs_d0"], 1.0)
            self.assertEqual(f16[0]["runtime_commit"], "abc1234")
            # telemetry unavailable -> None, never guessed
            self.assertIsNone(f16[0]["vram_peak_mb"])

    def test_outputs_and_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run(Path(tmp))
            summary = json.loads((Path(tmp) / "run" / "capacity.json").read_text())
            self.assertEqual(summary["stage"], "capacity")
            self.assertEqual(summary["per_kv"]["q4_0"]["capacity_max_depth"], 4096)
            self.assertEqual(summary["per_kv"]["q4_0"]["first_failure"]["status"], "runtime_abort")
            self.assertEqual(summary["per_kv"]["f16"]["first_failure"]["depth"], 131072)
            self.assertIn("pareto_by_depth", summary)
            with (Path(tmp) / "run" / "capacity.csv").open(newline="") as f:
                header = next(csv.reader(f))
            self.assertEqual(header[:3], ["schema_version", "host", "gpu"])  # legacy fields first
            self.assertIn("practical_candidate", header)

    def test_practical_differs_from_capacity(self):
        crit = PracticalCriteria(min_tg_tps=10.0, tg_ratio=0.5, pp_ratio=0.3)
        ok = capacity.evaluate_practical(completed=True, tg=15, pp=500, tg0=20, pp0=1000, wall_s=10,
                                         vram_peak=15000, vram_total=20000, crit=crit)
        slow = capacity.evaluate_practical(completed=True, tg=8, pp=500, tg0=20, pp0=1000, wall_s=10,
                                           vram_peak=15000, vram_total=20000, crit=crit)
        tight = capacity.evaluate_practical(completed=True, tg=18, pp=900, tg0=20, pp0=1000, wall_s=10,
                                            vram_peak=19900, vram_total=20000, crit=crit)
        self.assertTrue(ok[0])
        self.assertFalse(slow[0])
        self.assertIn("tg 8.0 < min", slow[1])
        self.assertFalse(tight[0])
        self.assertIn("VRAM headroom", tight[1])

    def test_unsupported_values_are_not_executed(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, model = self._env(Path(tmp))
            cfg = CapacityConfig(llama_bench=binary, model=model, depths=(0, 4096), kv_types=("f16", "q8_0"),
                                 flash_attn=0, per_run_timeout=5)
            rows, _ = capacity.run_capacity(cfg, Path(tmp) / "run", Path(tmp) / "tmp", backend=None)
            by = {(r["kv_type"], r["context_depth"]): r for r in rows}
            self.assertEqual(by[("q8_0", 0)]["status"], "unsupported")
            self.assertIn("flash attention", by[("q8_0", 0)]["error"])
            self.assertEqual(by[("f16", 0)]["status"], "success")

    def test_resume_skips_completed_and_rejects_changed_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary, model = self._env(root)
            cfg = CapacityConfig(llama_bench=binary, model=model, depths=(0, 4096), kv_types=("f16",),
                                 per_run_timeout=5)
            capacity.run_capacity(cfg, root / "run", root / "tmp", backend=None)
            calls = []

            def counting(*a, **k):
                calls.append(1)
                return run_monitored(*a, **k)

            rows, _ = capacity.run_capacity(cfg, root / "run", root / "tmp", backend=None, resume=True,
                                            runner=counting)
            self.assertEqual(calls, [])
            self.assertEqual(len(rows), 2)
            changed = CapacityConfig(llama_bench=binary, model=model, depths=(0, 4096), kv_types=("f16",),
                                     per_run_timeout=5, ngl=10)
            with self.assertRaises(SystemExit):
                capacity.run_capacity(changed, root / "run", root / "tmp", backend=None, resume=True)

    def test_dry_run_prints_commands_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary, model = self._env(Path(tmp))
            import io
            from contextlib import redirect_stdout
            buf = io.StringIO()
            with redirect_stdout(buf):
                capacity.main(["--llama-bench", str(binary), "--model", str(model), "--depths", "0", "4k",
                               "--kv", "f16,q8_0", "--dry-run"])
            lines = buf.getvalue().strip().splitlines()
            self.assertEqual(len(lines), 4)
            self.assertIn("-d 4096", lines[1])
            self.assertIn("-ctk q8_0 -ctv q8_0", lines[2])
            self.assertNotIn("-mmp", lines[0])


if __name__ == "__main__":
    unittest.main()
