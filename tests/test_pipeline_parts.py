import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

from llama_bench_tuner.command_builder import LlamaBenchCommand, build_llama_bench_command
from llama_bench_tuner.gpu_gate import check_gpu, wait_for_gpu
from llama_bench_tuner.measure import BenchPoint
from llama_bench_tuner.parsing import extract_metrics_by_depth, extract_tps_from_rows, parse_bench_csv
from llama_bench_tuner.pareto import dominates, pareto_by_group, pareto_front
from llama_bench_tuner.schema import PIPELINE_RESULT_FIELDS, pipeline_row
from llama_bench_tuner.status import BenchStatus, classify_bench_outcome
from llama_bench_tuner.telemetry import AmdSysfsBackend, GpuInfo, NvidiaSmiBackend, PeakMonitor, detect_backend


class FakeBackend:
    name = "nvidia-smi"

    def __init__(self, free_sequence, gpus=((0, 24000), (1, 24000))):
        self._free = list(free_sequence)
        self._gpus = gpus
        self.killed = False

    def list_gpus(self):
        # one availability snapshot per check (check_gpu lists GPUs once per poll)
        self._cur = self._free[0]
        if len(self._free) > 1:
            self._free.pop(0)
        return [GpuInfo(i, f"gpu{i}", t, "1", self.name) for i, t in self._gpus]

    def free_mib(self, index):
        value = self._cur
        return value.get(index) if isinstance(value, dict) else value

    def used_mib(self, index):
        return 0

    def sample(self, index):
        return {"used_mib": 123.0, "temp_c": 61.0, "sm_clock_mhz": 1500.0, "power_w": 99.0, "throttle_mask": 4.0}

    def compute_processes(self, index):
        return ["123, llama-server, 21000 MiB"] if index == 1 else []


class StatusTests(unittest.TestCase):
    def test_runtime_abort_is_not_oom(self):
        out = classify_bench_outcome(returncode=134, decode_tps=None,
                                     stderr="ggml-cuda.cu:107: ROCm error\nHW Exception Error")
        self.assertEqual(out.status, BenchStatus.RUNTIME_ABORT)
        self.assertFalse(out.ok)

    def test_oom_still_wins_over_generic_cuda_error(self):
        out = classify_bench_outcome(returncode=1, decode_tps=None, stderr="CUDA error: out of memory")
        self.assertEqual(out.status, BenchStatus.OOM)

    def test_swallowed_oom_is_recognised_from_pool_allocator_backtrace(self):
        stderr = ("ggml-cuda.cu:107: CUDA error\n"
                  "libggml-cuda.so.0(_ZN18ggml_cuda_pool_vmm5allocEmPm+0x352)[0x7b9bf529e842]")
        out = classify_bench_outcome(returncode=134, decode_tps=None, stderr=stderr)
        self.assertEqual(out.status, BenchStatus.OOM)

    def test_hw_exception_with_pool_free_backtrace_is_still_runtime_abort(self):
        out = classify_bench_outcome(returncode=134, decode_tps=None,
                                     stderr="rocdevice.cpp: HW Exception Error\nggml-cuda.cu:107: ROCm error")
        self.assertEqual(out.status, BenchStatus.RUNTIME_ABORT)

    def test_plain_failure_unchanged(self):
        out = classify_bench_outcome(returncode=2, decode_tps=None, stderr="usage: ...")
        self.assertEqual(out.status, BenchStatus.FAILED)

    def test_sigabrt_return_codes(self):
        self.assertEqual(classify_bench_outcome(returncode=-6, decode_tps=None).status, BenchStatus.RUNTIME_ABORT)


class ParsingTests(unittest.TestCase):
    CSV = ["n_prompt,n_gen,n_depth,avg_ts",
           "512,0,0,1000", "0,64,0,20", "512,0,8192,800", "0,64,8192,15"]

    def test_depths_kept_apart_while_legacy_aggregate_unchanged(self):
        rows = parse_bench_csv(self.CSV)
        by = extract_metrics_by_depth(rows)
        self.assertEqual(by[0].tg_tps, 20)
        self.assertEqual(by[8192].tg_tps, 15)
        self.assertEqual(by[8192].pp_tps, 800)
        # legacy behaviour: max over every row (documented limitation)
        self.assertEqual(extract_tps_from_rows(rows), (1000.0, 20.0))


class CommandBuilderTests(unittest.TestCase):
    def _spec(self, **kw):
        base = dict(llama_bench=Path("llama-bench"), model=Path("m.gguf"), threads=8, ngl=99, batch=2048,
                    ubatch=512, prompt=512, ngen=64, mmap=1, flash_attn=1)
        base.update(kw)
        return LlamaBenchCommand(**base)

    def test_legacy_command_is_byte_identical_when_new_fields_unset(self):
        self.assertEqual(
            build_llama_bench_command(self._spec()),
            ["llama-bench", "-m", "m.gguf", "-t", "8", "-ngl", "99", "-b", "2048", "-ub", "512", "-p", "512",
             "-n", "64", "-mmp", "1", "-o", "csv", "-fa", "1"])

    def test_new_options_are_appended_last(self):
        cmd = build_llama_bench_command(self._spec(depth=4096, cache_type_k="q8_0", cache_type_v="q8_0",
                                                   n_cpu_moe=12, repetitions=1, threads=None, mmap=None))
        self.assertNotIn("-t", cmd)
        self.assertNotIn("-mmp", cmd)
        tail = " ".join(cmd[cmd.index("-fa"):])
        self.assertEqual(tail, "-fa 1 -d 4096 -ctk q8_0 -ctv q8_0 -ncmoe 12 -r 1")


class GateTests(unittest.TestCase):
    def test_no_backend_cannot_gate_but_does_not_block(self):
        self.assertTrue(check_gpu(None, min_free_mib=1000).ok)

    def test_picks_gpu_with_most_free_memory_meeting_threshold(self):
        backend = FakeBackend([{0: 7000, 1: 1000}])
        result = check_gpu(backend, min_free_mib=5000)
        self.assertTrue(result.ok)
        self.assertEqual(result.gpu_index, 0)

    def test_busy_reports_occupants_without_touching_them(self):
        backend = FakeBackend([{0: 7000, 1: 1000}])
        result = check_gpu(backend, min_free_mib=19000)
        self.assertFalse(result.ok)
        self.assertIn("gpu1: 123, llama-server, 21000 MiB", result.active_processes)
        self.assertIn("below 19000", result.reason)

    def test_wait_until_free_then_timeout(self):
        sleeps = []
        clock = iter(range(0, 1000, 10))
        backend = FakeBackend([{0: 100, 1: 100}, {0: 100, 1: 100}, {0: 30000, 1: 100}])
        ok = wait_for_gpu(backend, min_free_mib=20000, wait=True, timeout_s=300, poll_s=10,
                          sleep=sleeps.append, clock=lambda: next(clock))
        self.assertTrue(ok.ok)
        self.assertEqual(ok.gpu_index, 0)
        self.assertGreaterEqual(len(sleeps), 2)
        busy = FakeBackend([{0: 100, 1: 100}])
        clock2 = iter(range(0, 1000, 50))
        timed_out = wait_for_gpu(busy, min_free_mib=20000, wait=True, timeout_s=100, poll_s=50,
                                 sleep=lambda s: None, clock=lambda: next(clock2))
        self.assertFalse(timed_out.ok)
        self.assertIn("timeout", timed_out.reason)

    def test_no_wait_flag_returns_immediately(self):
        slept = []
        result = wait_for_gpu(FakeBackend([{0: 1, 1: 1}]), min_free_mib=5000, wait=False, sleep=slept.append)
        self.assertFalse(result.ok)
        self.assertEqual(slept, [])


class TelemetryTests(unittest.TestCase):
    def test_nvidia_smi_parsing_with_na_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "nvidia-smi"
            script.write_text(
                "#!/bin/sh\n"
                'case "$*" in\n'
                '  *--query-gpu=index,name,memory.total,driver_version*) echo "0, Tesla P40, 23040, 580.1";;\n'
                '  *memory.used,temperature.gpu*) echo "1234, 55, [N/A], 70.5, 97, 0x0000000000000004";;\n'
                '  *clocks_event_reasons.active*) echo "[N/A]";;\n'
                '  *clocks_throttle_reasons.active*) echo "0x0000000000000004";;\n'
                '  *memory.used*) echo "1234";;\n'
                '  *memory.free*) echo "21800";;\n'
                "esac\n")
            script.chmod(script.stat().st_mode | stat.S_IXUSR)
            backend = NvidiaSmiBackend(str(script))
            gpus = backend.list_gpus()
            self.assertEqual((gpus[0].index, gpus[0].name, gpus[0].memory_total_mib), (0, "Tesla P40", 23040))
            self.assertEqual(backend.used_mib(0), 1234)
            self.assertEqual(backend.free_mib(0), 21800)
            sample = backend.sample(0)
            self.assertEqual(sample["used_mib"], 1234.0)
            self.assertIsNone(sample["sm_clock_mhz"])  # [N/A] -> None, not 0
            self.assertEqual(sample["power_w"], 70.5)
            self.assertEqual(sample["util_pct"], 97.0)

    def test_amd_sysfs_counts_vram_plus_gtt(self):
        with tempfile.TemporaryDirectory() as tmp:
            dev = Path(tmp) / "card0" / "device"
            dev.mkdir(parents=True)
            mib = 1 << 20
            for name, value in {"mem_info_vram_total": 2048 * mib, "mem_info_vram_used": 100 * mib,
                                "mem_info_gtt_total": 14000 * mib, "mem_info_gtt_used": 900 * mib}.items():
                (dev / name).write_text(str(value))
            backend = AmdSysfsBackend(Path(tmp))
            self.assertEqual(backend.list_gpus()[0].memory_total_mib, 16048)
            self.assertEqual(backend.used_mib(0), 1000)
            self.assertEqual(backend.free_mib(0), 15048)

    def test_monitor_without_backend_reports_ram_only_or_none(self):
        monitor = PeakMonitor(None, None, os.getpid(), interval=0.05)
        monitor.start()
        import time
        time.sleep(0.15)
        peaks = monitor.stop()
        self.assertIsNone(peaks.vram_peak_mib)
        self.assertIn(peaks.status, {"none", "ram_only"})

    def test_monitor_collects_peaks_from_backend(self):
        monitor = PeakMonitor(FakeBackend([{0: 1}]), 0, os.getpid(), interval=0.05)
        monitor.start()
        import time
        time.sleep(0.15)
        peaks = monitor.stop()
        self.assertEqual(peaks.vram_peak_mib, 123)
        self.assertEqual(peaks.temp_max_c, 61.0)
        self.assertEqual(peaks.throttle_reasons, "0x4")

    def test_detect_backend_none_when_nothing_present(self):
        old = os.environ.get("PATH")
        os.environ["PATH"] = ""
        os.environ.pop("BENCH_NVIDIA_SMI", None)
        try:
            backend = detect_backend("amd")
        finally:
            os.environ["PATH"] = old
        self.assertTrue(backend is None or backend.name == "amdgpu-sysfs")


class BusyOnlyTelemetryTests(unittest.TestCase):
    def test_idle_samples_do_not_count_as_clock_drops_or_throttle(self):
        import time

        class Idle(FakeBackend):
            def __init__(self):
                super().__init__([{0: 1}])
                self.n = 0

            def sample(self, index):
                self.n += 1
                if self.n <= 3:  # idle ramp-up: low clock, gpu_idle bit
                    return {"used_mib": 100.0, "temp_c": 30.0, "sm_clock_mhz": 544.0, "power_w": 40.0,
                            "util_pct": 0.0, "throttle_mask": 1.0}
                return {"used_mib": 17000.0, "temp_c": 70.0, "sm_clock_mhz": 1531.0, "power_w": 200.0,
                        "util_pct": 99.0, "throttle_mask": 0.0}
        monitor = PeakMonitor(Idle(), 0, os.getpid(), interval=0.03)
        monitor.start()
        time.sleep(0.4)
        peaks = monitor.stop()
        self.assertEqual(peaks.sm_clock_min_mhz, 1531.0)  # the 544 MHz idle readings are ignored
        self.assertIsNone(peaks.throttle_reasons)
        self.assertEqual(peaks.vram_peak_mib, 17000)


class ParetoSchemaTests(unittest.TestCase):
    ROWS = [
        {"kv": "f16", "pp": 900, "tg": 20, "vram": 16000},
        {"kv": "q8", "pp": 890, "tg": 20, "vram": 15000},   # dominates f16 on vram, ~equal elsewhere? not strictly (pp lower)
        {"kv": "q4", "pp": 850, "tg": 19, "vram": 14000},
        {"kv": "bad", "pp": 800, "tg": 18, "vram": 17000},  # dominated by all
        {"kv": "nodata", "pp": None, "tg": 30, "vram": 1},  # missing metric never dominates
    ]

    def test_front(self):
        front = pareto_front(self.ROWS, maximize=["pp", "tg"], minimize=["vram"])
        kinds = {r["kv"] for r in front}
        self.assertIn("f16", kinds)
        self.assertIn("q4", kinds)
        self.assertNotIn("bad", kinds)
        self.assertIn("nodata", kinds)  # cannot be proven dominated

    def test_dominates_requires_strict_improvement(self):
        a = {"pp": 1, "tg": 1, "vram": 1}
        self.assertFalse(dominates(a, dict(a), maximize=["pp", "tg"], minimize=["vram"]))

    def test_groups(self):
        rows = [{"d": 0, "pp": 1, "tg": 1, "vram": 1}, {"d": 0, "pp": 2, "tg": 2, "vram": 1},
                {"d": 4, "pp": 1, "tg": 1, "vram": 1}]
        groups = pareto_by_group(rows, "d", maximize=["pp", "tg"], minimize=["vram"])
        self.assertEqual(len(groups[0]), 1)
        self.assertEqual(len(groups[4]), 1)

    def test_pipeline_row_rejects_unknown_and_defaults_to_none(self):
        row = pipeline_row(stage="capacity")
        self.assertEqual(set(row), set(PIPELINE_RESULT_FIELDS))
        self.assertIsNone(row["vram_peak_mb"])
        with self.assertRaises(KeyError):
            pipeline_row(bogus=1)


if __name__ == "__main__":
    unittest.main()


class CandidateSelectionTests(unittest.TestCase):
    def test_every_source_depth_contributes_instead_of_the_shallowest_winning(self):
        import json
        from llama_bench_tuner.measure import BenchPoint
        from llama_bench_tuner.validation import collect_candidates

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = BenchPoint().to_dict()

            def study(depth, rows):
                d = root / "optuna" / f"d{depth}"
                d.mkdir(parents=True)
                (d / "optuna.json").write_text(json.dumps({
                    "depth": depth, "base": base, "space": {},
                    "best": [{"params": {"kv": kv, "batch": b}, "attrs": {"tg_tps": tg, "pp_tps": pp, "vram_peak_mb": v}}
                             for kv, b, tg, pp, v in rows]}))
            # at 32K every config is much faster than at 128K; a global ranking would drop all 128K ones
            study(32768, [("f16", 2048, 18, 700, 17000), ("q8_0", 2048, 17.5, 690, 16000),
                          ("q4_0", 2048, 17.2, 680, 15500)])
            study(131072, [("q4_0", 4096, 12, 390, 18000), ("q4_0", 1024, 11.5, 380, 17900)])
            chosen = collect_candidates(root, top_k=4)
            sources = [c["source"] for c in chosen]
            self.assertIn("optuna@131072", sources)
            self.assertIn("optuna@32768", sources)
            self.assertEqual(len({c["point"].config_key() for c in chosen}), 4)


class ReviewRegressionTests(unittest.TestCase):
    def test_grid_defaults_do_not_override_user_choices_and_skip_static_unsupported(self):
        from llama_bench_tuner.grid_stage import build_cases, default_space
        base = BenchPoint(ngl=20, flash_attn=1)
        space = default_space(base)
        self.assertNotIn("ngl", space)  # capacity ran at ngl=20; the grid must stay comparable
        self.assertEqual(space["flash_attn"], [1, 0])
        self.assertEqual(default_space(BenchPoint(flash_attn=0))["flash_attn"], [0])  # explicit FA off stays off
        cases = build_cases({**space, "batch": [1024], "ubatch": [512]}, [("f16", 8192), ("q8_0", 8192)], base, None)
        self.assertTrue(all(c.ngl == 20 for c in cases))
        self.assertFalse(any(c.kv != "f16" and c.flash_attn == 0 for c in cases))  # never generated
        self.assertEqual(len(cases), 3)  # f16 fa0, f16 fa1, q8_0 fa1

    def test_missing_metrics_do_not_make_everything_pareto_optimal(self):
        from llama_bench_tuner.pareto import usable_metrics
        rows = [{"pp": 1, "tg": 5, "ttft": None}, {"pp": 2, "tg": 6, "ttft": None}]
        self.assertEqual(usable_metrics(rows, ["pp", "tg", "ttft"]), ["pp", "tg"])
        front = pareto_front(rows, maximize=usable_metrics(rows, ["pp", "tg"]), minimize=usable_metrics(rows, ["ttft"]))
        self.assertEqual(front, [rows[1]])

    def test_classifier_ignores_benign_words_early_in_a_verbose_log(self):
        noisy = "\n".join(["load: tensor X is unsupported by this backend, using CPU"] * 3 + ["ok line"] * 300)
        stderr = noisy + "\nggml-cuda.cu:107: CUDA error\nlibggml-cuda.so.0(_ZN18ggml_cuda_pool_vmm5allocEmPm+0x352)"
        self.assertEqual(classify_bench_outcome(returncode=134, decode_tps=None, stderr=stderr).status, BenchStatus.OOM)
        stderr2 = noisy + "\nrocdevice.cpp: HW Exception Error"
        self.assertEqual(classify_bench_outcome(returncode=134, decode_tps=None, stderr=stderr2).status,
                         BenchStatus.RUNTIME_ABORT)

    def test_point_roundtrip_keeps_every_axis(self):
        p = BenchPoint(kv="q8_0", depth=4096, split_mode="row", nkvo=1, threads=8, n_cpu_moe=12, prompt=128, ngen=32)
        self.assertEqual(BenchPoint.from_dict(json.loads(json.dumps(p.to_dict()))), p)
        from llama_bench_tuner.validation import _config_of
        row = {"point_json": json.dumps(p.to_dict())}
        self.assertEqual(_config_of(row), p.config_key())
        self.assertNotEqual(p.config_key(), p.replace(split_mode="layer").config_key())

    def test_promising_tolerates_missing_pp(self):
        from llama_bench_tuner.grid_stage import promising
        rows = [{"capacity_ok": True, "context_depth": 8192, "tg_tps": 10.0, "pp_tps": None, "vram_peak_mb": None,
                 "case_key": "a"},
                {"capacity_ok": True, "context_depth": 8192, "tg_tps": 12.0, "pp_tps": None, "vram_peak_mb": None,
                 "case_key": "b"}]
        chosen = {r["case_key"] for r in promising(rows)}  # must not raise on pp=None
        self.assertIn("b", chosen)  # best tg is always kept

    def test_gpu_busy_trial_does_not_consume_trial_budget(self):
        from llama_bench_tuner.capabilities import probe_llama_bench
        from llama_bench_tuner.gpu_gate import GateResult
        from llama_bench_tuner.measure import GpuBusyError
        from llama_bench_tuner.optuna_mo import PrunePolicy, run_study
        from llama_bench_tuner.stage_common import Context
        import optuna
        from tests.test_pipeline_e2e import FAKE

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            binary = root / "llama-bench"
            binary.write_text(FAKE)
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            model = root / "m.gguf"
            model.write_bytes(b"x")
            calls = {"n": 0, "fail_at": 5}

            def gate():
                calls["n"] += 1
                if calls["n"] == calls["fail_at"]:
                    return GateResult(False, 0, 1, 0.0, "busy for the test")
                return GateResult(True, 0, 20000, 0.0, "ok")

            ctx = Context(llama_bench=binary, model=model, caps=probe_llama_bench(binary), backend=None, gpu=None,
                          gpu_index=None, timeout=30, gate=gate)
            out = root / "o"
            m = ctx.measurer(out, root / "log")
            kwargs = dict(space={"batch": [512, 1024, 2048], "ubatch": [256, 512], "kv": ["f16"]}, base=BenchPoint(),
                          depth=8192, n_trials=4, mode="score:fast", objectives=("tg", "pp"), policy=PrunePolicy(),
                          seed=0, population=None, refs={}, study_name="s")
            with self.assertRaises(GpuBusyError):
                run_study(ctx, m, out, **kwargs)
            calls["fail_at"] = -1
            summary = run_study(ctx, m, out, **kwargs)
            counted = summary["trial_states"].get("COMPLETE", 0) + summary["trial_states"].get("PRUNED", 0)
            self.assertEqual(counted, 4)  # the dead trial was re-run, not counted


class RetryTests(unittest.TestCase):
    def _measurer(self, root, results, retry):
        from llama_bench_tuner.capabilities import probe_llama_bench
        from llama_bench_tuner.executor import ExecResult
        from llama_bench_tuner.measure import Measurer
        from llama_bench_tuner.telemetry import Peaks
        binary = root / "llama-bench"
        binary.write_text("#!/bin/sh\necho '-d, --n-depth'\n")
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
        model = root / "m.gguf"
        model.write_bytes(b"x")
        calls = []
        seq = list(results)

        def runner(cmd, **kw):
            calls.append(1)
            kind = seq.pop(0)
            if kind == "ok":
                out = "n_prompt,n_gen,n_depth,avg_ts\n512,0,0,900\n0,64,0,20\n"
                return ExecResult(0, out, "", False, 1.0, "s", "e", Peaks())
            if kind == "ecc":
                return ExecResult(1, "", "cudaMalloc failed: uncorrectable ECC error encountered", False, 1.0, "s", "e", Peaks())
            if kind == "abort":
                return ExecResult(-6, "", "CUDA error: unknown error\nggml_abort", False, 1.0, "s", "e", Peaks())
            return ExecResult(1, "", "CUDA error: out of memory", False, 1.0, "s", "e", Peaks())
        (root / "raw").mkdir(exist_ok=True)
        (root / "log").mkdir(exist_ok=True)
        m = Measurer(binary, model, probe_llama_bench(binary), None, None, None, root / "raw", root / "log",
                     runner=runner, retry_aborts=retry)
        return m, calls

    def test_intermittent_abort_is_retried_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            m, calls = self._measurer(Path(tmp), ["abort", "ok"], retry=1)
            row = m.measure("t", BenchPoint())
            self.assertEqual(row["status"], "success")
            self.assertEqual(row["attempts"], 2)
            self.assertEqual(json.loads(row["attempt_history"]), ["runtime_abort", "success"])
            self.assertEqual(len(calls), 2)
            self.assertTrue(any(Path(tmp, "log").glob("*.attempt1.stderr.txt")))  # evidence of the first failure kept

    def test_persistent_abort_is_still_reported_and_oom_is_never_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            m, calls = self._measurer(Path(tmp), ["abort", "abort"], retry=1)
            row = m.measure("t", BenchPoint())
            self.assertEqual(row["status"], "runtime_abort")
            self.assertEqual(row["attempts"], 2)
        with tempfile.TemporaryDirectory() as tmp:
            m, calls = self._measurer(Path(tmp), ["oom", "ok"], retry=3)
            row = m.measure("t", BenchPoint())
            self.assertEqual(row["status"], "oom")
            self.assertEqual(len(calls), 1)
            self.assertIsNone(row["attempt_history"])


class SplitModelTests(unittest.TestCase):
    def test_split_gguf_size_sums_all_shards_and_never_reports_partial(self):
        from llama_bench_tuner.measure import model_size_bytes
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "m-00001-of-00002.gguf").write_bytes(b"a" * 10)
            self.assertIsNone(model_size_bytes(root / "m-00001-of-00002.gguf"))  # shard 2 missing
            (root / "m-00002-of-00002.gguf").write_bytes(b"b" * 5)
            self.assertEqual(model_size_bytes(root / "m-00001-of-00002.gguf"), 15)
            (root / "single.gguf").write_bytes(b"c" * 7)
            self.assertEqual(model_size_bytes(root / "single.gguf"), 7)


class HardwareFaultTests(unittest.TestCase):
    ECC = ("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 13876.48 MiB on device 0: "
           "cudaMalloc failed: uncorrectable ECC error encountered")

    def test_ecc_error_is_a_hardware_fault_not_oom(self):
        out = classify_bench_outcome(returncode=1, decode_tps=None, stderr=self.ECC)
        self.assertEqual(out.status, BenchStatus.RUNTIME_ABORT)
        self.assertTrue(out.error.startswith("GPU hardware fault"))

    def test_fault_is_not_retried_and_raises(self):
        from llama_bench_tuner.measure import GpuFaultError
        t = RetryTests()
        with tempfile.TemporaryDirectory() as tmp:
            m, calls = t._measurer(Path(tmp), ["ecc", "ok"], retry=3)
            with self.assertRaises(GpuFaultError):
                m.measure("t", BenchPoint())
            self.assertEqual(len(calls), 1)


class DeepRepsTests(unittest.TestCase):
    def test_reps_for_depth(self):
        from llama_bench_tuner.validation import _reps_for
        self.assertEqual(_reps_for(32768, 3, 2, 131072), 3)
        self.assertEqual(_reps_for(131072, 3, 2, 131072), 2)
        self.assertEqual(_reps_for(131072, 3, None, 131072), 3)
