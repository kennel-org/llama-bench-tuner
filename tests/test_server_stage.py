import json
import stat
import tempfile
import unittest
from pathlib import Path

from llama_bench_tuner.capabilities import probe_llama_bench
from llama_bench_tuner.server_adapter import (ServerSession, decode_throttle, measure_sizes, soak, spec_args,
                                              unique_prompt, window_rates)
from llama_bench_tuner.server_stage import run_server_stage, run_soak_stage
from llama_bench_tuner.stage_common import Context
from llama_bench_tuner.telemetry import GpuInfo

FAKE_SERVER = r'''#!/usr/bin/env python3
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
args = sys.argv[1:]
if "--help" in args:
    print("--spec-type none,draft-mtp,ngram-simple\n--spec-draft-n-max N")
    sys.exit(0)
port = int(args[args.index("--port") + 1])
spec = "--spec-type" in args and args[args.index("--spec-type") + 1] == "draft-mtp"
PP, TG = 4000.0, (400.0 if spec else 200.0)

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode(); self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self._json({"status": "ok"})
        if self.path == "/props": return self._json({"build_info": "bfake-123"})
        self._json({}, 404)
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize": return self._json({"tokens": list(range(len(body["content"].split())))})
        text = body["messages"][0]["content"]; n_prompt = len(text.split()); n = min(int(body.get("max_tokens", 16)), 40)
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        time.sleep(n_prompt / PP)
        for i in range(n):
            time.sleep(1.0 / TG)
            self.wfile.write(("data: " + json.dumps({"choices": [{"delta": {"content": "x"}}]}) + "\n\n").encode()); self.wfile.flush()
        timings = {"prompt_per_second": PP, "predicted_per_second": TG, "predicted_n": n}
        if spec: timings.update(draft_n=30, draft_n_accepted=21)
        self.wfile.write(("data: " + json.dumps({"choices": [], "usage": {"prompt_tokens": n_prompt, "completion_tokens": n}, "timings": timings}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''


class FakeGpu:
    name = "nvidia-smi"

    def __init__(self, throttle=0, clocks=(1500.0,)):
        self.throttle, self.clocks, self.i = throttle, clocks, 0

    def list_gpus(self): return [GpuInfo(0, "fake", 24000, "1", self.name)]
    def used_mib(self, i): return 1000
    def free_mib(self, i): return 20000
    def compute_processes(self, i): return []

    def sample(self, i):
        self.i += 1
        return {"used_mib": 17000.0, "temp_c": 70.0, "sm_clock_mhz": self.clocks[min(self.i, len(self.clocks) - 1)],
                "power_w": 150.0, "throttle_mask": float(self.throttle)}


def make_server(root: Path) -> Path:
    path = root / "llama-server"
    path.write_text(FAKE_SERVER)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class ServerAdapterTests(unittest.TestCase):
    def test_unique_prompt_hits_target_with_server_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = make_server(Path(tmp))
            with ServerSession([str(server)], None, None, Path(tmp) / "s.log", startup_timeout=30) as s:
                text, n = unique_prompt(s, 1000, seed=7)
                self.assertLess(abs(n - 1000), 40)
                self.assertNotEqual(unique_prompt(s, 1000, seed=8)[0], text)  # no shared prefix -> no cache reuse

    def test_measure_sizes_reports_measured_ttft_and_throughput(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = make_server(Path(tmp))
            with ServerSession([str(server)], None, None, Path(tmp) / "s.log", startup_timeout=30) as s:
                res = measure_sizes(s, [400, 2000], reps=2, max_tokens=20)
            self.assertEqual([r["n_ok"] for r in res], [2, 2])
            self.assertGreater(res[1]["ttft_s"]["median"], res[0]["ttft_s"]["median"])  # bigger prompt -> later first token
            self.assertAlmostEqual(res[0]["tg_tps"]["median"], 200.0)

    def test_spec_args_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = make_server(Path(tmp))
            self.assertEqual(spec_args(server, "draft-mtp", 3), ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])
            self.assertIsNone(spec_args(server, "draft-eagle9", 3))
            self.assertIsNone(spec_args(Path(tmp) / "missing", "draft-mtp", 3))

    def test_windows_and_throttle_decoding(self):
        stamps = [i * 0.1 for i in range(100)]  # 10 tok/s for 10 s
        rates = window_rates(stamps, 2.0)
        self.assertEqual(len(rates), 4)  # the trailing 8-10 s window is incomplete (last stamp 9.9 s) and is dropped
        self.assertTrue(all(abs(r[2] - 10.0) < 1e-6 for r in rates))
        self.assertEqual(decode_throttle("0x24"), ["sw_power_cap", "sw_thermal_slowdown"])
        self.assertEqual(decode_throttle(None), [])

    def _soak(self, tmp, backend):
        server = make_server(Path(tmp))
        with ServerSession([str(server)], backend, 0, Path(tmp) / "s.log", startup_timeout=30,
                           record_series=True, poll_interval=0.1) as s:
            return soak(s, duration_s=3.0, window_s=1.0, prompt_tokens=64, max_tokens=40)

    def test_soak_passes_when_clocks_and_speed_are_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._soak(tmp, FakeGpu())
            self.assertGreaterEqual(len(r["windows"]), 2)
            self.assertTrue(r["verdict"]["pass"], r["verdict"])
            self.assertTrue(r["verdict"]["telemetry_available"])

    def test_soak_fails_on_thermal_throttle_bit_or_clock_drop(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._soak(tmp, FakeGpu(throttle=0x20))
            self.assertFalse(r["verdict"]["pass"])
            self.assertIn("sw_thermal_slowdown", r["verdict"]["throttle_reasons"])
        with tempfile.TemporaryDirectory() as tmp:
            r = self._soak(tmp, FakeGpu(clocks=tuple([1500.0] * 14 + [1000.0] * 200)))
            self.assertFalse(r["verdict"]["pass"])
            self.assertGreater(r["verdict"]["sm_clock_drop"], 0.10)

    def test_soak_without_telemetry_never_claims_a_thermal_pass_silently(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._soak(tmp, None)
            self.assertFalse(r["verdict"]["telemetry_available"])  # caller can see the verdict is speed-only


class ServerStageTests(unittest.TestCase):
    def _ctx(self, root: Path) -> Context:
        server = make_server(root)
        return Context(llama_bench=server, model=root / "m.gguf", caps=probe_llama_bench(server), backend=None,
                       gpu=None, gpu_index=None, timeout=30)

    def _profile(self, root: Path, server: Path) -> Path:
        d = root / "run" / "profiles"
        d.mkdir(parents=True)
        p = d / "p-fast.json"
        p.write_text(json.dumps({"profile": "fast", "available": True, "context": 8192, "server_context": 8768,
                                 "commands": {"llama_server": f"{server} -m x -c 8768"}, "mtp_speculation": None}))
        (root / "m.gguf").write_bytes(b"x")
        return p

    def test_server_stage_annotates_profile_with_measured_values_and_mtp_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._ctx(root)
            path = self._profile(root, ctx.llama_bench)
            report = run_server_stage(ctx, root / "run", root / "run" / "server", root / "log", sizes=[500, 2000],
                                      reps=2, max_tokens=20, spec_compare=True, spec_type="draft-mtp",
                                      spec_draft_n_max=3, startup_timeout=30)
            prof = json.loads(path.read_text())
            ms = prof["measured_server"]
            self.assertEqual(ms["ttft_kind"], "measured")
            self.assertEqual(ms["runtime_build_info"], "bfake-123")
            self.assertEqual(len(ms["sizes"]), 2)
            spec = prof["mtp_speculation"]
            self.assertEqual(spec["status"], "active")
            self.assertAlmostEqual(spec["sizes"][0]["tg_speedup"], 2.0, places=1)
            self.assertAlmostEqual(spec["sizes"][0]["draft_acceptance"], 0.7)
            self.assertIn("--spec-type draft-mtp", prof["commands"]["llama_server_mtp"])
            self.assertNotIn("--spec-type", prof["commands"]["llama_server"])  # profile itself is not re-chosen
            self.assertTrue((root / "run" / "server" / "server_rows.csv").exists())
            self.assertIn("fast", report["profiles"])

    def test_sizes_are_clamped_to_the_profile_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._ctx(root)
            self._profile(root, ctx.llama_bench)
            report = run_server_stage(ctx, root / "run", root / "run" / "server", root / "log", sizes=[500, 32768],
                                      reps=1, max_tokens=20, spec_compare=False, spec_type="draft-mtp",
                                      spec_draft_n_max=3, startup_timeout=30)
            self.assertEqual(report["profiles"]["fast"]["sizes"], [500])  # 32768 does not fit server_context 8768

    def test_soak_stage_writes_series_and_annotates_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = self._ctx(root)
            path = self._profile(root, ctx.llama_bench)
            res = run_soak_stage(ctx, root / "run", root / "run" / "soak", root / "log", profile_name="fast",
                                 minutes=0.04, window_s=1.0, startup_timeout=30)
            self.assertTrue((root / "run" / "soak" / "soak_fast.json").exists())
            self.assertIn("soak_server", json.loads(path.read_text()))
            self.assertIn("verdict", res)


if __name__ == "__main__":
    unittest.main()
