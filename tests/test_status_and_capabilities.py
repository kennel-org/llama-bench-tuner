import os
import stat
import tempfile
import unittest
from pathlib import Path

from llama_bench_tuner.capabilities import probe_llama_bench
from llama_bench_tuner.status import BenchStatus, classify_bench_outcome


class StatusTests(unittest.TestCase):
    def test_taxonomy(self):
        self.assertEqual(BenchStatus.SUCCESS, classify_bench_outcome(returncode=1, decode_tps=1.0).status)
        self.assertEqual(BenchStatus.OOM, classify_bench_outcome(returncode=1, decode_tps=None, stderr="CUDA out of memory").status)
        self.assertEqual(BenchStatus.UNSUPPORTED, classify_bench_outcome(returncode=1, decode_tps=None, stderr="unrecognized option --n-depth").status)
        self.assertEqual(BenchStatus.TIMEOUT, classify_bench_outcome(returncode=None, decode_tps=None, timed_out=True).status)
        self.assertEqual(BenchStatus.SKIPPED, classify_bench_outcome(returncode=None, decode_tps=None, skip_reason="policy").status)


class CapabilityTests(unittest.TestCase):
    def test_probe_only_claims_help_documented_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "llama-bench"
            binary.write_text("#!/bin/sh\nprintf '%s\\n' '--n-depth --cache-type-k --n-cpu-moe --split-mode --device --flash-attn'\n")
            binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
            result = probe_llama_bench(binary)
        self.assertEqual(0, result.returncode)
        self.assertTrue(result.flags["n_depth"])
        self.assertTrue(result.flags["cache_type_k"])
        self.assertFalse(result.flags["cache_type_v"])
        self.assertFalse(result.flags["main_gpu"])
        self.assertTrue(result.help_sha256)

    def test_missing_binary_is_a_failed_probe_not_an_exception(self):
        result = probe_llama_bench(Path("/definitely/missing/llama-bench"))
        self.assertFalse(any(result.flags.values()))
        self.assertIn("not found", result.error)


if __name__ == "__main__":
    unittest.main()
