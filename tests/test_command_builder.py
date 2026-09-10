import json
import unittest
from pathlib import Path

from llama_bench_tuner.command_builder import (
    LlamaBenchCommand,
    build_llama_bench_command,
    native_reported_offload_json,
    native_reported_placement_json,
    requested_offload_json,
    requested_placement_json,
)
from llama_bench_tuner.parsing import parse_bench_csv


class CommandBuilderTests(unittest.TestCase):
    def setUp(self):
        self.spec = LlamaBenchCommand(
            llama_bench=Path("/bin/llama-bench"), model=Path("/models/model.gguf"),
            threads=14, ngl=16, batch=8, ubatch=4, prompt=2048, ngen=256,
            mmap=1, flash_attn=1, nkvo=0, split_mode="layer", verbose=True,
        )

    def test_grid_command_stays_in_legacy_order(self):
        self.assertEqual([
            "/bin/llama-bench", "-m", "/models/model.gguf", "-t", "14",
            "-ngl", "16", "-b", "8", "-ub", "4", "-p", "2048", "-n", "256",
            "-mmp", "1", "-o", "csv", "-v", "-fa", "1", "-nkvo", "0", "-sm", "layer",
        ], build_llama_bench_command(self.spec))

    def test_modern_binary_without_mmp_uses_load_mode(self):
        spec = LlamaBenchCommand(**{**self.spec.__dict__, "legacy_mmap_flag": False})
        command = build_llama_bench_command(spec)
        self.assertNotIn("-mmp", command)
        self.assertIn("-lm", command)
        self.assertEqual("mmap", command[command.index("-lm") + 1])

    def test_modern_binary_mmap_disabled_uses_load_mode_none(self):
        spec = LlamaBenchCommand(**{**self.spec.__dict__, "mmap": 0, "legacy_mmap_flag": False})
        command = build_llama_bench_command(spec)
        self.assertEqual("none", command[command.index("-lm") + 1])

    def test_optuna_command_can_keep_its_existing_optional_order(self):
        spec = LlamaBenchCommand(**{
            **self.spec.__dict__, "verbose": False,
            "option_order": ("nkvo", "split_mode", "flash_attn"),
        })
        command = build_llama_bench_command(spec)
        self.assertEqual(["-o", "csv", "-nkvo", "0", "-sm", "layer", "-fa", "1"], command[-8:])

    def test_request_and_native_reported_metadata_are_not_conflated(self):
        rows = parse_bench_csv([
            "n_gpu_layers,n_cpu_moe,no_kv_offload,split_mode,devices,tensor_split,n_prompt,n_gen,avg_ts",
            "20,3,1,row,CUDA0;CUDA1,0.5/0.5,0,32,12.5",
        ])
        self.assertEqual({"n_gpu_layers": 16, "split_mode": "layer"}, json.loads(requested_placement_json(self.spec)))
        self.assertEqual({"no_kv_offload": 0}, json.loads(requested_offload_json(self.spec)))
        self.assertEqual(
            {"n_gpu_layers": "20", "split_mode": "row", "devices": "CUDA0;CUDA1", "tensor_split": "0.5/0.5"},
            json.loads(native_reported_placement_json(rows)),
        )
        self.assertEqual(
            {"n_gpu_layers": "20", "n_cpu_moe": "3", "no_kv_offload": "1"},
            json.loads(native_reported_offload_json(rows)),
        )


if __name__ == "__main__":
    unittest.main()
