import unittest

from llama_bench_tuner.parsing import extract_tps_from_csv, parse_bench_csv


class ParsingTests(unittest.TestCase):
    def test_preserves_native_rows_and_legacy_aggregate(self):
        lines = [
            "build_commit,n_gpu_layers,n_prompt,n_gen,n_depth,split_mode,avg_ts",
            "67672dc5,99,512,0,4096,layer,100.5",
            "67672dc5,99,0,128,4096,layer,50.25",
        ]
        rows = parse_bench_csv(lines)
        self.assertEqual(2, len(rows))
        self.assertEqual("67672dc5", rows[0].values["build_commit"])
        self.assertEqual(4096, rows[0].n_depth)
        self.assertEqual("prefill", rows[0].phase)
        self.assertEqual("decode", rows[1].phase)
        self.assertEqual((100.5, 50.25), extract_tps_from_csv(lines))

    def test_keeps_legacy_phase_labels(self):
        lines = ["type,t/s", "pp512,123.0", "tg128,45.0"]
        rows = parse_bench_csv(lines)
        self.assertEqual(["prefill", "decode"], [row.phase for row in rows])
        self.assertEqual((123.0, 45.0), extract_tps_from_csv(lines))


if __name__ == "__main__":
    unittest.main()
