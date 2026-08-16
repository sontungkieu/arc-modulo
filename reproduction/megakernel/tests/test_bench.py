from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LoopBenchmarkTest(unittest.TestCase):
    def test_torch_cpu_eager_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "loop.json"
            command = [
                sys.executable,
                str(ROOT / "bench_loop.py"),
                "--backend",
                "torch",
                "--device",
                "cpu",
                "--modes",
                "eager",
                "--particles",
                "2",
                "--steps",
                "3",
                "--height",
                "8",
                "--width",
                "8",
                "--channels",
                "4",
                "--blocks",
                "2",
                "--warmups",
                "1",
                "--repeats",
                "1",
                "--output",
                str(output),
            ]
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(payload["true_megakernel"])
            self.assertEqual(payload["results"][0]["status"], "ok")
            self.assertGreater(payload["results"][0]["particle_steps_per_s"], 0)


class ResultComparisonTest(unittest.TestCase):
    def test_acceptance_requires_speed_and_correctness(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            common = {
                "benchmark": "fk_steering_real_pipeline_denoising",
                "device_map_requested": "single",
                "status": "ok",
                "device_memory": {"0": {"peak_allocated_bytes": 123}},
                "particle_steps_per_s": 10.0,
                "deterministic_output_hashes": ["same"],
                "output_probes_first_256": [[0.0, 1.0]],
            }
            eager = {**common, "mode": "eager", "median_s": 10.0}
            optimized = {**common, "mode": "compile-unet", "median_s": 8.0}
            eager_path = root / "eager.json"
            optimized_path = root / "optimized.json"
            result_path = root / "comparison.json"
            eager_path.write_text(json.dumps(eager), encoding="utf-8")
            optimized_path.write_text(json.dumps(optimized), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "compare_results.py"),
                    "--input",
                    str(eager_path),
                    "--input",
                    str(optimized_path),
                    "--output",
                    str(result_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["accepted_modes"], ["compile-unet:single"])

    def test_missing_optional_input_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "compile.json"
            result_path = root / "comparison.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "compare_results.py"),
                    "--input",
                    str(missing),
                    "--output",
                    str(result_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["missing_inputs"], [str(missing)])
            self.assertIn("no eager baseline", payload["recommendation"])


if __name__ == "__main__":
    unittest.main()
