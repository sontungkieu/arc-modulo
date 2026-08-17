from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_paired_module():
    spec = importlib.util.spec_from_file_location(
        "bench_fk_paired", ROOT / "bench_fk_paired.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_component_comparison_module():
    spec = importlib.util.spec_from_file_location(
        "compare_fk_components", ROOT / "compare_fk_components.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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


class PairedNumericsTest(unittest.TestCase):
    def test_difference_uses_allclose_contract(self) -> None:
        import numpy as np

        module = _load_paired_module()
        reference = np.array([1.0, 10.0], dtype=np.float32)
        candidate = np.array([1.001, 10.04], dtype=np.float32)
        result = module.difference(reference, candidate, atol=0.002, rtol=0.005)
        self.assertTrue(result["allclose"])
        self.assertEqual(result["violation_fraction"], 0.0)

    def test_trace_reports_first_failed_step(self) -> None:
        import numpy as np

        module = _load_paired_module()
        reference = [np.zeros(2, dtype=np.float32) for _ in range(3)]
        candidate = [array.copy() for array in reference]
        candidate[1][0] = 1.0
        result = module.trace_difference(reference, candidate, atol=1e-3, rtol=1e-3)
        self.assertFalse(result["all_steps_allclose"])
        self.assertEqual(result["first_failed_step"], 1)


class ComponentPlacementCanaryTest(unittest.TestCase):
    def test_pixel_difference_detects_exact_and_tolerated_matches(self) -> None:
        import numpy as np

        module = _load_component_comparison_module()
        reference = np.array([[[[0, 1, 2]]]], dtype=np.uint8)
        exact = module.pixel_difference(reference, reference.copy(), atol=0, rtol=0)
        self.assertTrue(exact["exact"])
        self.assertTrue(exact["allclose"])

        candidate = reference.copy()
        candidate[0, 0, 0, 2] = 3
        tolerated = module.pixel_difference(reference, candidate, atol=1, rtol=0)
        self.assertFalse(tolerated["exact"])
        self.assertTrue(tolerated["allclose"])
        self.assertEqual(tolerated["max_abs"], 1.0)

    def test_trace_difference_requires_matching_resampling_indices(self) -> None:
        module = _load_component_comparison_module()
        reference = [
            {
                "sampling_idx": 4,
                "resampled": True,
                "indices": [0, 1],
                "raw_rewards": [0.1, 0.2],
                "candidate_rewards": [0.1, 0.2],
                "weights": [1.0, 2.0],
                "ess": None,
            }
        ]
        candidate = [{**reference[0], "indices": [1, 1]}]
        result = module.trace_difference(reference, candidate, atol=0, rtol=0)
        self.assertFalse(result["allclose"])
        self.assertFalse(result["discrete_match"])
        self.assertEqual(result["first_failed_checkpoint"], 0)

    def test_summary_requires_parity_and_full_split_fit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parity_path = root / "parity.json"
            full_path = root / "full.json"
            output_path = root / "summary.json"
            parity_path.write_text(
                json.dumps(
                    {
                        "numerical_comparison_available": True,
                        "numerical_match": True,
                    }
                ),
                encoding="utf-8",
            )
            full_path.write_text(
                json.dumps(
                    {
                        "split_fits": True,
                        "single_fits": False,
                        "verdict": "split_fits_single_oom",
                    }
                ),
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "summarize_fk_component_canary.py"),
                    "--parity",
                    str(parity_path),
                    "--full",
                    str(full_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["decision"], "split-components-feasible")


if __name__ == "__main__":
    unittest.main()
