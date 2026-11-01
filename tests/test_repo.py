"""Submission-readiness checks for the repository itself. Run with `make test` (it sets LOGIBRIDGE_ROOT)."""
import csv
import os
import unittest

HERE = os.path.dirname(os.path.abspath(__file__)) or "."
ROOT = os.environ.get("LOGIBRIDGE_ROOT") or (os.path.dirname(HERE) if os.path.basename(HERE) == "tests" else HERE)

# every file and folder in the suggested structure (plus the ones this repository adds for the pipeline)
REQUIRED = [
    "README.md", "scenario_architecture/constraint_analysis.md", "scenario_architecture/system_architecture.png",
    "hardware/hardware_justification.md",
    "data_pipeline/simulator.py", "data_pipeline/preprocessing.py", "data_pipeline/training_stats.npy", "data_pipeline/mqtt_architecture.md",
    "training/generate_dataset.py", "training/train_model.py", "training/convert_ptq.py", "training/prune_quantise.py", "training/models",
    "inference/Dockerfile", "inference/inference_service.py", "inference/model.tflite",
    "monitoring/drift_monitor.py", "monitoring/reference_dist.json",
    "deployment/logibridge_deploy.yml",
    "optimisation/benchmark.py", "optimisation/results/benchmark_results.csv", "optimisation/results/pareto_chart.png",
    "reports/phase1_report.pdf", "reports/phase2_report.pdf", "reports/final_report.pdf",
]
# files that cannot be produced without TensorFlow; the README and SUBMISSION_CHECKLIST.md say how to create them
KNOWN_MISSING = {"inference/model.tflite"}


class TestStructure(unittest.TestCase):
    def test_every_file_in_the_suggested_structure_exists_except_the_known_gap(self):
        missing = {p for p in REQUIRED if not os.path.exists(os.path.join(ROOT, p))}
        self.assertTrue(missing <= KNOWN_MISSING, f"unexpected missing files: {sorted(missing - KNOWN_MISSING)}")

    def test_models_folder_holds_the_trained_weights(self):
        models = os.listdir(os.path.join(ROOT, "training/models"))
        self.assertIn("model_weights.npz", models)
        self.assertIn("model_weights_onset.npz", models)

    def test_documents_are_not_empty_and_carry_their_key_numbers(self):
        checks = {"scenario_architecture/constraint_analysis.md": ("259.89", "7,660", "17.3 s"),
                  "hardware/hardware_justification.md": ("Option 1", "2.81 ms", "compute-bound"),
                  "data_pipeline/mqtt_architecture.md": ("logibridge/trucks/{id}/inference", "QoS"),
                  "README.md": ("make test", "SUBMISSION_CHECKLIST.md")}
        for path, needles in checks.items():
            with open(os.path.join(ROOT, path), encoding="utf-8") as f:
                text = f.read()
            for n in needles:
                self.assertIn(n, text, f"{path} should mention {n!r}")

    def test_the_simulator_was_renamed_everywhere(self):
        offenders = []
        for folder, _, files in os.walk(ROOT):
            if any(part in folder for part in (".work", ".git", "__pycache__")):
                continue
            for name in files:
                if name.endswith((".py", ".md", ".sh", ".yml", ".txt")) and name not in ("test_repo.py", "README.md"):
                    with open(os.path.join(folder, name), encoding="utf-8", errors="ignore") as f:
                        if "sensor_simulator" in f.read():
                            offenders.append(os.path.relpath(os.path.join(folder, name), ROOT))
        self.assertEqual(offenders, [])

    def test_makefile_and_staging_tools_exist(self):
        for p in ("Makefile", "tools/stage.sh", "tools/collect.py", "requirements.txt", "requirements-tf.txt", ".gitignore"):
            self.assertTrue(os.path.exists(os.path.join(ROOT, p)), p)


class TestReports(unittest.TestCase):
    def test_three_merged_reports_with_bookmarks(self):
        from pypdf import PdfReader
        expected = {"phase1_report.pdf": 3, "phase2_report.pdf": 5, "final_report.pdf": 8}
        for name, sections in expected.items():
            r = PdfReader(os.path.join(ROOT, "reports", name))
            self.assertGreaterEqual(len(r.pages), 8, name)
            self.assertEqual(len(r.outline), sections, name)

    def test_final_report_has_the_pipeline_mapping_as_section_3(self):
        from pypdf import PdfReader
        outline = [o.title for o in PdfReader(os.path.join(ROOT, "reports", "final_report.pdf")).outline]
        self.assertTrue(outline[2].startswith("Section 3: Task D3"))

    def test_every_individual_report_is_present(self):
        names = os.listdir(os.path.join(ROOT, "reports", "individual"))
        for key in ("A1", "A2", "B1", "B2", "C1", "C2", "C3", "D1", "D2", "D3", "E1", "E2", "E3", "F1", "F2", "F3"):
            self.assertTrue(any(n.startswith(f"LogiEdge_Task_{key}_") for n in names), key)


class TestBenchmarkOutputs(unittest.TestCase):
    def test_results_csv_has_the_five_metrics_and_is_honest_about_what_is_pending(self):
        with open(os.path.join(ROOT, "optimisation/results/benchmark_results.csv")) as f:
            rows = list(csv.DictReader(f))
        cols = rows[0].keys()
        for c in ("mean_latency_ms", "p95_latency_ms", "file_kb", "accuracy_percent", "energy_mj", "critical_recall_percent"):
            self.assertIn(c, cols)
        variants = [r for r in rows if not r["method"].startswith("numpy-self-check")]
        self.assertEqual(len(variants), 3)
        for r in variants:
            if r["method"] == "pending":                       # no invented numbers for unmeasured cells
                self.assertEqual((r["mean_latency_ms"], r["p95_latency_ms"], r["energy_mj"]), ("", "", ""))
            self.assertGreater(float(r["critical_recall_percent"]), 95.0)

    def test_pareto_front(self):
        import make_pareto as mp
        self.assertEqual(mp.pareto_front([(1, 5), (2, 3), (3, 4), (4, 1)]), [0, 1, 3])
        self.assertEqual(mp.pareto_front([(1, 1)]), [0])

    def test_build_rows_marks_measured_variants(self):
        import json
        import build_results_csv as bc
        with open(os.path.join(ROOT, "training/results/variants_results.json")) as f:
            variants = json.load(f)
        fake = [{"variant": "M1", "mean_latency_ms": 0.02, "p95_latency_ms": 0.03, "file_kb": 4.9, "cpu_percent": 12.5,
                 "tdp_watts": 15.0, "energy_mj": 0.0375, "accuracy_percent": 100.0, "critical_recall_percent": 100.0}]
        rows = bc.build_rows(variants, fake, None)
        self.assertEqual([r["method"] for r in rows], ["tflite", "pending", "pending"])
        self.assertEqual(rows[0]["status"], "measured")
        self.assertEqual(rows[0]["mean_latency_ms"], 0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
