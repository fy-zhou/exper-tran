import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / "publish_foa_stage.py"
spec = importlib.util.spec_from_file_location("publication", PATH)
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)


class PublicationTests(unittest.TestCase):
    def test_no_early_completion_without_summary(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pub, "FOA", Path(d)):
            self.assertFalse(pub.finished())

    def test_partial_never_complete(self):
        with tempfile.TemporaryDirectory() as d, patch.object(pub, "FOA", Path(d)):
            rows = [{"status": "COMPLETE", "planned_images": 150, "returned_queries": 150}] * 2
            rows += [{"status": "PARTIAL", "planned_images": 150, "returned_queries": 149}]
            (Path(d) / "task_summary.json").write_text(json.dumps(rows))
            self.assertFalse(pub.finished())

    def test_summary_alone_does_not_validate_raw_evidence(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertFalse(pub.evaluation_integrity(Path(d)))

    def test_partial_export_preserves_run_identity_and_missing_score(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out = root / "outputs"
            foa = out / "extension/runs/revised_foa"
            co = out / "extension/runs/co"
            foa.mkdir(parents=True)
            co.mkdir(parents=True)
            row = {
                "method": "foa_attack",
                "steps": 300,
                "target": "Qwen3.5-27B",
                "planned_images": 150,
                "returned_queries": 2,
                "status": "PARTIAL",
                "target_hits": 1,
                "ASR": None,
            }
            (foa / "task_summary.json").write_text(json.dumps([row]))
            dest = root / "report"
            with (
                patch.object(pub, "OUT", out),
                patch.object(pub, "FOA", foa),
                patch.object(pub, "CO", co),
                patch.object(pub, "DEST", dest),
                patch.object(pub.subprocess, "check_output", return_value="test-commit"),
            ):
                result = pub.export()
            self.assertEqual(result["status"], "PARTIAL_OR_INVALID")
            data = json.loads((dest / "all_results.json").read_text())
            self.assertEqual(data[0]["run"], "revised_foa")
            self.assertIsNone(data[0]["ASR"])
            self.assertTrue((dest / "structured_evidence.tar.gz").exists())

    def test_no_model_audit_is_not_pass(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(pub, "FOA", Path(d)),
            patch.object(pub, "CO", Path(d)),
        ):
            self.assertFalse(pub.precision_integrity())


if __name__ == "__main__":
    unittest.main()
