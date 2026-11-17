import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location(
    "final_requested", HERE / "publish_final_requested.py"
)
final = importlib.util.module_from_spec(spec)
spec.loader.exec_module(final)


class FinalScopeTests(unittest.TestCase):
    def test_foa_alone_not_complete(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(final, "ANY", Path(d)),
            patch.object(final.stage, "finished", return_value=True),
        ):
            self.assertFalse(final.requested_complete())

    def test_all_three_anyattack_targets_required(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.object(final, "ANY", Path(d)),
            patch.object(final.stage, "finished", return_value=True),
        ):
            path = Path(d) / "task_summary.json"
            row = {"status": "COMPLETE", "planned_images": 150, "returned_queries": 150}
            path.write_text(json.dumps([row] * 3))
            self.assertTrue(final.requested_complete())
            path.write_text(json.dumps([row] * 2))
            self.assertFalse(final.requested_complete())


if __name__ == "__main__":
    unittest.main()
