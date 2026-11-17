import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from prepare_or_generate import validate_packet  # noqa: E402
from reproduction import checksum, official_module, verify_source  # noqa: E402


class PreparationTests(unittest.TestCase):
    def packet(self, root):
        files = []
        for i in range(4):
            p = root / f"{i}.png"
            p.write_bytes(str(i).encode())
            files.append(p)
        return {
            "images": [
                {
                    "image_id": "one",
                    "clean": str(files[0]),
                    "clean_sha256": checksum(files[0]),
                    "target_references": [
                        {"path": str(p), "sha256": checksum(p)} for p in files[1:]
                    ],
                    "retrieval_provenance": {"selection": "source_only_fixed"},
                }
            ]
        }

    def test_references_valid(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(validate_packet(self.packet(Path(d)), "m_attack_v2")), 1)

    def test_duplicate_reference_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.packet(Path(d))
            r = p["images"][0]["target_references"]
            r[2] = r[1]
            with self.assertRaises(ValueError):
                validate_packet(p, "m_attack_v2")

    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.packet(Path(d))
            r = p["images"][0]
            r["target_references"][0] = {"path": r["clean"], "sha256": r["clean_sha256"]}
            with self.assertRaises(ValueError):
                validate_packet(p, "m_attack_v2")

    def test_missing_reference_not_padded(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.packet(Path(d))
            p["images"][0]["target_references"].pop()
            with self.assertRaises(ValueError):
                validate_packet(p, "m_attack_v2")

    def test_unrelated_import_not_executed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "module.py"
            p.write_text(
                "import absent_api_client\n"
                'raise RuntimeError("entrypoint")\nclass Core:\n value=3\n'
            )
            self.assertEqual(official_module(p, classes={"Core"}).Core.value, 3)

    def test_all_required_source_classes_compile(self):
        import ast

        root = Path("/home/ubuntu/baseline-upstreams")
        for method in ["m_attack_v2", "anyattack"]:
            verify_source(root / method, method)
        for p in [
            "m_attack_v2/attack/base.py",
            "m_attack_v2/surrogates/loss.py",
            "anyattack/models/model.py",
            "anyattack/models/clip/model.py",
        ]:
            compile(ast.parse((root / p).read_text()), p, "exec")

    def test_import_does_not_import_torch_or_init_cuda(self):
        import subprocess

        code = (
            f"import sys;sys.path.insert(0,{str(HERE)!r});"
            'import prepare_or_generate;assert "torch" not in sys.modules'
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
