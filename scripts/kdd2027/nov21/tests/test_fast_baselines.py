# ruff: noqa: E402
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / "src"))
import methods


class FastBaselineTests(unittest.TestCase):
    def test_mfii_native_update_has_correct_direction_and_bounds(self):
        import torch
        import torch.nn.functional as F

        class Encoder:
            forwards = 0

            def __call__(self, x):
                self.forwards += 1
                return F.normalize(x.mean(dim=(-1, -2)), dim=1)

        clean = torch.tensor([0.5, 0.25, 0.75]).view(1, 3, 1, 1).expand(1, 3, 8, 8).clone()
        reference = torch.tensor([0.1, 0.8, 0.4]).view(1, 3, 1, 1).expand_as(clean)
        adv, trace = methods.generate("attackvlm_mfii", [Encoder()], clean, reference, {}, 50)
        self.assertLessEqual(float((adv - clean).abs().max()), 16 / 255 + 1e-7)
        self.assertGreater(
            trace["cosine_similarity_trace"][-1], trace["cosine_similarity_trace"][0]
        )
        self.assertEqual(trace["model_forward_calls"], 50)

    def test_native_spectrum_round_trip_and_gradient(self):
        import torch

        root = Path("/home/ubuntu/baseline-upstreams/common_weakness")
        methods.verified(root, methods.CWA_PIN)
        module = methods.official_module(
            root / "attacks/AdversarialInput/SpectrumSimulationAttack.py",
            classes={"dct", "idct", "dct_2d", "idct_2d"},
            allowed_imports=("torch", "numpy"),
        )
        x = torch.rand(1, 3, 8, 8, requires_grad=True)
        restored = module.idct_2d(module.dct_2d(x))
        torch.testing.assert_close(restored, x, rtol=1e-5, atol=1e-6)
        self.assertTrue(torch.isfinite(torch.autograd.grad(restored.sum(), x)[0]).all())

    def test_gpt_packet_count_and_legacy_code_preserved(self):
        import json

        root = Path("/home/ubuntu/exper-tran/outputs/kdd2027_baseline_reproduction/nov21/gpt61_low")
        p = root / "packets/m_attack_10/private/manifest.json"
        if not p.exists():
            self.skipTest("Packet not prepared")
        data = json.loads(p.read_text())
        self.assertEqual(len(data["cases"]), 10)
        self.assertEqual(len({r["image_id"] for r in data["cases"]}), 10)
        self.assertEqual(
            sorted([r["transition"] for r in data["cases"]]),
            sorted(["T01", "T03", "T05", "T07", "T09"] * 2),
        )


if __name__ == "__main__":
    unittest.main()
