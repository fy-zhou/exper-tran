import ast
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from reproduction import official_module, v2_core  # noqa: E402


class AdmissionAndLossTests(unittest.TestCase):
    def test_delegation_accepts_file_object_and_preserves_other_locks(self):
        tree = ast.parse((HERE / "admitted_worker.py").read_text())
        function = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "delegated"
        )
        seen = []
        root = Path("/output")
        namespace = {
            "os": types.SimpleNamespace(
                readlink=lambda p: seen.append(p) or str(root / "reproduction_owner.lock"),
                getppid=lambda: 123,
            ),
            "OUT": root,
            "parent": 123,
            "original": lambda *args: "real_lock",
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), "admission_test", "exec"),
            namespace,
        )
        with tempfile.TemporaryFile() as handle:
            self.assertIsNone(namespace["delegated"](handle, 1))
            self.assertEqual(seen[-1], f"/proc/self/fd/{handle.fileno()}")
        namespace["os"].readlink = lambda p: "/different.lock"
        self.assertEqual(namespace["delegated"](7, 1), "real_lock")

    def test_v2_singleton_fix_preserves_batched_loss_and_gradient(self):
        try:
            import einops  # noqa: F401
        except ImportError:
            self.skipTest("Optional isolated dependencies unavailable")
        import torch

        root = Path("/home/ubuntu/baseline-upstreams/m_attack_v2")
        _, fixed = v2_core(root)
        original = official_module(
            root / "surrogates/loss.py",
            classes={"cosine_similarity", "EnsAlignmentLoss", "EnsWeightedMultiAlignmentLoss"},
            allowed_imports=("typing", "einops", "torch"),
            injected={"BaseFeatureExtractor": torch.nn.Module},
        )

        class Features(torch.nn.Module):
            def forward(self, x, return_dict=False):
                return x.mean(dim=(-2, -1))

        x = torch.rand(2, 3, 4, 4, requires_grad=True)
        target = torch.rand(6, 3, 4, 4)
        expected = original.EnsWeightedMultiAlignmentLoss([Features()], 0.3)(x, target, 3)
        actual = fixed.EnsWeightedMultiAlignmentLoss([Features()], 0.3)(x, target, 3)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, x)[0], torch.autograd.grad(expected, x)[0]
        )
        single = x[:1].detach().requires_grad_()
        loss = fixed.EnsWeightedMultiAlignmentLoss([Features()], 0.3)(single, target[:3], 3)
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss, single)[0]).all())


if __name__ == "__main__":
    unittest.main()
