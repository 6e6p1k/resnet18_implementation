"""Shape and skip-connection tests for the torch models in src/models.

Run:  python -m unittest discover -s tests -v      (or: pytest tests)
The tests are skipped when torch is not installed. Keras is not needed.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = nn = None

if torch is not None:
    from src.models.basic_residual_block import basic_block_torch
    from src.models.freq_amp_layer import freq_amp_layer_torch
    from src.models.freq_amp_res_block import freq_amp_block_torch
    from src.models.resnet18 import resnet18_torch
    from src.models.amp_freq_resnet18 import amp_freq_resnet18_torch

    class _ZeroShortcut(nn.Module):
        """Wraps a block's shortcut so the skip path contributes nothing."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return torch.zeros_like(self.inner(x))

needs_torch = unittest.skipUnless(torch is not None, "torch is not installed")


def _blocks():
    return [
        ("basic", lambda cin, cout, s: basic_block_torch(cin, cout, s)),
        ("freqamp", lambda cin, cout, s: freq_amp_block_torch(cin, cout, s, amp_factor=1.5, thresh_quantile=0.4)),
    ]


@needs_torch
class TestResidualBlocks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shapes(self):
        # 7x7 with stride 1 keeps an odd map: it exercises the irfftn size fix in FreqAmp.
        for name, make in _blocks():
            for cin, cout, stride, hw in [(64, 64, 1, 8), (64, 128, 2, 8), (64, 64, 1, 7), (64, 128, 2, 7)]:
                with self.subTest(block=name, cin=cin, cout=cout, stride=stride, hw=hw):
                    block = make(cin, cout, stride).eval()
                    y = block(torch.randn(2, cin, hw, hw))
                    side = (hw - 1) // stride + 1
                    self.assertEqual(tuple(y.shape), (2, cout, side, side))

    def test_skip_path_is_active(self):
        """Zeroing the shortcut must change the output. The old out.add() bug made the skip a no-op."""
        for name, make in _blocks():
            with self.subTest(block=name):
                block = make(64, 64, 1).eval()
                x = torch.randn(2, 64, 8, 8)
                with_skip = block(x)
                block.shortcut = _ZeroShortcut(block.shortcut)
                without_skip = block(x)
                self.assertFalse(torch.allclose(with_skip, without_skip))

    def test_identity_passes_through_when_main_path_is_zero(self):
        """With conv2 zeroed, bn2 (eval mode, fresh stats) outputs zeros, so the block must return relu(x)."""
        block = basic_block_torch(64, 64, 1).eval()
        nn.init.zeros_(block.conv2.weight)
        x = torch.randn(2, 64, 8, 8)
        torch.testing.assert_close(block(x), torch.relu(x))


@needs_torch
class TestFreqAmpLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_shape_preserved_including_odd_sizes(self):
        layer = freq_amp_layer_torch(amp_factor=2.0, thresh_quantile=0.3)
        for hw in [(8, 8), (7, 7), (15, 15), (4, 6), (1, 1)]:
            with self.subTest(hw=hw):
                x = torch.randn(2, 3, *hw)
                self.assertEqual(tuple(layer(x).shape), tuple(x.shape))

    def test_amp_factor_one_is_identity(self):
        x = torch.randn(2, 3, 8, 8)
        torch.testing.assert_close(freq_amp_layer_torch(amp_factor=1.0)(x), x, atol=1e-5, rtol=1e-5)

    def test_amplification_changes_input_and_has_no_parameters(self):
        layer = freq_amp_layer_torch(amp_factor=2.0, thresh_quantile=0.3)
        x = torch.randn(2, 3, 8, 8)
        self.assertFalse(torch.allclose(layer(x), x))
        self.assertEqual(sum(p.numel() for p in layer.parameters()), 0)

    def test_weak_spectrum_is_boosted_never_attenuated(self):
        x = torch.randn(2, 3, 8, 8)
        y = freq_amp_layer_torch(amp_factor=2.0, thresh_quantile=0.3)(x)
        mx = torch.fft.rfftn(x, dim=(-2, -1)).abs()
        my = torch.fft.rfftn(y, dim=(-2, -1)).abs()
        self.assertTrue(torch.all(my >= mx - 1e-4))
        self.assertTrue(torch.any(my > mx + 1e-4))


@needs_torch
class TestResNet18(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def _models(self):
        return {
            "plain": resnet18_torch(num_classes=10),
            "freqamp": amp_freq_resnet18_torch(num_classes=10, amp_factors=[1.5] * 4, thresh_quantiles=[0.4] * 4),
        }

    def test_forward_shapes(self):
        # With the 7x7/2 stem, 32x32 gives 7x7 maps in layer1 and 64x64 gives 15x15: both odd.
        for name, model in self._models().items():
            model.eval()
            for size in (32, 64):
                with self.subTest(model=name, input=size):
                    self.assertEqual(tuple(model(torch.randn(2, 3, size, size)).shape), (2, 10))

    def test_freqamp_model_contains_freq_amp_layers_and_plain_does_not(self):
        models = self._models()
        count = lambda m: sum(isinstance(mod, freq_amp_layer_torch) for mod in m.modules())
        self.assertEqual(count(models["plain"]), 0)
        self.assertEqual(count(models["freqamp"]), 8)
        n = lambda m: sum(p.numel() for p in m.parameters())
        self.assertEqual(n(models["plain"]), n(models["freqamp"]))


if __name__ == "__main__":
    unittest.main()
