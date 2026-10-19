"""FreqAmp: amplify the weak coefficients of a feature map's 2-D spectrum.

``freq_amp`` is the function, ``freq_amp_layer_torch`` the ``nn.Module`` and,
when Keras is installed, ``freq_amp_layer_keras`` a Keras layer for the torch
backend (it calls torch ops directly, so it does not work on other backends).
"""
import torch
from torch import nn
from torch import fft

try:
    import keras
except ImportError:  # keras is optional: only freq_amp_layer_keras needs it
    keras = None


def freq_amp(x, amp_factor=2.0, thresh_quantile=0.3, dims=(-2, -1)):
    """Multiply low-magnitude frequency coefficients of ``x`` by ``amp_factor``.

    Steps:
        1. ``z = rfftn(x)`` over ``dims`` (the last FFT dim keeps only the
           non-redundant half, ``W // 2 + 1``).
        2. ``mag = |z|``.
        3. One threshold per leading index (sample and channel for a
           ``(B, C, H, W)`` input): the ``thresh_quantile`` quantile of ``mag``
           over the flattened FFT dims.
        4. Coefficients with ``mag < thresh`` are scaled by ``amp_factor``;
           the rest are left alone.
        5. ``irfftn`` back, with ``s`` set to the original sizes. Without ``s``
           irfftn assumes an even last dimension and returns one column fewer
           for odd widths (7x7 -> 7x6), which breaks the residual add.

    The mask depends on the data, so the layer is non-linear. It has no
    parameters; gradients flow through ``z * scale`` with the mask treated as
    a constant. ``dims`` must be contiguous trailing dimensions.

    Args:
        x: real tensor, e.g. ``(B, C, H, W)``.
        amp_factor: > 1 amplifies weak coefficients, 1 is the identity,
            < 1 suppresses them.
        thresh_quantile: fraction of coefficients counted as weak, e.g. 0.3
            means the bottom 30% by magnitude per sample and channel.
        dims: dimensions to transform, normally the spatial ``(-2, -1)``.

    Returns:
        A real tensor with the same shape and dtype as ``x``.
    """
    dims = tuple(dims)
    sizes = [x.shape[d] for d in dims]
    z = fft.rfftn(x, dim=dims)                                   # (..., H, W//2+1), complex
    mag = torch.abs(z)                                           # (..., H, W//2+1), real
    mag_flat = mag.flatten(start_dim=dims[0], end_dim=dims[-1])  # (..., H*(W//2+1))
    thresh = torch.quantile(mag_flat, thresh_quantile, dim=-1, keepdim=True)
    for _ in range(len(dims) - 1):
        thresh = thresh.unsqueeze(-1)                            # (..., 1, 1) for broadcasting
    mask = (mag < thresh).to(mag.dtype)
    scale = 1.0 + (amp_factor - 1.0) * mask
    return fft.irfftn(z * scale, s=sizes, dim=dims)


class freq_amp_layer_torch(nn.Module):
    """Frequency-domain amplification layer for ``(B, C, H, W)`` input.

    See :func:`freq_amp` for the computation. Amplifying weak coefficients
    also amplifies noise, so the effect has to be validated on the task.
    """

    def __init__(self, amp_factor=2.0, thresh_quantile=0.3, dims=(-2, -1)):
        super().__init__()
        self.amp_factor = float(amp_factor)
        self.thresh_quantile = float(thresh_quantile)
        self.dims = tuple(dims)

    def forward(self, x):
        return freq_amp(x, self.amp_factor, self.thresh_quantile, self.dims)

    def extra_repr(self):
        return (f"amp_factor={self.amp_factor}, "
                f"thresh_quantile={self.thresh_quantile}, dims={self.dims}")


class freq_amp_layer_keras(keras.Layer if keras is not None else object):
    """Keras (torch backend only) FreqAmp layer for channels-last ``(B, H, W, C)`` input."""

    def __init__(self, amp_factor=2.0, thresh_quantile=0.3, dims=(-2, -1), **kwargs):
        if keras is None:
            raise ImportError("freq_amp_layer_keras needs keras with KERAS_BACKEND=torch")
        super().__init__(**kwargs)
        self.amp_factor = float(amp_factor)
        self.thresh_quantile = float(thresh_quantile)
        self.dims = tuple(dims)

    def call(self, x):
        x_t = x.permute(0, 3, 1, 2)                              # (B, C, H, W): FFT on trailing dims
        y = freq_amp(x_t, self.amp_factor, self.thresh_quantile, self.dims)
        return y.permute(0, 2, 3, 1)                             # back to (B, H, W, C)

    def get_config(self):
        base = super().get_config()
        return {**base, "amp_factor": self.amp_factor,
                "thresh_quantile": self.thresh_quantile, "dims": self.dims}
