import torch
from torch import nn
from torch import fft

class freq_amp_layer(nn.Module):
    """
    Frequency-domain amplification layer.

    This layer applies a real FFT over the last two spatial dimensions,
    identifies low-magnitude frequency coefficients, and amplifies them
    by a fixed factor. The threshold is defined per sample and per channel
    using a magnitude quantile over the 2D frequency map.

    Steps:
        1. Transform input to the frequency domain with `rfftn`.
        2. Compute the magnitude of each frequency coefficient.
        3. Compute a per-(batch, channel) threshold using `thresh_quantile`.
        4. Mark coefficients with magnitude below the threshold as "weak".
        5. Multiply weak coefficients by `amp_factor`.
        6. Transform back to the spatial domain with `irfftn`.

    Args:
        amp_factor (float, default=2.0):
            Multiplicative boost applied to weak frequency coefficients.
            - `amp_factor > 1.0` amplifies weak frequencies
            - `amp_factor = 1.0` leaves them unchanged
            - `amp_factor < 1.0` suppresses them

        thresh_quantile (float, default=0.3):
            Quantile used to define weak coefficients. For example,
            `0.3` means the bottom 30% of frequency magnitudes
            (per sample, per channel) are considered weak.

        dims (tuple[int, int], default=(-2, -1)):
            Dimensions over which to apply the FFT. Typically these are
            the last two spatial dimensions `(H, W)`.

    Input:
        x (Tensor):
            Real-valued tensor of shape `(B, C, H, W)`.

    Output:
        Tensor:
            Real-valued tensor of shape `(B, C, H, W)`.

    Notes:
        - Because `rfftn` is used, only the non-redundant half of the last
          frequency dimension is stored internally.
        - The threshold is computed independently for each batch element
          and channel.
        - Amplifying low-magnitude frequencies may also amplify noise, so
          the effect should be validated for the target task.
    """
    def __init__(self, amp_factor=2.0, thresh_quantile=0.3, dims=(-2, -1)):
        super().__init__()
        self.amp_factor = amp_factor  # how much to boost weak freq
        self.thresh_quantile = thresh_quantile  # e.g., boost bottom 30% magnitudes
        self.dims = dims

    def forward(self, x):
        # fft last 2 dims (H, W)
        z = fft.rfftn(x, dim=self.dims)
        # out shape = (B, C, H, W_r)
        # where W_r = W // 2 + 1
        # out dtype = torch.complex64 

        # compute abs value 
        mag = torch.abs(z)
        # out shape = (B, C H, W_r)
        # out dtype = float32

        # Compute threshold: e.g., 30th percentile of magnitudes

        # flatten last two dims (H, W_r)
        mag_flat_for_quantile = mag.flatten(
            start_dim=self.dims[0],
            end_dim=self.dims[1]
        )
        # out shape = (B, C, H * W_r) 

        # Apply quantile over the last dimension (H * W)
        thresh_flat_output = torch.quantile(
            mag_flat_for_quantile,
            self.thresh_quantile,
            dim=-1,
            keepdim=True
        )
        # out shape = (B, C, 1)


        # to apply correctly, we need a shape like (B, C, 1, 1)
        thresh = thresh_flat_output.unsqueeze(-1)
        # out shape = (B, C, 1, 1)

        # mask to isolate weak frequencies
        mask = (mag < thresh).float()

        # amplify weak frequencies
        scale = 1.0 + (self.amp_factor - 1.0) * mask 

        # appply amplification
        z_amp = z * scale

        # return to spa
        y = fft.irfftn(z_amp, dim=self.dims)  
        
        return y