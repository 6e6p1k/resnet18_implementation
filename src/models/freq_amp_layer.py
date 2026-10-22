import torch
from torch import nn
from torch import fft
import keras 
from keras import ops


class freq_amp_layer_torch(nn.Module):
    # boosts weak (low-magnitude) freq components
    def __init__(self, amp_factor=2.0, thresh_quantile=0.3, dims=(-2, -1)):
        super().__init__()
        self.amp_factor = amp_factor  # weak frequencies boosting coefficient
        self.thresh_quantile = thresh_quantile
        self.dims = dims

    def forward(self, x):
        z = fft.rfftn(x, dim=self.dims)  # enter frequency space, (B, C, H, W//2+1)
        mag = torch.abs(z)               # (B, C, H, W//2+1)

        # quantile only takes one dim, so flatten H and W first
        mag_flat_for_quantile = mag.flatten(start_dim=self.dims[0], end_dim=self.dims[1])  # (B, C, H*(W//2+1))

        # compute threshold: eg, 30th percentile of magnitudes
        thresh_flat_output = torch.quantile(
            mag_flat_for_quantile,
            self.thresh_quantile,
            dim=-1,
            keepdim=True
        )  # (B, C, 1)
        thresh = thresh_flat_output.unsqueeze(-1)  # (B, C, 1, 1)

        # amplify weak frequencies using masking
        mask = (mag < thresh).float()
        scale = 1.0 + (self.amp_factor - 1.0) * mask
        z_amp = z * scale

        y = fft.irfftn(z_amp, s=[x.shape[d] for d in self.dims], dim=self.dims)  # back to spatial, (B, C, H, W)
        return y
    

class freq_amp_layer_keras(keras.Layer):
    def __init__(self, amp_factor=2.0, thresh_quantile=0.3, dims=(-2, -1), **kwargs):
        super().__init__(**kwargs)
        self.amp_factor = amp_factor
        self.thresh_quantile = thresh_quantile
        self.dims = dims

    def call(self, x):
        # channels last -> channels first, fft wants trailing dims
        x_t = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        H, W = x_t.shape[-2], x_t.shape[-1]
        z = fft.rfftn(x_t, s=(H, W), dim=(-2, -1))  # enter frequency space, (B, C, H, W//2+1)

        mag = torch.abs(z)                    # (B, C, H, W//2+1)
        mag_flat = mag.flatten(start_dim=-2)  # (B, C, H*(W//2+1))

        thresh = torch.quantile(mag_flat, self.thresh_quantile, dim=-1, keepdim=True)  # (B, C, 1)
        thresh = thresh.unsqueeze(-1)  # (B, C, 1, 1)

        # amplify weak frequencies using masking
        mask = (mag < thresh).float()
        scale = 1.0 + (self.amp_factor - 1.0) * mask
        z_amp = z * scale

        y = fft.irfftn(z_amp, s=(H, W), dim=(-2, -1))  # back to spatial, (B, C, H, W)
        return y.permute(0, 2, 3, 1)  # (B, H, W, C)

    def get_config(self):
        base = super().get_config()
        return {**base, "amp_factor": self.amp_factor, 
                "thresh_quantile": self.thresh_quantile, "dims": self.dims}


class learnable_freq_amp_layer_keras(keras.Layer):
    def __init__(self, thresh_quantile=0.3, **kwargs):
        super().__init__(**kwargs)
        self.thresh_quantile = thresh_quantile

    def build(self, input_shape):
        # one gain per channel, starts at 1 so the layer starts as identity
        self.gain = self.add_weight(shape=(input_shape[-1],), initializer="ones", trainable=True, name="gain")

    def call(self, x):
        x_t = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)

        H, W = x_t.shape[-2], x_t.shape[-1]
        z = fft.rfftn(x_t, s=(H, W), dim=(-2, -1))  # enter frequency space, (B, C, H, W//2+1)

        mag = torch.abs(z)                    # (B, C, H, W//2+1)
        mag_flat = mag.flatten(start_dim=-2)  # (B, C, H*(W//2+1))

        thresh = torch.quantile(mag_flat, self.thresh_quantile, dim=-1, keepdim=True)  # (B, C, 1)
        thresh = thresh.unsqueeze(-1)  # (B, C, 1, 1)

        # amplify weak frequencies using masking
        mask = (mag < thresh).float()
        gain = self.gain.value.view(1, -1, 1, 1)  # (1, C, 1, 1)
        scale = 1.0 + (gain - 1.0) * mask
        z_amp = z * scale

        y = fft.irfftn(z_amp, s=(H, W), dim=(-2, -1))  # back to spatial, (B, C, H, W)
        return y.permute(0, 2, 3, 1)  # (B, H, W, C)

    def get_config(self):
        base = super().get_config()
        return {**base, "thresh_quantile": self.thresh_quantile}
