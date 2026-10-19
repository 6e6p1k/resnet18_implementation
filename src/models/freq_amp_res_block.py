from torch import nn

from src.models.freq_amp_layer import freq_amp_layer_keras, freq_amp_layer_torch

try:
    from keras import layers
except ImportError:  # keras is optional: only freq_amp_block_keras needs it
    layers = None


class freq_amp_block_torch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 amp_factor=2,
                 thresh_quantile=0.3):
        
        super().__init__()

        # main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.freq_amp = freq_amp_layer_torch(amp_factor=amp_factor, thresh_quantile=thresh_quantile)

        # shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        shortcut = self.shortcut(x)

        # amplify weak frequencies
        shortcut = self.freq_amp(shortcut)

        # merge
        out.add_(shortcut)
        out = self.relu(out)
        return out
    

def freq_amp_block_keras(x, out_channels, stride=1, amp_factor=2.0, thresh_quantile=0.3):
    in_channels = x.shape[-1]
    shortcut = x

    # main path
    x = layers.Conv2D(out_channels, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(out_channels, 3, strides=1, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    # shortcut path
    if stride != 1 or in_channels != out_channels:
        shortcut = layers.Conv2D(out_channels, 1, strides=stride, padding="valid", use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # amplify weak frequencies
    shortcut = freq_amp_layer_keras(amp_factor=amp_factor, thresh_quantile=thresh_quantile)(shortcut)

    # merge
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x