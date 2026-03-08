from torch import nn
import keras
from keras import layers

# torch
class basic_block_torch(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        
        # main path
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

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

        # merge 
        out.add(self.shortcut(x))
        out = self.relu(out)

        return out
    

# keras
def basic_block_keras(x, out_channels, stride=1):
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

    # merge
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)

    return x
