try:
    import keras
    from keras import layers
except ImportError:  # keras is optional: only the *_keras builders need it
    keras = layers = None
from torch import nn
from src.models.basic_residual_block import basic_block_keras, basic_block_torch


# torch
class resnet18_torch(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        # stem
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        # residual blocks [2, 2, 2, 2]
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(basic_block_torch(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        # stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # residual blocks 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # cls
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

         
# keras
def resnet18_keras(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # stem
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(x)

    # residual blocks [2, 2, 2, 2]
    x = basic_block_keras(x, 64, stride=1)
    x = basic_block_keras(x, 64, stride=1)

    x = basic_block_keras(x, 128, stride=2)
    x = basic_block_keras(x, 128, stride=1)

    x = basic_block_keras(x, 256, stride=2)
    x = basic_block_keras(x, 256, stride=1)

    x = basic_block_keras(x, 512, stride=2)
    x = basic_block_keras(x, 512, stride=1)

    # cls
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes)(x)

    return keras.Model(inputs, outputs)

            



