# resnet18_implementation

Experimenting with ResNet18's architecture: ResNet18 from scratch in PyTorch and Keras, and FreqAmp, a layer that rescales the weakest FFT coefficients of the residual shortcut.

## Results

CIFAR-100, ResNet18 with a 3x3 stem, 120 epochs, AdamW (lr 1e-3, weight decay 5e-4), cosine schedule, batch 256, label smoothing 0.1 (`notebooks/ResNet18_with_FreqAmpBlocks.ipynb`):

| model | best test acc | final test acc |
|---|---|---|
| ResNet18 | 74.55% (epoch 103) | 74.26% |

Tiny ImageNet (64x64, 200 classes), Keras on the torch backend, 30 epochs, AdamW (lr 1e-3), no schedule, batch 128, seed 0, RTX 5070, one run each. FreqAmp rescales the weakest 20 / 40 / 60 / 80% of coefficients in stages 1 to 4:

| model | best val acc | final val acc | notebook |
|---|---|---|---|
| ResNet18 | 42.15% (epoch 18) | 40.59% | `test.ipynb` |
| FreqAmp ResNet18, x8 / x6 / x4 / x2 | 37.07% (epoch 21) | 34.97% | `test.ipynb` |
| FreqAmp ResNet18, x0.8 / x0.6 / x0.4 / x0.2 | 42.32% (epoch 16) | 40.55% | `test_suppress.ipynb` |
| FreqAmp ResNet18, learnable per-channel gain starting at 1 | 43.33% (epoch 14) | 40.47% | `test_learnable.ipynb` |

Boosting the weak (mostly high) frequencies on the shortcut costs about 5 points; shrinking them, or letting the network learn the gain, lands at the baseline, and the learned gains end up mostly below 1.
