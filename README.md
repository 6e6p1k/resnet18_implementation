# resnet18_implementation

ResNet18 written from scratch in PyTorch and in Keras (torch backend), plus two experimental frequency-domain layers built on `torch.fft`: **FreqAmp**, which amplifies weak spectral coefficients on the residual shortcut, and **fReLU**, a learnable soft-threshold on FFT magnitude tried as an activation. The experiments live in the notebooks; `src/` holds the reusable modules.

## What is implemented

| Path | What |
|---|---|
| `src/models/basic_residual_block.py` | Basic residual block: torch class `basic_block_torch`, Keras builder `basic_block_keras`. |
| `src/models/resnet18.py` | ResNet18 [2,2,2,2] with an ImageNet-style stem (7x7/2 conv + maxpool): `resnet18_torch`, `resnet18_keras`. |
| `src/models/freq_amp_layer.py` | FreqAmp: `freq_amp()` function, `freq_amp_layer_torch`, `freq_amp_layer_keras` (torch backend only). |
| `src/models/freq_amp_res_block.py` | Residual block with FreqAmp on the shortcut branch, torch and Keras. |
| `src/models/amp_freq_resnet18.py` | ResNet18 built from FreqAmp blocks, with `amp_factors` and `thresh_quantiles` per stage. |
| `src/torch_to_keras_layer.py` | Wraps any `nn.Module` as a Keras layer (torch backend). Not used elsewhere. |
| `experiments/compare_freqamp.py` | Seeded CIFAR-10 comparison, plain ResNet18 vs FreqAmp-on-shortcut, side-by-side table. |
| `tests/test_models.py` | Forward shapes, skip path active, FreqAmp shape / identity / boost checks. |
| `notebooks/ResNet18_with_FreqAmpBlocks.ipynb` | CIFAR-100 baseline (120 epochs) and the old, invalid FreqAmp comparison. |
| `notebooks/fReLU.ipynb`, `notebooks/fReLU_(attempt_2).ipynb` | fReLU definition and CIFAR-10 runs against ReLU. |
| `notebooks/test.ipynb` | Tiny ImageNet attempt with the Keras FreqAmp ResNet18. Never completed. |

## Recorded results

All single runs, no repeated seeds. Numbers are read from the notebook outputs.

| Experiment | Setup | Result |
|---|---|---|
| ResNet18 baseline, CIFAR-100 | notebook `ResNet18` (3x3/1 stem), 120 epochs, AdamW lr 1e-3, cosine, bs 256, label smoothing 0.1 | best test **74.55%** (epoch 103), final 74.26%, train ~100% |
| fReLU vs ReLU, CIFAR-10, 10 epochs | small CNN, Adam, `fReLU.ipynb` | ReLU **76.2%**; fReLU 73.2% (fixed λ), 72.6% (learnable λ) |
| fReLU vs ReLU, CIFAR-10, 40 epochs | `TinyConvNet`, AdamW + cosine, `fReLU_(attempt_2).ipynb` | ReLU **89.8%**; fReLU 75.8% |
| FreqAmp vs plain, CIFAR-100 | `ResNet18_with_FreqAmpBlocks.ipynb` | **Invalid.** The typo `itentity = self.freq_amp(identity)` discards FreqAmp in one model and the other never calls it, so both runs were plain ResNet18; the second was stopped at epoch 55 (72.71%). |
| FreqAmp vs plain, CIFAR-10, 40 epochs | mini-ResNet, `fReLU_(attempt_2).ipynb` | 90.58% vs 90.52% final, but the "control" `MiniResNet` also instantiates `FreqAmpResBlock`, so this is not a comparison either. |
| Tiny ImageNet | Keras `amp_freq_resnet18_keras`, 64x64, 200 classes, `test.ipynb` | Interrupted at step 118/782 of epoch 1. No result. Do not cite. |

So: the CIFAR-100 baseline is real, fReLU is a clean negative result, and FreqAmp has never been validly tested. **The fair FreqAmp comparison is pending:** `experiments/compare_freqamp.py` is the rerun; it is included but has not been run yet (it needs a GPU), so this repository makes no claim about whether FreqAmp helps.

The saved outputs in `ResNet18_with_FreqAmpBlocks.ipynb` were produced before the `itentity` typo was corrected in the cell source (it now reads `identity = self.freq_amp(identity)`), so they still reflect the invalid runs. The typo sat in the notebook's `BasicBlock`; its separate `FreqAmpBlock` class never calls FreqAmp at all, and that is not changed here.

## The two layers

**FreqAmp** (`src/models/freq_amp_layer.py`). For each sample and channel take the 2-D real FFT `Z = rFFT2(x)`, the magnitudes `m = |Z|` and the threshold `τ = quantile_q(m)`, then scale `Z' = Z · (1 + (α − 1) · 1[m < τ])` and return `y = irFFT2(Z')` at the original spatial size. It has no parameters, the mask is data-dependent (so the layer is non-linear), and it sits on the shortcut branch before the residual add, with `α` and `q` chosen per stage.

**fReLU** (notebooks only). Soft-shrinkage of the spectrum magnitude with the phase kept: `Z = FFT2(x)`, `Z' = max(|Z| − λ, 0) · Z / (|Z| + ε)`, `y = Re(iFFT2(Z'))`, with `λ ≥ 0` learnable (one scalar; a per-channel variant is defined but was never trained). Used in place of ReLU after every conv it lowered CIFAR-10 accuracy in both runs above. An earlier element-wise `max(Re Z, t)` version is commented out in `fReLU.ipynb`.

## How to run

```bash
pip install -r requirements.txt              # torch, torchvision, numpy; keras is optional
python -m unittest discover -s tests -v      # 9 tests

# smoke test on CPU (downloads CIFAR-10 to data/raw; not yet run)
python experiments/compare_freqamp.py --epochs 1 --train-samples 1024 --test-samples 512

# the real comparison (GPU), then repeat with other seeds
python experiments/compare_freqamp.py --epochs 30 --seed 0
python experiments/compare_freqamp.py --epochs 30 --seed 1
```

The script seeds everything, trains both models from identical initial weights on identical data order and augmentation, prints a table (params, number of FreqAmp layers, best / final test accuracy, time) and writes `runs/compare_freqamp.json`. `--stem cifar` (default) swaps the 7x7/2 stem for a 3x3/1 stem on both models so 32x32 inputs keep 32→16→8→4 feature maps; `--stem imagenet` uses the stem as written in `src/` (maps 7→4→2→1, and FreqAmp on a 1x1 map is a no-op). `--amp-factors` and `--thresh-quantiles` take one value per stage. The Keras builders need `KERAS_BACKEND=torch`.

## Limitations

- The 9 unit tests pass on CPU (torch 2.14.1, Python 3.14). The Keras builders and `freq_amp_layer_keras` are not covered by them and have not been run.
- `experiments/compare_freqamp.py` has not been run end to end: data loading, training and the results table are untested.
- Every recorded number is one run; differences below about half a point are noise.
- The notebooks' "validation" sets are the test sets.
- Tiny ImageNet was never trained.
- `torch.quantile` makes FreqAmp slow on large maps and raises on inputs above 2^24 elements.
- The torch 7x7 stem uses `padding=1`; standard ResNet18 and the Keras version (`padding="same"`) use 3.
- `notebooks/test.ipynb` needs `mltoolkit`, which is not in this repo; `fReLU.ipynb` calls an undefined `Net`, so the 10-epoch fReLU runs cannot be re-executed as saved.
- No spectral analysis of learned filters and no convolution-theorem demo exist yet, despite the FFT layers.

## Next steps

1. Run `compare_freqamp.py` for three seeds and record the table here. If FreqAmp does not beat plain ResNet18 by more than the seed spread, record that as the result.
2. Move fReLU into `src/` and test it in one place (e.g. only on the shortcut) instead of after every conv.
3. Plot the magnitude spectra of the learned first-layer filters, with and without FreqAmp.
4. Fix the stem padding, or make the CIFAR stem a constructor option instead of a script-side swap.
