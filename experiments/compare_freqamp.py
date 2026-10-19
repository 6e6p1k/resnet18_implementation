#!/usr/bin/env python3
"""Plain ResNet18 vs ResNet18 with FreqAmp on the residual shortcut, on CIFAR-10.

Both models come from src/models and are trained under one seed with the same
initial weights (FreqAmp has no parameters), data order, augmentation,
optimizer and schedule. The result is a side-by-side table.

The earlier notebook comparison (notebooks/ResNet18_with_FreqAmpBlocks.ipynb)
was invalid: a typo (`itentity = self.freq_amp(identity)`) discarded the
FreqAmp output in one model and the other model never called FreqAmp at all,
so both runs were plain ResNet18. This script instantiates the two classes
explicitly and prints how many FreqAmp layers each model really contains.

Smoke run on CPU (about a minute):
    python experiments/compare_freqamp.py --epochs 1 --train-samples 1024 --test-samples 512

Real run (GPU), then repeat with other seeds:
    python experiments/compare_freqamp.py --epochs 30 --seed 0
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn, optim
import torchvision
import torchvision.transforms as T

from src.models.resnet18 import resnet18_torch
from src.models.amp_freq_resnet18 import amp_freq_resnet18_torch
from src.models.freq_amp_layer import freq_amp_layer_torch

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--train-samples", type=int, default=0,
                   help="use only the first N training images (0 = all 50k)")
    p.add_argument("--test-samples", type=int, default=0,
                   help="use only the first N test images (0 = all 10k)")
    p.add_argument("--models", nargs="+", default=["plain", "freqamp"],
                   choices=["plain", "freqamp"])
    p.add_argument("--amp-factors", type=float, nargs=4, default=[1.5, 1.5, 1.5, 1.5],
                   metavar="A", help="FreqAmp amp_factor per stage (layer1..layer4)")
    p.add_argument("--thresh-quantiles", type=float, nargs=4, default=[0.4, 0.4, 0.4, 0.4],
                   metavar="Q", help="FreqAmp thresh_quantile per stage (layer1..layer4)")
    p.add_argument("--stem", choices=["cifar", "imagenet"], default="cifar",
                   help="cifar: 3x3/1 conv and no maxpool on both models, so 32x32 inputs "
                        "keep 32->16->8->4 feature maps; imagenet: the 7x7/2 conv + maxpool "
                        "as written in src/ (maps 7->4->2->1; FreqAmp on a 1x1 map is a no-op)")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw"))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=str(ROOT / "runs" / "compare_freqamp.json"),
                   help="where to write the JSON results ('' to skip)")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_loaders(args, generator):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    train_set = torchvision.datasets.CIFAR10(args.data_dir, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(args.data_dir, train=False, download=True, transform=test_tf)
    if args.train_samples:
        train_set = torch.utils.data.Subset(train_set, range(min(args.train_samples, len(train_set))))
    if args.test_samples:
        test_set = torch.utils.data.Subset(test_set, range(min(args.test_samples, len(test_set))))
    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=pin, generator=generator)  # generator fixes the shuffle order and worker seeds
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=256, shuffle=False, num_workers=args.num_workers, pin_memory=pin)
    return train_loader, test_loader


def build_model(name, args):
    if name == "plain":
        model = resnet18_torch(num_classes=10)
    elif name == "freqamp":
        model = amp_freq_resnet18_torch(num_classes=10, amp_factors=list(args.amp_factors),
                                        thresh_quantiles=list(args.thresh_quantiles))
    else:
        raise ValueError(f"unknown model {name!r}")
    if args.stem == "cifar":
        # Same swap on both models, applied after construction so the RNG
        # draws for every other layer line up and both start from equal weights.
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    return model


def count_freq_amp(model):
    return sum(isinstance(m, freq_amp_layer_torch) for m in model.modules())


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        correct += (model(images).argmax(1) == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def run(name, args, device):
    seed_everything(args.seed)
    model = build_model(name, args).to(device)
    train_loader, test_loader = get_loaders(args, torch.Generator().manual_seed(args.seed))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    n_params = sum(p.numel() for p in model.parameters())
    n_amp = count_freq_amp(model)
    print(f"\n=== {name}: {n_params / 1e6:.2f}M params, {n_amp} FreqAmp layers, stem={args.stem} ===")
    history, best, t0 = [], 0.0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = correct = total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        train_loss, train_acc = loss_sum / total, 100.0 * correct / total
        test_acc = evaluate(model, test_loader, device)
        best = max(best, test_acc)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "train_acc": train_acc, "test_acc": test_acc})
        print(f"[{name}] epoch {epoch:3d}/{args.epochs}  loss {train_loss:.3f}  "
              f"train {train_acc:6.2f}%  test {test_acc:6.2f}%  best {best:6.2f}%")
    return {"model": name, "params": n_params, "freq_amp_layers": n_amp,
            "best_test_acc": best, "final_test_acc": history[-1]["test_acc"],
            "final_train_acc": history[-1]["train_acc"],
            "seconds": time.time() - t0, "history": history}


def print_table(results, args):
    cols = ["model", "params", "FreqAmp layers", "best test %", "final test %", "final train %", "time"]
    rows = [[r["model"], f"{r['params'] / 1e6:.2f}M", str(r["freq_amp_layers"]),
             f"{r['best_test_acc']:.2f}", f"{r['final_test_acc']:.2f}",
             f"{r['final_train_acc']:.2f}", f"{r['seconds'] / 60:.1f} min"] for r in results]
    widths = [max(len(c), *(len(row[i]) for row in rows)) for i, c in enumerate(cols)]
    fmt = lambda cells: "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"
    print(f"\nCIFAR-10, seed {args.seed}, {args.epochs} epochs, bs {args.batch_size}, lr {args.lr}, "
          f"stem {args.stem}, amp_factors {args.amp_factors}, thresh_quantiles {args.thresh_quantiles}")
    print(fmt(cols))
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print(fmt(row))
    if len(results) == 2:
        delta = results[1]["best_test_acc"] - results[0]["best_test_acc"]
        print(f"\n{results[1]['model']} minus {results[0]['model']}: {delta:+.2f} points (best test). "
              "This is one seed: differences under about half a point are noise. "
              "Rerun with several --seed values before concluding anything.")


def main():
    args = parse_args()
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu")
                          if args.device == "auto" else args.device)
    print(f"device: {device}, torch {torch.__version__}")
    results = [run(name, args, device) for name in args.models]
    print_table(results, args)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"config": vars(args), "results": results}, indent=2))
        print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
