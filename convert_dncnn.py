import torch
import torch.nn as nn
import pnnx
import os
import sys
import gc

# ═══════════════════════════════════════════════════
# DnCNN Architecture (17 layers)
# Zhang et al. "Beyond a Gaussian Denoiser"
# ═══════════════════════════════════════════════════

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(channels, features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        # Middle layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        # Last layer: Conv (no activation)
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # Residual learning: output = input - noise
        noise = self.network(x)
        return torch.clamp(x - noise, 0, 1)

print("1. Creating DnCNN model...")
model = DnCNN(channels=3, num_layers=17, features=64)
model.eval()
torch.set_grad_enabled(False)

params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")

print("2. Converting to NCNN via PNNX...")
dummy = torch.randn(1, 3, 256, 256)

try:
    pnnx.export(model, "dncnn", inputs=dummy)
    print("   Done!")
except Exception as e:
    print(f"   PNNX failed: {e}")
    sys.exit(1)

del model, dummy
gc.collect()

print("3. Verifying...")
pf = "dncnn.ncnn.param"
bf = "dncnn.ncnn.bin"

if os.path.exists(pf) and os.path.exists(bf):
    sp = os.path.getsize(pf) / 1024
    sb = os.path.getsize(bf) / 1024
    print(f"  {pf}: {sp:.1f} KB")
    print(f"  {bf}: {sb:.1f} KB")

    with open(pf, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("Input"):
            parts = line.split()
            if len(parts) >= 3:
                print(f"  Input blob: {parts[2]}")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("#") and not line.startswith("7767517"):
            parts = line.split()
            if len(parts) >= 4:
                print(f"  Output blob: {parts[-1]}")
                break

    total_kb = sp + sb
    print(f"\nTotal: {total_kb:.0f} KB ({total_kb/1024:.1f} MB)")
    print("DnCNN NCNN OK!")
else:
    print("FAILED!")
    print(f"Files: {os.listdir('.')}")
    sys.exit(1)
