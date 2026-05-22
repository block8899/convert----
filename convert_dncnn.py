import torch
import torch.nn as nn
import pnnx
import os
import sys
import gc
import urllib.request

# ═══════════════════════════════════════════════════
# DnCNN Architecture (17 layers, color denoise)
# ═══════════════════════════════════════════════════

class DnCNN(nn.Module):
    def __init__(self, channels=3, num_layers=17, features=64):
        super(DnCNN, self).__init__()
        layers = []
        layers.append(nn.Conv2d(channels, features, 3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(features, features, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(features, channels, 3, padding=1, bias=False))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        noise = self.network(x)
        return torch.clamp(x - noise, 0, 1)

print("1. Creating DnCNN model...")
model = DnCNN(channels=3, num_layers=17, features=64)
model.eval()
torch.set_grad_enabled(False)

params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {params:,}")

print("2. Downloading pretrained weights...")
WEIGHT_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth"
WEIGHT_PATH = "dncnn_color_blind.pth"

if not os.path.exists(WEIGHT_PATH):
    print("   Downloading...")
    urllib.request.urlretrieve(WEIGHT_URL, WEIGHT_PATH)
    print(f"   Done: {os.path.getsize(WEIGHT_PATH)/1024/1024:.1f} MB")
else:
    print(f"   Already exists: {os.path.getsize(WEIGHT_PATH)/1024/1024:.1f} MB")

print("   Loading weights...")
ckpt = torch.load(WEIGHT_PATH, map_location="cpu", weights_only=False)

# KAIR saves as dict with 'params' key
if isinstance(ckpt, dict):
    if 'params' in ckpt:
        state_dict = ckpt['params']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    elif 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# Handle 'module.' prefix from DataParallel
new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace('module.', '')
    new_state_dict[name] = v

model.load_state_dict(new_state_dict, strict=True)
print("   Weights loaded OK!")

print("3. Converting to NCNN via PNNX...")
dummy = torch.randn(1, 3, 256, 256)

try:
    pnnx.export(model, "dncnn", inputs=dummy)
    print("   Done!")
except Exception as e:
    print(f"   PNNX failed: {e}")
    sys.exit(1)

del model, dummy
gc.collect()

print("4. Verifying...")
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
