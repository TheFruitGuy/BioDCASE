# canary_final_load.py — confirm whether final_best.pt loads cleanly into the HNM trainer's pipeline
import torch, numpy as np
import config as cfg
cfg.USE_3CLASS = False   # mimic the trainer's auto-override

from spectrogram import SpectrogramExtractor
from ensemble_predict import build_model_for_ckpt

device = torch.device("cuda:0")
ckpt = torch.load("runs/final_20260524_171403/final_best.pt",
                  map_location=device, weights_only=False)
print(f"ckpt keys: {list(ckpt.keys())}")
print(f"state_dict size: {len(ckpt['model_state_dict'])}")

model, mt = build_model_for_ckpt(ckpt, device)
spec = SpectrogramExtractor().to(device)
with torch.no_grad():
    _ = model(spec(torch.randn(1, cfg.SAMPLE_RATE * 30, device=device)))

missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
print(f"\nMISSING ({len(missing)}):")
for k in missing: print(" ", k)
print(f"\nUNEXPECTED ({len(unexpected)}):")
for k in unexpected: print(" ", k)

# Classifier head sanity (per output channel)
clf = model.classifier
print(f"\nclassifier.weight per-channel L2 norm:")
for i, n in enumerate(cfg.CALL_TYPES_7):
    print(f"  {i} {n:8} {clf.weight[i].norm().item():.4f}")
print(f"classifier.bias:")
for i, n in enumerate(cfg.CALL_TYPES_7):
    print(f"  {i} {n:8} {clf.bias[i].item():+.4f}")

# Forward pass on a fixed dummy
model.eval()
with torch.no_grad():
    audio = torch.randn(4, cfg.SAMPLE_RATE * 30, device=device, generator=torch.Generator(device).manual_seed(0)) * 0.1
    out = model(spec(audio))
    probs = torch.sigmoid(out)
    print(f"\nper-channel probs over {probs.numel()//7} frames:")
    print(f"  max:  {probs.amax(dim=(0,1)).cpu().tolist()}")
    print(f"  mean: {probs.mean(dim=(0,1)).cpu().tolist()}")