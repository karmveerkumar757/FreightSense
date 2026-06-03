import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import torch
import spacy

print("⚡ Testing Hardware Integration Status...\n")

# 1. Verify PyTorch CUDA connection
pytorch_gpu = torch.cuda.is_available()
print(f"📦 PyTorch CUDA Live: {pytorch_gpu}")
if pytorch_gpu:
    print(f"   -> Connected Device: {torch.cuda.get_device_name(0)}")

# 2. Verify spaCy CUDA connection
try:
    spacy_gpu = spacy.require_gpu()
except Exception as e:
    spacy_gpu = False
    print(f"⚠️ spaCy GPU Activation Error: {e}")

print(f"\n🏷️ spaCy GPU Activated: {spacy_gpu}")

if pytorch_gpu and spacy_gpu:
    print("\n🎉 Success! Your RTX 2050 is completely configured for local deep learning!")
else:
    print("\n⚠️ Check installation paths. One or both frameworks are falling back to CPU.")