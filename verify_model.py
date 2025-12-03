#!/usr/bin/env python3
import torch
import os

model_path = r'c:\Users\user\Desktop\STiTy-github\SimulStreaming\large-v2.pt'
print(f'File size: {os.path.getsize(model_path) / 1e9:.2f} GB')

try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f'Checkpoint type: {type(checkpoint)}')
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        print(f'Number of keys: {len(keys)}')
        print(f'First 10 keys: {keys[:10]}')
        print('Model checkpoint appears to be valid')
    else:
        print('Warning: Checkpoint is not a dict')
except Exception as e:
    print(f'Error loading checkpoint: {e}')
    import traceback
    traceback.print_exc()
