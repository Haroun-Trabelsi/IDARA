import torch
import numpy as np
from model_architecture import MultimodalRNN

# Load checkpoint
ckpt = torch.load('inference_model/best_fine_tuned_model.pt', map_location='cpu', weights_only=False)
print("Checkpoint keys:", list(ckpt.keys()))

# Create model
model = MultimodalRNN(ckpt.get('config', {}))
print("Model created successfully")

# Load weights
missing_keys, unexpected_keys = model.load_state_dict(ckpt['state_dict'], strict=False)
print(f"Missing keys: {len(missing_keys)}")
print(f"Unexpected keys: {len(unexpected_keys)}")
if missing_keys:
    print("Missing:", missing_keys[:5])
if unexpected_keys:
    print("Unexpected:", unexpected_keys[:5])

model.eval()

# Test with dummy data
dummy_seq = torch.randn(1, 32, 2048)
dummy_static = torch.randn(1, 10)

with torch.no_grad():
    output = model(dummy_seq, dummy_static)
    print("Raw logits:", output)
    probs = torch.softmax(output, dim=-1)
    print("Softmax probs:", probs)
    pred_class = torch.argmax(output, dim=-1).item()
    print("Predicted class index:", pred_class)
    
# Check if weights are actually loaded by examining a few parameters
print("\nSample model parameters:")
for name, param in list(model.named_parameters())[:3]:
    print(f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
