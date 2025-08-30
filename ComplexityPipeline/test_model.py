import torch
import numpy as np
import os
from pathlib import Path
from train_model import MultimodalRNN
import joblib

# 1. Load model components
model_path = 'D:/Data Eng/project/test_app/inference_model/best_vfx_model.pt'
scaler_path = 'D:/Data Eng/project/test_app/inference_model/static_scaler.pkl'
imputer_path = 'D:/Data Eng/project/test_app/inference_model/static_imputer.pkl'
features_dir = 'D:/Data Eng/project/test_app/features'

# Load checkpoint with safe_globals
import numpy
from torch.serialization import safe_globals
with safe_globals([numpy.core.multiarray.scalar]):
    checkpoint = torch.load(model_path, weights_only=False)
    
model = MultimodalRNN(checkpoint['config'] if 'config' in checkpoint else {})

# Handle different checkpoint formats
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
elif 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = {k: v for k, v in checkpoint.items() 
                 if not k.endswith(('_state_dict', 'config', 'epoch', 'loss', 'metrics'))}

# Remove 'module.' prefix if present (from DataParallel)
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
# Fix attention layer naming mismatch
state_dict = {k.replace('attention_attention_layer', 'attention_module.attention_layer'): v 
             for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()

scaler = joblib.load(scaler_path)
imputer = joblib.load(imputer_path)

# 2. Find test sequence files
test_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')][:3]  # Test first 3

# 3. Test function
def test_model():
    for seq_file in test_files:
        # Load sequence
        seq_path = os.path.join(features_dir, seq_file)
        sequence = np.load(seq_path).astype(np.float32)
        
        # Create mock static features (or modify to use real ones)
        static_features = np.array([1.0, 0.9, 0.5, 0.2, 0.1, 0.0, 0.8, 0.1, 0.8, 0.3])
        static_features = imputer.transform(static_features.reshape(1, -1))
        static_features = scaler.transform(static_features)
        
        # Predict
        with torch.no_grad():
            # Transform sequence to expected dimensions
            if sequence.shape[-1] != 2048:
                if sequence.shape[-1] == 512:
                    # Simple replication to match dimensions (temporary solution)
                    sequence = np.tile(sequence, (1, 4))
                else:
                    raise ValueError(f"Unexpected sequence dimension: {sequence.shape[-1]}")
                    
            seq_tensor = torch.tensor(sequence).unsqueeze(0).float()
            static_tensor = torch.tensor(static_features).float()
            
            output = model(seq_tensor, static_tensor)
            logits = output['logits'] if isinstance(output, dict) else output
            probs = torch.softmax(logits / 2.0, dim=1).numpy()[0]
            
        print(f"Testing: {seq_file}")
        print(f"Sequence shape: {sequence.shape}")
        print(f"Probabilities: {dict(zip(['Easy','Medium','Hard'], probs))}")
        print("-"*40)

if __name__ == "__main__":
    test_model()
