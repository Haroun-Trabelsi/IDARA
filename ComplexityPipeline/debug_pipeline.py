import numpy as np
import torch
from pathlib import Path
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

class_names = config['multimodal_model']['class_names']
print("Class names from config:", class_names)

# Load the actual features from the test file
features_path = Path("features/022ac208_ep103_scn0044_sh0010-easy.npy")
if features_path.exists():
    sequence_features = np.load(features_path)
    print(f"Loaded sequence features shape: {sequence_features.shape}")
    
    # Test static features (from your log)
    complexity_scores = {
        'zoom_score': 1.0100737810134888,
        'blur_score': 0.9991365075111389,
        'distortion_score': 0.5610195398330688,
        'motion_score': 0.20575301349163055,
        'light_score': 0.1516227126121521,
        'noise_score': 0.000400502176489681,
        'overlap_score': 0.915795087814331,
        'parallax_score': 0.06096377223730087,
        'focus_pull_score': 0.8220421671867371
    }
    
    # Calculate sequence_mean (10th static feature)
    sequence_mean = np.mean(sequence_features)
    print(f"Sequence mean: {sequence_mean}")
    
    # Build static features array
    static_features = [
        complexity_scores['zoom_score'],
        complexity_scores['blur_score'], 
        complexity_scores['distortion_score'],
        complexity_scores['motion_score'],
        complexity_scores['light_score'],
        complexity_scores['noise_score'],
        complexity_scores['overlap_score'],
        complexity_scores['parallax_score'],
        complexity_scores['focus_pull_score'],
        sequence_mean
    ]
    
    print("Static features:", static_features)
    
    # Load model and test
    from model_architecture import MultimodalRNN
    ckpt = torch.load('inference_model/best_fine_tuned_model.pt', map_location='cpu', weights_only=False)
    model = MultimodalRNN(ckpt.get('config', {}))
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    # Run inference
    with torch.no_grad():
        seq_tensor = torch.tensor(sequence_features, dtype=torch.float32).unsqueeze(0)
        static_tensor = torch.tensor(static_features, dtype=torch.float32).unsqueeze(0)
        
        print(f"Input shapes - seq: {seq_tensor.shape}, static: {static_tensor.shape}")
        
        output = model(seq_tensor, static_tensor)
        print("Raw logits:", output)
        
        probs = torch.softmax(output, dim=-1).numpy()[0]
        print("Probabilities:", probs)
        
        pred_index = np.argmax(probs)
        predicted_class = class_names[pred_index]
        
        print(f"Predicted index: {pred_index}")
        print(f"Predicted class: {predicted_class}")
        print(f"Expected class: Easy (from filename)")
        
        # Check if this matches what the pipeline would output
        result = {
            'predicted_class': predicted_class,
            'confidence': f"{probs[pred_index]:.2%}",
            'probabilities': {name: f"{prob:.2%}" for name, prob in zip(class_names, probs)}
        }
        print("Final result:", result)
        
else:
    print("Features file not found!")
